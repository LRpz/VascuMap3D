import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.ndimage.morphology import distance_transform_edt as edt
from .skeletonization import measure_edge_length

def compute_cross_sectional_areas(mask, skeleton_image, binary_edt):
    """
    Compute local cross-sectional areas of vessels at skeleton points.

    Approximates area as an ellipse: Area = pi * (major_axis/2) * (minor_axis/2).
    Minor axis is derived from local 3D EDT (radius).
    Major axis is derived from 2D projection EDT (radius).

    NOTE: this is a fast approximation. The major axis comes from the z-MAX-projection, so
    vessels overlapping in projection are merged and the area is over-estimated in dense
    regions. For a projection-free measurement use compute_cross_sectional_areas_exact.

    Args:
        mask (np.ndarray): Binary vessel mask.
        skeleton_image (np.ndarray): Binary skeleton image.
        binary_edt (np.ndarray): 3D Euclidean Distance Transform.

    Returns:
        np.ndarray: 3D volume with area values at skeleton points.
    """
    edt_2d = edt(np.max(mask, axis=0))
    area_3d = np.zeros_like(binary_edt, dtype=float)
    z_idx, y_idx, x_idx = np.where(skeleton_image > 0)

    minor_axis = 2 * binary_edt[z_idx, y_idx, x_idx]
    major_axis = 2 * edt_2d[y_idx, x_idx]

    areas = np.pi * (major_axis/2) * (minor_axis/2)
    area_3d[z_idx, y_idx, x_idx] = areas
    return area_3d

def compute_cross_sectional_areas_exact(mask, skeleton_image, chunk_size=None, overlap=128):
    """
    Local vessel volume carried by each skeleton voxel via a nearest-centerline (Voronoi)
    partition of the foreground -- the basis for an exact cross-sectional area/volume.

    Every foreground voxel is assigned, in full 3D, to its nearest skeleton voxel using a
    Euclidean feature transform. The number of voxels assigned to a skeleton voxel equals the
    vessel volume (voxel^3) carried by that centerline point. Dividing by the local axial
    spacing yields the true perpendicular cross-sectional area; summing along a centerline
    yields the vessel's volume exactly.

    Unlike compute_cross_sectional_areas (which takes the major axis from a z-projection and
    therefore merges vessels overlapping in projection), the 3D assignment keeps overlapping
    vessels separate, so it stays reliable in dense networks.

    Computed chunked on the GPU (cupy). Overlap must exceed the largest vessel radius so a
    foreground voxel's nearest centerline is never in an unseen neighbouring chunk (same
    requirement as the EDT and skeletonization). chunk_size/overlap can be reduced to bound
    GPU memory on very large volumes.

    Args:
        mask (np.ndarray): Binary vessel mask (foreground > 0).
        skeleton_image (np.ndarray): Binary skeleton (centerline) image.
        chunk_size (tuple, optional): (Z, Y, X) chunk size. Defaults to (Z, 512, 512).
        overlap (int or tuple, optional): Halo, in voxels, around each chunk. A scalar is
            applied to all axes; a (z, y, x) tuple sets it per axis. Clamped per axis so it
            never exceeds the volume/chunk (e.g. thin-Z volumes). Defaults to 128.

    Returns:
        np.ndarray: Volume (voxel^3) carried by each skeleton voxel; zero elsewhere.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cp_edt

    shape = mask.shape
    if chunk_size is None:
        chunk_size = (shape[0], 512, 512)
    if np.isscalar(overlap):
        overlap = (overlap, overlap, overlap)
    ov = tuple(int(min(overlap[i], shape[i], chunk_size[i])) for i in range(3))

    local_volume = np.zeros(shape, dtype=np.float32)
    pool = cp.get_default_memory_pool()

    for z in range(0, shape[0], chunk_size[0]):
        for y in range(0, shape[1], chunk_size[1]):
            for x in range(0, shape[2], chunk_size[2]):
                z0, z1 = max(0, z - ov[0]), min(shape[0], z + chunk_size[0] + ov[0])
                y0, y1 = max(0, y - ov[1]), min(shape[1], y + chunk_size[1] + ov[1])
                x0, x1 = max(0, x - ov[2]), min(shape[2], x + chunk_size[2] + ov[2])

                skel_chunk = cp.asarray(skeleton_image[z0:z1, y0:y1, x0:x1] > 0)
                if not bool(skel_chunk.any()):
                    del skel_chunk
                    pool.free_all_blocks()
                    continue
                mask_chunk = cp.asarray(mask[z0:z1, y0:y1, x0:x1] > 0)

                # nearest skeleton voxel (in local chunk coords) for every voxel: distance is
                # to the nearest zero, so zeros must sit ON the skeleton -> invert the mask
                res = cp_edt(~skel_chunk, return_distances=False, return_indices=True)
                inds = res[0] if isinstance(res, tuple) else res

                # inner (non-overlap) write region within the chunk
                wz0, wz1 = z - z0, (z - z0) + min(chunk_size[0], shape[0] - z)
                wy0, wy1 = y - y0, (y - y0) + min(chunk_size[1], shape[1] - y)
                wx0, wx1 = x - x0, (x - x0) + min(chunk_size[2], shape[2] - x)

                fg = mask_chunk[wz0:wz1, wy0:wy1, wx0:wx1]
                if not bool(fg.any()):
                    del skel_chunk, mask_chunk, res, inds
                    pool.free_all_blocks()
                    continue

                # global skeleton-voxel coords for each inner foreground voxel
                gz = cp.asnumpy(inds[0][wz0:wz1, wy0:wy1, wx0:wx1][fg]) + z0
                gy = cp.asnumpy(inds[1][wz0:wz1, wy0:wy1, wx0:wx1][fg]) + y0
                gx = cp.asnumpy(inds[2][wz0:wz1, wy0:wy1, wx0:wx1][fg]) + x0
                np.add.at(local_volume, (gz, gy, gx), 1.0)

                del skel_chunk, mask_chunk, res, inds, fg, gz, gy, gx
                pool.free_all_blocks()

    return local_volume

def shortest_path(coordinates):
    """Calculate Euclidean distance between first and last points."""
    return np.linalg.norm(coordinates[0] - coordinates[-1])

def compute_vessel_metrics(graph, area_image, vessel_metrics_df, area_image_exact=None, voxel_size_um=2.0):
    """
    Compute geometric and topological metrics for each vessel segment (edge).

    Metrics include: vessel_length, tortuosity, vessel_volume, cross-sectional area
    (mean, median, std), orientation, connectivity. Lengths are returned in um, areas in
    um^2 and volumes in um^3 (via voxel_size_um). vessel_volume is the sum of the
    per-centerline-voxel cross-sectional areas (an approximation assuming ~1 voxel axial
    spacing, biased low on oblique runs); vessel_volume_exact (when area_image_exact is
    provided) is unbiased.

    Args:
        graph (nx.Graph): Input graph.
        area_image (np.ndarray): 3D volume with cross-sectional area values (voxel^2).
        vessel_metrics_df (pd.DataFrame): (Unused in this function, returns new DF).
        area_image_exact (np.ndarray, optional): 3D volume of local vessel volume per
            centerline voxel (from compute_cross_sectional_areas_exact). When given, adds
            *_cross_sectional_area_exact and vessel_volume_exact columns. Defaults to None.
        voxel_size_um (float, optional): Isotropic voxel size in um for unit conversion.
            Defaults to 2.0.

    Returns:
        pd.DataFrame: DataFrame containing metrics for each edge.
    """
    zs, ys, xs = [], [], []
    vessel_volumes, vessel_lengths, end_to_end_distances = [], [], []
    tortuosities, is_sprouts = [], []
    mean_cross_sectional_areas, median_cross_sectional_areas, std_cross_sectional_areas = [], [], []
    node1_degrees, node2_degrees = [], []
    orientation_zs, orientation_ys, orientation_xs = [], [], []

    voxel_area_um2 = voxel_size_um ** 2
    voxel_volume_um3 = voxel_size_um ** 3
    mean_cross_sectional_areas_exact, median_cross_sectional_areas_exact = [], []
    std_cross_sectional_areas_exact, vessel_volumes_exact = [], []

    edges = list(graph.edges())
    if not edges:
        cols = ['z', 'y', 'x', 'vessel_volume', 'vessel_length', 'end_to_end_distance',
                'tortuosity', 'is_sprout', 'mean_cross_sectional_area',
                'median_cross_sectional_area', 'std_cross_sectional_area',
                'node1_degree', 'node2_degree', 'orientation_z', 'orientation_y', 'orientation_x']
        return pd.DataFrame(columns=cols)

    node_degrees_dict = dict(graph.degree())

    for u, v in edges:
        try:
            pts = graph[u][v]['pts']
            if len(pts) < 2:
                continue

            zs.append(pts[:, 0])
            ys.append(pts[:, 1])
            xs.append(pts[:, 2])

            segment_areas = area_image[pts[:, 0], pts[:, 1], pts[:, 2]]
            mean_cross_sectional_areas.append(np.nanmean(segment_areas) * voxel_area_um2)
            median_cross_sectional_areas.append(np.nanmedian(segment_areas) * voxel_area_um2)
            std_cross_sectional_areas.append(np.nanstd(segment_areas) * voxel_area_um2)

            # NOTE: approximate volume = sum of per-centerline-voxel areas (assumes ~1 voxel
            # axial spacing); biased low on oblique runs. See vessel_volume_exact below.
            vessel_volumes.append(np.nansum(segment_areas) * voxel_volume_um3)

            if area_image_exact is not None:
                # exact local volume (voxel^3) per centerline voxel, divided by the local
                # axial spacing -> true perpendicular cross-sectional area (voxel^2)
                exact_local_vol = area_image_exact[pts[:, 0], pts[:, 1], pts[:, 2]]
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                step = np.empty(len(pts))
                step[0], step[-1] = seg[0], seg[-1]
                if len(pts) > 2:
                    step[1:-1] = (seg[:-1] + seg[1:]) / 2.0
                step = np.clip(step, 1e-6, None)
                exact_area = exact_local_vol / step
                mean_cross_sectional_areas_exact.append(np.nanmean(exact_area) * voxel_area_um2)
                median_cross_sectional_areas_exact.append(np.nanmedian(exact_area) * voxel_area_um2)
                std_cross_sectional_areas_exact.append(np.nanstd(exact_area) * voxel_area_um2)
                vessel_volumes_exact.append(np.nansum(exact_local_vol) * voxel_volume_um3)

            l = measure_edge_length(pts)
            sp = shortest_path(pts)
            tort = np.clip(l / (sp + 1e-8), 0, 5)
            vessel_lengths.append(l * voxel_size_um)
            end_to_end_distances.append(sp * voxel_size_um)
            tortuosities.append(tort)

            deg_u = node_degrees_dict.get(u, 0)
            deg_v = node_degrees_dict.get(v, 0)
            node1_degrees.append(deg_u)
            node2_degrees.append(deg_v)

            is_sprouts.append(graph.nodes[u]['sprout'] or graph.nodes[v]['sprout'])

            direction_vector = pts[-1] - pts[0]
            norm = np.linalg.norm(direction_vector)
            if norm > 1e-8:
                normalized_vector = direction_vector / norm
            else:
                normalized_vector = np.array([np.nan, np.nan, np.nan])

            orientation_zs.append(normalized_vector[0])
            orientation_ys.append(normalized_vector[1])
            orientation_xs.append(normalized_vector[2])

        except (KeyError, IndexError):
            continue

    valid_edges = [(u,v) for i,(u,v) in enumerate(edges) if i < len(vessel_volumes)]
    vessel_metrics_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(valid_edges, names=['node1', 'node2']))

    vessel_metrics_df['z'] = zs
    vessel_metrics_df['y'] = ys
    vessel_metrics_df['x'] = xs

    vessel_metrics_df['vessel_volume'] = vessel_volumes
    vessel_metrics_df['vessel_length'] = vessel_lengths
    vessel_metrics_df['end_to_end_distance'] = end_to_end_distances
    vessel_metrics_df['tortuosity'] = tortuosities
    vessel_metrics_df['is_sprout'] = is_sprouts

    vessel_metrics_df['mean_cross_sectional_area'] = mean_cross_sectional_areas
    vessel_metrics_df['median_cross_sectional_area'] = median_cross_sectional_areas
    vessel_metrics_df['std_cross_sectional_area'] = std_cross_sectional_areas

    if area_image_exact is not None:
        vessel_metrics_df['mean_cross_sectional_area_exact'] = mean_cross_sectional_areas_exact
        vessel_metrics_df['median_cross_sectional_area_exact'] = median_cross_sectional_areas_exact
        vessel_metrics_df['std_cross_sectional_area_exact'] = std_cross_sectional_areas_exact
        vessel_metrics_df['vessel_volume_exact'] = vessel_volumes_exact

    vessel_metrics_df['node1_degree'] = node1_degrees
    vessel_metrics_df['node2_degree'] = node2_degrees
    vessel_metrics_df['orientation_z'] = orientation_zs
    vessel_metrics_df['orientation_y'] = orientation_ys
    vessel_metrics_df['orientation_x'] = orientation_xs

    return vessel_metrics_df

def compute_junction_metrics(graph, junction_metrics_df, distance_threshold=50, voxel_size_um=2.0):
    """
    Compute metrics for graph nodes (junctions and endpoints).

    Metrics include: type (junction vs sprout tip), degree, distance to nearest neighbor
    of same/different type.

    Args:
        graph (nx.Graph): Input graph.
        junction_metrics_df (pd.DataFrame): (Unused, returns new DF).
        distance_threshold (float, optional): Neighbor-counting radius in um. Defaults to 50.
        voxel_size_um (float, optional): Isotropic voxel size in um; node distances are
            reported in um. Defaults to 2.0.

    Returns:
        pd.DataFrame: DataFrame containing metrics for each node.
    """
    nodes = list(graph.nodes())
    if not nodes:
            cols = ['z', 'y', 'x', 'node_degree', 'is_sprout_tip', 'is_junction',
                    'nearest_junction_distance', 'nearest_sprout_tip_distance',
                    'junction_neighbor_count', 'sprout_tip_neighbor_count']
            return pd.DataFrame(index=nodes, columns=cols)

    positions = np.array([graph.nodes[n]['pts'] for n in nodes])
    positions_um = positions * voxel_size_um  # node-to-node distances reported in um

    junction_metrics_df = pd.DataFrame(index=nodes)
    junction_metrics_df['z'] = positions[:, 0]
    junction_metrics_df['y'] = positions[:, 1]
    junction_metrics_df['x'] = positions[:, 2]

    degrees = np.array([graph.degree[n] for n in nodes])
    junction_metrics_df['node_degree'] = degrees

    junction_metrics_df['is_sprout_tip'] = [graph.nodes[u]['sprout'] for u in nodes]
    junction_metrics_df['is_junction'] = [not graph.nodes[u]['sprout'] for u in nodes]

    endpoint_mask = junction_metrics_df['is_sprout_tip'].values
    branch_point_mask = junction_metrics_df['is_junction'].values

    endpoint_indices = np.where(endpoint_mask)[0]
    branch_point_indices = np.where(branch_point_mask)[0]

    has_endpoints = len(endpoint_indices) > 0
    has_branch_points = len(branch_point_indices) > 0

    nan_fill = np.nan
    junction_metrics_df['nearest_junction_distance'] = nan_fill
    junction_metrics_df['nearest_sprout_tip_distance'] = nan_fill
    junction_metrics_df['junction_neighbor_count'] = 0
    junction_metrics_df['sprout_tip_neighbor_count'] = 0

    if has_endpoints and has_branch_points:
        dist_matrix = cdist(positions_um, positions_um)

        dists_from_branches = dist_matrix[branch_point_mask, :]
        branch_to_branch = dists_from_branches[:, branch_point_mask]
        np.fill_diagonal(branch_to_branch, np.inf)
        junction_metrics_df.loc[branch_point_mask, 'nearest_junction_distance'] = np.min(branch_to_branch, axis=1)
        junction_metrics_df.loc[branch_point_mask, 'junction_neighbor_count'] = np.sum(branch_to_branch <= distance_threshold, axis=1)

        branch_to_endpoint = dists_from_branches[:, endpoint_mask]
        junction_metrics_df.loc[branch_point_mask, 'nearest_sprout_tip_distance'] = np.min(branch_to_endpoint, axis=1)
        junction_metrics_df.loc[branch_point_mask, 'sprout_tip_neighbor_count'] = np.sum(branch_to_endpoint <= distance_threshold, axis=1)

        dists_from_endpoints = dist_matrix[endpoint_mask, :]
        endpoint_to_branch = dists_from_endpoints[:, branch_point_mask]
        junction_metrics_df.loc[endpoint_mask, 'nearest_junction_distance'] = np.min(endpoint_to_branch, axis=1)
        junction_metrics_df.loc[endpoint_mask, 'junction_neighbor_count'] = np.sum(endpoint_to_branch <= distance_threshold, axis=1)

        endpoint_to_endpoint = dists_from_endpoints[:, endpoint_mask]
        np.fill_diagonal(endpoint_to_endpoint, np.inf)
        junction_metrics_df.loc[endpoint_mask, 'nearest_sprout_tip_distance'] = np.min(endpoint_to_endpoint, axis=1)
        junction_metrics_df.loc[endpoint_mask, 'sprout_tip_neighbor_count'] = np.sum(endpoint_to_endpoint <= distance_threshold, axis=1)

    elif has_branch_points:
        dist_matrix = cdist(positions_um, positions_um)
        np.fill_diagonal(dist_matrix, np.inf)
        junction_metrics_df['nearest_junction_distance'] = np.min(dist_matrix, axis=1)
        junction_metrics_df['junction_neighbor_count'] = np.sum(dist_matrix <= distance_threshold, axis=1)

    elif has_endpoints:
        dist_matrix = cdist(positions_um, positions_um)
        np.fill_diagonal(dist_matrix, np.inf)
        junction_metrics_df['nearest_sprout_tip_distance'] = np.min(dist_matrix, axis=1)
        junction_metrics_df['sprout_tip_neighbor_count'] = np.sum(dist_matrix <= distance_threshold, axis=1)

    cols_to_check = ['nearest_junction_distance', 'nearest_sprout_tip_distance',
                     'junction_neighbor_count', 'sprout_tip_neighbor_count']
    for col in cols_to_check:
        if col not in junction_metrics_df.columns:
             fill_val = np.inf if 'distance' in col else 0
             junction_metrics_df[col] = fill_val

    return junction_metrics_df

def fractal_dimension_and_lacunarity(array, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0, plot=False):
    """
    Calculate fractal dimension and lacunarity of a binary array using box counting.

    Args:
        array (np.ndarray): Binary input array.
        max_box_size (int, optional): Max box size. Defaults to min dimension log2.
        min_box_size (int, optional): Min box size. Defaults to 1.
        n_samples (int, optional): Number of scales to sample. Defaults to 20.
        n_offsets (int, optional): Number of grid offsets to average over (for robustness). Defaults to 0.
        plot (bool, optional): Whether to plot the fit. Defaults to False.

    Returns:
        tuple: (fractal_dimension, lacunarity)
    """
    if max_box_size == None:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
    scales = np.unique(scales)
    locs = np.array(np.where(array > 0))
    Ns = []
    lacunarity_values = []
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        for offset in offsets:
            bin_edges = [np.arange(0-offset, i+offset, scale) for i in array.shape]
            H1, _ = np.histogramdd(locs.T, bins=bin_edges)
            touched.append(np.sum(H1>0))
        non_zero_hist = H1[H1 > 0]
        lacunarity_values.append(np.var(non_zero_hist) / (np.mean(non_zero_hist) ** 2))
        Ns.append(min(touched))
    Ns = np.array(Ns)
    unique_Ns, indices = np.unique(Ns, return_index=True)
    unique_Ns = unique_Ns[unique_Ns > 0]
    unique_scales = scales[indices[:len(unique_Ns)]]
    unique_lacunarity = np.array(lacunarity_values)[indices[:len(unique_Ns)]]
    coeffs = np.polyfit(np.log(1 / unique_scales), np.log(unique_Ns), 1)
    return coeffs[0], np.mean(unique_lacunarity)

def convex_hull_volume(binary_image):
    """Calculate the volume of the convex hull of foreground pixels."""
    points = np.argwhere(binary_image)
    if len(points) < 4: return 0
    hull = ConvexHull(points)
    return hull.volume
