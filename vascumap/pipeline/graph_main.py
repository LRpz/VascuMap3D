import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tifffile as tif
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import binary_fill_holes, distance_transform_edt
import sknw
import networkx as nx

from .utils import cupy_chunk_processing
from .skeletonization import (skeletonize_3d_parallel, prune_graph, finalize_graph,
                              collect_border_vicinity_edges)
from .mesh_contraction import reposition_graph_edges, contract_gpu
from .metrics import (compute_cross_sectional_areas, compute_cross_sectional_areas_exact, convex_hull_volume,
                      compute_vessel_metrics,
                      compute_junction_metrics, fractal_dimension_and_lacunarity)
from .visualization import visualize_vessel_network, graph2image

def construct_vessel_network(binary_image, global_metrics_df, skel_method="thinning", contraction_timelim=5*60, do_prune_graph=True, voxel_size_um=2.0, compute_exact_area=True,
                             min_spur_um=80.0, min_loop_um=40.0, min_component_um=20.0,
                             prune_length_cutoff_um=50.0, vicinity_xy_um=100.0,
                             vicinity_z_um=2.0, keep_rings=False,
                             tip_area_ratio_cutoff=0.20, keep_isolated=False,
                             chunk_size=None, overlap=None):
    """
    Construct a graph representation of the vessel network from a binary mask.

    Uses either skeletonization (thinning) or mesh contraction. Writes the global volume
    metrics (chip_volume, vessel_volume, vessel_volume_fraction) into global_metrics_df.
    Every length threshold is given in microns and converted to voxels with voxel_size_um.

    Args:
        binary_image (np.ndarray): Binary vessel mask on an isotropic grid.
        global_metrics_df (pd.DataFrame): DataFrame to store global metrics.
        skel_method (str, optional): Method ('thinning', 'contraction', 'auto'). Defaults to "thinning".
        contraction_timelim (int, optional): Time limit for contraction in seconds. Defaults to 300.
        do_prune_graph (bool, optional): Run the tip/origin area prune and the spur, loop and
            component removal. Defaults to True.
        voxel_size_um (float, optional): Isotropic voxel size in um. Defaults to 2.0.
        compute_exact_area (bool, optional): Also compute the nearest-centerline (Voronoi)
            cross-sectional areas. Defaults to True.
        min_spur_um (float, optional): Terminal branches shorter than this are removed. Defaults to 80.
        min_loop_um (float, optional): Self-loops shorter than this are removed. Defaults to 40.
        min_component_um (float, optional): Components with a total length below this are
            removed. Defaults to 20.
        prune_length_cutoff_um (float, optional): Endpoint branches up to this length are
            removed by prune_graph regardless of taper. Defaults to 50.
        vicinity_xy_um (float, optional): XY border margin; edges inside it are discarded. Defaults to 100.
        vicinity_z_um (float, optional): Z border margin; edges inside it are discarded. Defaults to 2.
        keep_rings (bool, optional): Keep closed rings without a node (sknw ring=True). Defaults to False.
        tip_area_ratio_cutoff (float, optional): Maximum tip/origin cross-sectional area ratio
            for a terminal branch to count as a tapering sprout. Defaults to 0.20.
        keep_isolated (bool, optional): Keep degree-0 nodes, the only skeleton of compact,
            near-spherical objects. Defaults to False.
        chunk_size (tuple, optional): (Z, Y, X) GPU chunk size. Defaults to (Z, 512, 512).
        overlap (tuple, optional): Per-axis chunk overlap in voxels; must exceed the largest
            vessel radius. Defaults to (128, 128, 128).

    Returns:
        tuple:
            - nx.Graph: The pruned, border-trimmed vessel graph.
            - np.ndarray: Cross-sectional areas (voxel^2) at skeleton voxels.
            - np.ndarray or None: Local vessel volume (voxel^3) per skeleton voxel, from the
              nearest-centerline partition, or None if compute_exact_area is False.
            - np.ndarray: The hole-filled binary mask.
    """
    def remove_self_loops(g, min_length):
        g.remove_edges_from([(u, v) for u, v, d in g.edges(data=True)
                            if u == v and d['weight'] < min_length])

    def prune_spurs(g, min_length):
        while True:
            removed = False
            for n in [x for x in g.nodes if g.degree(x) == 1]:
                if g.degree(n) != 1:
                    continue
                u, v = list(g.edges(n))[0]
                if g[u][v]['weight'] < min_length:
                    g.remove_edge(u, v)
                    removed = True
            g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
            if not removed:
                break

    def remove_small_components(g, min_length):
        for comp in list(nx.connected_components(g)):
            total = sum(d['weight'] for _, _, d in g.subgraph(comp).edges(data=True))
            if total < min_length:
                g.remove_nodes_from(comp)

    def remove_degree2_nodes(g):
        """Dissolve degree-2 nodes, merging their two edges into one edge whose
        'pts' run continuously end to end (correctly ordered, no plot artefacts)."""
        changed = True
        while changed:
            changed = False
            for n in list(g.nodes):
                edges = list(g.edges(n))                      # each is (n, nbr)
                if len(edges) != 2 or any(v == n for _, v in edges):  # skip self-loops
                    continue
                (_, u), (_, w) = edges
                if u == w or g.has_edge(u, w):                # would create a self/parallel edge (overwriting an existing u-w vessel)
                    continue
                P1 = g[n][u]['pts']
                P2 = g[n][w]['pts']
                on = g.nodes[n]['o']                          # junction coordinate

                # orient P1 to END at the junction, P2 to START at the junction
                if np.linalg.norm(P1[0] - on) < np.linalg.norm(P1[-1] - on):
                    P1 = P1[::-1]
                if np.linalg.norm(P2[-1] - on) < np.linalg.norm(P2[0] - on):
                    P2 = P2[::-1]

                pts = np.vstack([P1, on[None, :], P2])        # u .. n .. w, continuous
                weight = g[n][u]['weight'] + g[n][w]['weight']
                g.add_edge(u, w, pts=pts, weight=weight)
                g.remove_node(n)
                changed = True

    # Border margins used to discard edge-of-volume artifacts later (see
    # collect_border_vicinity_edges); the chip volume below is measured on the SAME
    # interior region so the density numerator (border-trimmed vessel length) and the
    # denominator (chip volume) share a spatial domain.
    # Every threshold below is specified in MICRONS and converted to voxels here.
    # They were originally hardcoded in voxels for the 2 um pipeline, which silently
    # made them 3x stricter in physical terms on a 6 um volume. The defaults are the
    # micron equivalents of the old voxel constants at 2 um, so 2 um runs are
    # bit-for-bit unchanged: 80/2=40, 40/2=20, 20/2=10, 50/2=25, 100/2=50, 2/2=1.
    _vx = float(voxel_size_um)
    VICINITY_Z = max(0, int(round(vicinity_z_um / _vx)))
    VICINITY_XY = max(0, int(round(vicinity_xy_um / _vx)))

    # Per-axis overlap, capped at 128 in each dimension. Overlap must exceed the largest
    # vessel radius: binary_fill_holes, the EDT and skeletonization are non-local, so too
    # small an overlap corrupts thick vessels near chunk seams. Capping per axis (rather
    # than one scalar) keeps the XY overlap at 128 even for thin-Z volumes (Z < 128), where
    # a uniform overlap would otherwise collapse to the Z size.
    #
    # Both are overridable now, defaulting to exactly the values above so existing
    # callers are unchanged. Passing the whole volume as ONE chunk removes seams
    # altogether, which is strictly better whenever it fits in VRAM: all three
    # operations are non-local, so an overlap bounds the error rather than
    # eliminating it.
    if chunk_size is None:
        chunk_size = (binary_image.shape[0], 512, 512)
    if overlap is None:
        overlap = (128, 128, 128)
    overlap = tuple(int(min(overlap[i], binary_image.shape[i])) for i in range(3))

    binary_filled_holes = cupy_chunk_processing(
        volume=binary_image,
        processing_func=binary_fill_holes,
        chunk_size=chunk_size,
        overlap=overlap,
        ).astype(np.float32)

    # Global volume metrics in physical units (um^3). Vessel volume uses the hole-filled
    # mask; chip (envelope) volume is the convex hull of the border-trimmed interior so it
    # matches the spatial domain of the trimmed graph.
    voxel_volume_um3 = voxel_size_um ** 3
    interior = binary_filled_holes[
        VICINITY_Z:binary_image.shape[0] - VICINITY_Z,
        VICINITY_XY:binary_image.shape[1] - VICINITY_XY,
        VICINITY_XY:binary_image.shape[2] - VICINITY_XY,
    ]
    chip_volume = (convex_hull_volume(interior) if np.any(interior) else 0) * voxel_volume_um3
    vessel_volume = np.sum(binary_filled_holes) * voxel_volume_um3
    vessel_volume_fraction = vessel_volume / chip_volume if chip_volume > 0 else 0

    global_metrics_df.loc[0, 'chip_volume'] = chip_volume
    global_metrics_df.loc[0, 'vessel_volume'] = vessel_volume
    global_metrics_df.loc[0, 'vessel_volume_fraction'] = vessel_volume_fraction

    if skel_method == "auto":
        if vessel_volume_fraction > 0.3:
            skel_method = "thinning"
        else:
            skel_method = "contraction"
        print(f"Vessel volume fraction: {vessel_volume_fraction:.3f}. Using {skel_method} method for skeletonization")

    binary_edt = cupy_chunk_processing(
        volume=binary_filled_holes,
        processing_func=distance_transform_edt,
        chunk_size=chunk_size,
        overlap=overlap,
        )

    area_image_exact = None

    if skel_method == "thinning":
        skeleton_image = skeletonize_3d_parallel(
            binary_filled_holes,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        graph = sknw.build_sknw(skeleton_image, multi=False, iso=keep_isolated, ring=keep_rings)

        for n in list(graph.nodes()):
            # Keep the node's FULL voxel set before collapsing 'pts' to one
            # representative coordinate. For a compact blob that whole set IS the
            # skeleton, and discarding it is why such blobs rendered empty.
            graph.nodes[n]['pts_all'] = np.atleast_2d(graph.nodes[n]['pts'])
            graph.nodes[n]['pts'] = graph.nodes[n]['pts'][0]
            graph.nodes[n]['sprout'] = graph.degree(n) == 1
        
        skeleton_image = graph2image(graph, binary_image.shape, include_nodes=keep_isolated)
        area_image = compute_cross_sectional_areas(binary_filled_holes, skeleton_image, binary_edt)

        # do_prune_graph was previously accepted and never tested: every prune
        # below ran regardless of what the caller passed. It now gates all of
        # them, so do_prune_graph=False really does return the raw skeleton
        # topology.
        if do_prune_graph:
            graph = prune_graph(
                    graph=graph,
                    area_3d=area_image,
                    tip_area_ratio_cutoff=tip_area_ratio_cutoff,
                    length_cutoff=max(1, int(round(prune_length_cutoff_um / _vx)))
                )

            # ---- parameters (voxels, derived from the micron values above) --------
            MIN_SPUR   = max(1, int(round(min_spur_um / _vx)))       # terminal branches shorter than this are removed
            MIN_LOOP   = max(1, int(round(min_loop_um / _vx)))       # self-loops shorter than this are removed
            MIN_COMP   = max(1, int(round(min_component_um / _vx)))  # connected components with total length below this are removed
            print(f'[prune] voxel {_vx} um -> MIN_SPUR {MIN_SPUR} MIN_LOOP {MIN_LOOP} '
                  f'MIN_COMP {MIN_COMP} VICINITY_XY {VICINITY_XY} VICINITY_Z {VICINITY_Z} '
                  f'rings {keep_rings} tip_area_ratio {tip_area_ratio_cutoff} '
              f'keep_isolated {keep_isolated}', flush=True)

            remove_self_loops(graph, MIN_LOOP)
            prune_spurs(graph, MIN_SPUR)
            remove_small_components(graph, MIN_COMP)
        else:
            print(f'[prune] DISABLED (do_prune_graph=False): no tip-area prune, '
                  f'no spur/loop/component removal. VICINITY_XY {VICINITY_XY} '
                  f'VICINITY_Z {VICINITY_Z} rings {keep_rings}', flush=True)

        # Not pruning: dissolving degree-2 nodes merges two collinear edges into
        # one and removes no vasculature, so it runs either way -- without it the
        # edge list is split at arbitrary skeleton points.
        remove_degree2_nodes(graph)

    elif skel_method == "contraction":
         from .mesh_contraction import create_graph_contraction # Local import
         G, *_ = create_graph_contraction(binary_edt, time_lim=contraction_timelim)

         G_repositioned = reposition_graph_edges(
            G, 
            binary_edt, 
            min_segment_length=10, 
            max_disp=10, 
            num_iterations=1000, 
            tol=1e-20,
            step_size=1,
            )

         graph = finalize_graph(G_repositioned, binary_edt)

    trimmed_graph = collect_border_vicinity_edges(graph, binary_image.shape, vicinity_z=VICINITY_Z, vicinity_xy=VICINITY_XY, keep_isolated=keep_isolated)
    isolated_nodes = [node for node in trimmed_graph.nodes() if trimmed_graph.degree[node] == 0]

    # With keep_isolated=True a degree-0 node is not debris -- it is the entire
    # skeleton of a compact, near-spherical object, which is the only
    # representation such a blob has. Removing them here silently undid iso=True
    # and left the returned graph disagreeing with area_image about which objects
    # exist.
    if isolated_nodes and not keep_isolated:
        trimmed_graph.remove_nodes_from(isolated_nodes)

    skeleton_image = graph2image(graph, binary_image.shape, include_nodes=keep_isolated)
    area_image = compute_cross_sectional_areas(binary_filled_holes, skeleton_image, binary_edt)

    if compute_exact_area:
        area_image_exact = compute_cross_sectional_areas_exact(
            binary_filled_holes, skeleton_image, chunk_size=chunk_size, overlap=overlap
        )


    return trimmed_graph, area_image, area_image_exact, binary_filled_holes

def calculate_graph_metrics(graph, sample_name, area_image, global_metrics_df, area_image_exact=None, voxel_size_um=2.0):
    """
    Calculate metrics from the vessel graph and area image.

    Aggregates vessel and junction metrics into global statistics (counts, densities,
    medians and standard deviations). Lengths are in um, areas in um^2, volumes in um^3.

    Args:
        graph (nx.Graph): Vessel graph.
        sample_name (str): Name of the sample.
        area_image (np.ndarray): 3D area volume (voxel^2).
        global_metrics_df (pd.DataFrame): DataFrame to append global metrics to.
        area_image_exact (np.ndarray, optional): Local vessel volume per skeleton voxel
            (voxel^3), from compute_cross_sectional_areas_exact. Defaults to None.
        voxel_size_um (float, optional): Isotropic voxel size in um. Defaults to 2.0.

    Returns:
        tuple:
            - global_metrics_df (pd.DataFrame): Updated global metrics.
            - junction_metrics_df (pd.DataFrame): Detailed junction metrics.
            - vessel_metrics_df (pd.DataFrame): Detailed vessel metrics.
    """
    junction_metrics_df = pd.DataFrame()
    vessel_metrics_df = pd.DataFrame()

    if graph.number_of_edges() > 0:
        # fractal dimension / lacunarity of the FINAL (pruned + trimmed) centerline, so it
        # describes the same structure as the length/count metrics below.
        skeleton_from_graph = graph2image(graph, area_image.shape)
        fd, lacunarity = fractal_dimension_and_lacunarity(skeleton_from_graph > 0)
        total_vessel_length = np.sum([np.linalg.norm(np.diff(graph[u][v]['pts'], axis=0), axis=1).sum() for u, v in graph.edges()]) * voxel_size_um
        # Junctions are genuine branch points (degree >= 3). The 'sprout' flag is intentionally
        # the original-topology label (pruning only removes sprouts, so a junction that loses
        # branches must not be relabeled a sprout), and is used for the edge classification below.
        branchpoints_count = sum(1 for n in graph.nodes() if graph.degree(n) >= 3) 
        vessels_count = sum(1 for u, v in graph.edges() if not graph.nodes[u]['sprout'] and not graph.nodes[v]['sprout'])
        sprouts_count = graph.number_of_edges() - vessels_count 

    else: 
        fd, lacunarity = np.nan, np.nan
        total_vessel_length = 0
        branchpoints_count = 0
        vessels_count = 0 
        sprouts_count = 0

    chip_volume = global_metrics_df.loc[0, 'chip_volume']
    safe_length = total_vessel_length if total_vessel_length > 0 else np.nan
    safe_chip = chip_volume if chip_volume > 0 else np.nan

    global_metrics_df.loc[0, 'total_vessel_length'] = total_vessel_length
    global_metrics_df.loc[0, 'total_sprout_count'] = sprouts_count
    global_metrics_df.loc[0, 'total_branch_count'] = vessels_count
    global_metrics_df.loc[0, 'total_junction_count'] = branchpoints_count
    global_metrics_df.loc[0, 'skeleton_fractal_dimension'] = fd
    global_metrics_df.loc[0, 'skeleton_lacunarity'] = lacunarity
    global_metrics_df.loc[0, 'vessel_length_density'] = total_vessel_length / safe_chip
    global_metrics_df.loc[0, 'sprout_density_per_vessel_length'] = sprouts_count / safe_length
    global_metrics_df.loc[0, 'junction_density_per_vessel_length'] = branchpoints_count / safe_length
    global_metrics_df.loc[0, 'sprout_density_per_chip_volume'] = sprouts_count / safe_chip
    global_metrics_df.loc[0, 'junction_density_per_chip_volume'] = branchpoints_count / safe_chip

    if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
        vessel_metrics_df = compute_vessel_metrics(graph, area_image, vessel_metrics_df, area_image_exact=area_image_exact, voxel_size_um=voxel_size_um)
        junction_metrics_df = compute_junction_metrics(graph, junction_metrics_df, distance_threshold=500, voxel_size_um=voxel_size_um)

        if not vessel_metrics_df.empty:
            global_metrics_df.loc[0, 'median_vessel_tortuosity'] = np.nanmedian(vessel_metrics_df['tortuosity'])
            global_metrics_df.loc[0, 'vessel_tortuosity_heterogeneity'] = np.nanstd(vessel_metrics_df['tortuosity'])
            global_metrics_df.loc[0, 'median_vessel_cross_sectional_area'] = np.nanmedian(vessel_metrics_df['median_cross_sectional_area'])
            global_metrics_df.loc[0, 'vessel_cross_sectional_area_heterogeneity'] = np.nanstd(vessel_metrics_df['median_cross_sectional_area'])

            if 'median_cross_sectional_area_exact' in vessel_metrics_df.columns:
                global_metrics_df.loc[0, 'median_vessel_cross_sectional_area_exact'] = np.nanmedian(vessel_metrics_df['median_cross_sectional_area_exact'])
                global_metrics_df.loc[0, 'vessel_cross_sectional_area_heterogeneity_exact'] = np.nanstd(vessel_metrics_df['median_cross_sectional_area_exact'])

        if not junction_metrics_df.empty:
            junction_mask = junction_metrics_df['is_junction']
            sprout_tip_mask = junction_metrics_df['is_sprout_tip']

            junction_distances = junction_metrics_df.loc[junction_mask, 'nearest_junction_distance']
            global_metrics_df.loc[0, 'median_junction_nearest_neighbor_distance'] = np.nanmedian(junction_distances)
            global_metrics_df.loc[0, 'junction_spacing_heterogeneity'] = np.nanstd(junction_distances)

            sprout_tip_distances = junction_metrics_df.loc[sprout_tip_mask, 'nearest_sprout_tip_distance']
            global_metrics_df.loc[0, 'median_sprout_tip_nearest_neighbor_distance'] = np.nanmedian(sprout_tip_distances)
            global_metrics_df.loc[0, 'sprout_tip_spacing_heterogeneity'] = np.nanstd(sprout_tip_distances)

    if not junction_metrics_df.empty:
        junction_metrics_df['sample'] = sample_name
    if not vessel_metrics_df.empty:
        vessel_metrics_df['sample'] = sample_name

    return global_metrics_df, junction_metrics_df, vessel_metrics_df

def process_directory(masks_dir_path, suffix, save_local_metrics, visualization=True, skel_method="thinning", contraction_timelim=5*60, voxel_size_um=2.0, compute_exact_area=True):
    """
    Process all masks in a directory.

    Args:
        masks_dir_path (str): Directory containing mask files.
        suffix (str): Suffix of mask files (e.g., '_mask').
        save_local_metrics (bool): Whether to save per-vessel/junction metrics.
        visualization (bool, optional): Whether to save visualization images. Defaults to True.
        skel_method (str, optional): Skeletonization method. Defaults to "thinning".
        contraction_timelim (int, optional): Time limit for contraction. Defaults to 300.
        voxel_size_um (float, optional): Isotropic voxel size of the masks in um. Defaults to 2.0.
        compute_exact_area (bool, optional): Also compute the nearest-centerline
            cross-sectional areas. Defaults to True.
    """

    masks_dir = Path(masks_dir_path)
    masks_paths = list(masks_dir.glob(f'*{suffix}.tif'))
    global_csv_path = masks_dir / f'global_metrics_{skel_method}.csv'

    cumulative_global_metrics_df = pd.DataFrame()

    for mask_path in tqdm(masks_paths, desc='Processing masks'):
        sample_name = mask_path.stem
        try:
            binary_image = tif.imread(str(mask_path))
            
            if np.sum(binary_image) > 0:
                
                global_metrics_df = pd.DataFrame({'sample': [sample_name]})

                graph, area_image, area_image_exact, binary_filled_holes = construct_vessel_network(
                    binary_image, 
                    global_metrics_df, 
                    skel_method=skel_method, 
                    contraction_timelim=contraction_timelim,
                    voxel_size_um=voxel_size_um,
                    compute_exact_area=compute_exact_area,
                    )

                global_metrics_df, junction_metrics_df, vessel_metrics_df = calculate_graph_metrics(
                    graph, sample_name, area_image, global_metrics_df, area_image_exact=area_image_exact, voxel_size_um=voxel_size_um)
                
                if save_local_metrics:
                    junction_csv_path = masks_dir / f'{sample_name}_junction_metrics_{skel_method}.h5'
                    vessel_csv_path = masks_dir / f'{sample_name}_vessel_metrics_{skel_method}.h5'

                    junction_metrics_df.to_hdf(junction_csv_path, index=False, key='df', mode='w')
                    vessel_metrics_df.to_hdf(vessel_csv_path, index=False, key='df', mode='w')
                
                if visualization:
                    visualize_vessel_network(binary_filled_holes, graph, area_image,
                                             sample_name=sample_name, voxel_size_um=voxel_size_um)
                    plt.savefig(masks_dir / f'{sample_name}_graph_{skel_method}.png', dpi=300)
                    plt.close('all')
            else: 
                global_metrics_df = pd.DataFrame({'sample' : [sample_name]})
            
            cumulative_global_metrics_df = pd.concat([cumulative_global_metrics_df, global_metrics_df], ignore_index=True)
            cumulative_global_metrics_df.to_csv(global_csv_path, index=False) 

        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            continue

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute graph metrics on the vessel segmentation masks.")

    parser.add_argument("--masks_dir_path", type=str, required=True,
                        help="Path to the directory containing masks.")
    
    parser.add_argument("--file_suffix", type=str, default='_mask', # Changed type to str
                        help="Suffix of the mask files.")
    
    parser.add_argument("--save_local_metrics", action='store_true',
                        help="Save vessel and junction metrics for each mask, only save the global metrics otherwise.")

    parser.add_argument("--graph_visualization", action='store_true',
                        help="Save graph visualization.")
    
    parser.add_argument("--method", type=str, default="thinning",
                        help="Skeletonization method: thinning or contraction")

    parser.add_argument("--voxel_size_um", type=float, default=2.0,
                        help="Isotropic voxel size of the masks in microns.")

    parser.add_argument("--no_exact_area", action='store_true',
                        help="Skip the nearest-centerline (exact) cross-sectional areas.")

    args = parser.parse_args()

    process_directory(
        masks_dir_path=args.masks_dir_path,
        suffix=args.file_suffix,
        save_local_metrics=args.save_local_metrics,
        visualization=args.graph_visualization,
        skel_method=args.method,
        voxel_size_um=args.voxel_size_um,
        compute_exact_area=not args.no_exact_area,
        )
