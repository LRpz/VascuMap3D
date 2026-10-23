import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from .utils import contrast

def visualize_vessel_network(binary_image, graph, area_image=None, sample_name="Vessel Network",
                             voxel_size_um=2.0):
    """
    Draw the vessel graph over a z-sum projection of the mask.

    Edges are coloured by the median equivalent diameter of their cross-sections,
    d = 2 * sqrt(A / pi), in um. Nodes are coloured by type: sprout tips (lime),
    other free ends (orange) and junctions (white).

    Args:
        binary_image (np.ndarray): Binary vessel mask (used for the background projection).
        graph (nx.Graph): Vessel graph; each edge carries its centreline voxels in 'pts'.
        area_image (np.ndarray, optional): Cross-sectional areas (voxel^2) at skeleton voxels.
            Without it, every edge is drawn in one colour. Defaults to None.
        sample_name (str, optional): Title for the plot. Defaults to "Vessel Network".
        voxel_size_um (float, optional): Isotropic voxel size in um. Defaults to 2.0.
    """
    from matplotlib.collections import LineCollection

    segments, diameters = [], []
    for u, v, d in graph.edges(data=True):
        pts = np.asarray(d['pts']).astype(np.int64)
        if len(pts) < 2:
            continue
        segments.append(np.column_stack((pts[:, 2], pts[:, 1])))  # (x, y) in the projection
        if area_image is not None:
            areas = area_image[pts[:, 0], pts[:, 1], pts[:, 2]]
            diameters.append(2.0 * np.sqrt(np.median(areas) / np.pi) * voxel_size_um)
    diameters = np.asarray(diameters, dtype=float)

    mask_2d = contrast(np.sum(binary_image, axis=0), 0, 75)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 20))
    ax.imshow(mask_2d, cmap='gray', alpha=0.35)

    if segments:
        if diameters.size:
            # half the minimum keeps the thinnest vessels off magma's black end
            vmin, vmax = 0.5 * np.min(diameters), np.percentile(diameters, 90)
            norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
            lc = LineCollection(segments, cmap=cm.magma, norm=norm, linewidths=1.5)
            lc.set_array(diameters)
        else:
            lc = LineCollection(segments, colors='tab:red', linewidths=1.0)
        ax.add_collection(lc)

    nodes = list(graph.nodes())
    if nodes:
        pos = np.array([np.asarray(graph.nodes[n]['pts']).reshape(-1, 3).mean(axis=0) for n in nodes])
        deg = np.array([graph.degree(n) for n in nodes])
        sprout = np.array([bool(graph.nodes[n].get('sprout', False)) for n in nodes])
        colors = np.where(deg == 1, np.where(sprout, 'lime', 'orange'), 'white')
        ax.scatter(pos[:, 2], pos[:, 1], c=colors, s=15, alpha=0.5, zorder=5)

    if segments and diameters.size:
        cax = ax.inset_axes([0.35, -0.03, 0.3, 0.012])
        cbar = fig.colorbar(lc, cax=cax, orientation='horizontal', ticks=[])
        cax.set_title('Median vessel diameter (um)', fontsize=12)
        cax.text(-0.02, 0.5, f'{norm.vmin:.0f}', transform=cax.transAxes, ha='right', va='center', fontsize=11)
        cax.text(1.02, 0.5, f'{norm.vmax:.0f}', transform=cax.transAxes, ha='left', va='center', fontsize=11)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title(sample_name, fontsize=20)
    fig.tight_layout()


def graph2image(graph, shape, include_nodes=False):
    """
    Convert a graph back into a binary skeleton image.

    Args:
        graph (nx.Graph): Input graph.
        shape (tuple): Shape of the output image.
        include_nodes (bool, optional): Also draw node voxels ('pts_all' when kept), so a
            component made of a single node is not lost. Defaults to False.

    Returns:
        np.ndarray: Binary skeleton image.
    """

    pruned_skeleton_image = np.zeros(shape)

    def _draw(coords):
        coords = np.atleast_2d(np.asarray(coords))
        if coords.size == 0:
            return
        c = np.empty_like(coords)
        for ax in range(3):
            c[:, ax] = np.clip(coords[:, ax], 0, shape[ax] - 1)
        pruned_skeleton_image[c[:, 0], c[:, 1], c[:, 2]] = 1

    # Nodes are OPTIONAL and off by default, because every existing caller was
    # written against an edges-only skeleton. With include_nodes=True a component
    # whose skeleton is a single node -- a compact, near-spherical blob -- finally
    # contributes voxels instead of silently vanishing, and junction voxels stop
    # being missing from the middle of every branch point. 'pts_all' is the full
    # voxel set of the node when the caller kept it; 'pts' may have been reduced
    # to a single representative coordinate.
    if include_nodes:
        for n, d in graph.nodes(data=True):
            _draw(d.get('pts_all', d.get('pts')))

    for u, v, d in graph.edges(data=True):
        coords = d['pts']

        clipped_coords = np.zeros_like(coords)
        clipped_coords[:, 0] = np.clip(coords[:, 0], 0, shape[0] - 1)
        clipped_coords[:, 1] = np.clip(coords[:, 1], 0, shape[1] - 1)
        clipped_coords[:, 2] = np.clip(coords[:, 2], 0, shape[2] - 1)

        pruned_skeleton_image[clipped_coords[:, 0], clipped_coords[:, 1], clipped_coords[:, 2]] = 1
    
    return pruned_skeleton_image
