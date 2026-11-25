"""Split predicted Shenzhen buildings and reconstruct one test sample at LoD1 and LoD2."""

import argparse
from collections import Counter
import json
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
import shapely
from shapely.geometry import LineString
from shapely.ops import split
import trimesh

from shenzhen_als import SEED, read_manifest


CELL = 0.5
MAX_STEP = 2.0
MIN_CELL_COUNT = 80


def instances(xyz, mask):
    cells = np.floor(xyz[mask, :2] / CELL).astype(np.int32)
    if not len(cells):
        return np.zeros(len(xyz), dtype=np.int32)
    shape = tuple(cells.max(axis=0) + 1)
    cell_keys = tuple(cells.T)
    heights = np.full(shape, -np.inf, dtype=np.float32)
    np.maximum.at(heights, cell_keys, xyz[mask, 2])
    occupied = np.isfinite(heights)
    support = ndimage.binary_closing(occupied, structure=np.ones((3, 3))) | occupied
    if not np.array_equal(support, occupied):
        nearest = ndimage.distance_transform_edt(~occupied, return_distances=False, return_indices=True)
        added = support & ~occupied
        heights[added] = heights[tuple(nearest[:, added])]
    node = np.full(shape, -1, dtype=np.int32)
    node[support] = np.arange(support.sum(), dtype=np.int32)
    rows, cols = [], []
    for axes in (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ):
        a, b = axes
        pair = support[a] & support[b]
        i, j = np.where(pair)
        close = np.abs(heights[a][i, j] - heights[b][i, j]) <= MAX_STEP
        rows.append(node[a][i[close], j[close]])
        cols.append(node[b][i[close], j[close]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(support.sum(), support.sum()))
    total, component = connected_components(graph, directed=False)
    sizes = np.bincount(component[node[occupied]], minlength=total)
    kept = np.flatnonzero(sizes >= MIN_CELL_COUNT)
    kept = kept[np.argsort(-sizes[kept], kind="stable")]
    lookup = np.zeros(total, dtype=np.int32)
    lookup[kept] = np.arange(1, len(kept) + 1)
    labels = np.zeros(len(xyz), dtype=np.int32)
    labels[mask] = lookup[component[node[cell_keys]]]
    return labels


def roof_samples(points):
    cells = np.floor(points[:, :2] / CELL).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.lexsort((-points[:, 2], inverse))
    first = np.r_[True, np.diff(inverse[order]) > 0]
    return points[order[first]]


def roof_fit(roof):
    xy = roof[:, :2]
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(roof))
    parts = np.array_split(indices, [int(len(roof) * 0.6), int(len(roof) * 0.8)])
    train, validation, test = parts
    center = xy[train].mean(axis=0)
    level = np.median(roof[train, 2])
    best = None
    for theta in np.arange(0, np.pi, np.pi / 12):
        direction = np.array([np.cos(theta), np.sin(theta)])
        projection = (xy - center) @ direction
        for shift in np.quantile(projection[train], [0.35, 0.5, 0.65]):
            design = np.column_stack((np.ones(len(roof)), xy - center, np.maximum(projection - shift, 0)))
            coeff = np.linalg.lstsq(design[train], roof[train, 2], rcond=None)[0]
            if abs(coeff[-1]) < 0.04:
                continue
            rmse = np.sqrt(np.mean((design[validation] @ coeff - roof[validation, 2]) ** 2))
            if best is None or rmse < best[0]:
                best = (rmse, direction, shift, coeff)
    if best is None:
        return None
    error, direction, shift, coeff = best
    constant_error = np.sqrt(np.mean((roof[validation, 2] - level) ** 2))
    if constant_error <= 0.15 or error >= 0.9 * constant_error:
        return None
    return center, level, direction, shift, coeff, test, constant_error, error


def mesh_for(polygons, roof_height, ground):
    vertices, vertex_ids, faces = [], {}, []

    def vertex(x, y, z):
        key = tuple(round(float(v), 6) for v in (x, y, z))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(key)
        return vertex_ids[key]

    roof_edges = Counter()
    bottom = {}
    for polygon in polygons:
        triangles = shapely.constrained_delaunay_triangles(polygon)
        for triangle in triangles.geoms:
            xy = np.asarray(triangle.exterior.coords)[:3]
            top = [vertex(x, y, z) for (x, y), z in zip(xy, roof_height(xy))]
            base = [vertex(x, y, ground) for x, y in xy]
            bottom.update(zip(top, base))
            faces.extend([top, base[::-1]])
            for a, b in zip(top, top[1:] + top[:1]):
                roof_edges[tuple(sorted((a, b)))] += 1
    for (a, b), count in roof_edges.items():
        if count == 1:
            faces.extend([[a, b, bottom[b]], [a, bottom[b], bottom[a]]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def render_sample(path, points, meshes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = points[::max(1, len(points) // 10000)]
    figure = plt.figure(figsize=(11, 4), layout="constrained")
    lower, upper = meshes[1].bounds
    for index, title in enumerate(("Extracted building points", "LoD1", "LoD2")):
        axis = figure.add_subplot(1, 3, index + 1, projection="3d")
        if index == 0:
            axis.scatter(*points.T, c=points[:, 2], cmap="viridis", s=2, linewidths=0)
        else:
            mesh = meshes[index - 1]
            axis.plot_trisurf(
                *mesh.vertices.T, triangles=mesh.faces, color=("#7795ab" if index == 1 else "#4a9987"),
                linewidth=0.15, edgecolor="#435666", alpha=0.95,
            )
        axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2] + 1), title=title)
        axis.set_box_aspect(np.maximum(upper - lower, 1))
        axis.view_init(elev=30, azim=-55)
        axis.set_axis_off()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_tile(path, xyz, truth, predicted, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(xyz), min(220000, len(xyz)), replace=False)
    xyz = xyz[sample]
    truth, predicted, labels = truth[sample], predicted[sample], labels[sample]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True, layout="constrained")
    for axis, title in zip(axes, ("Source elevation", "Predicted instances", "Point-level errors")):
        axis.set(title=title, xlabel="Local X (m)")
        axis.set_aspect("equal")
    axes[0].set_ylabel("Local Y (m)")
    order = np.argsort(xyz[:, 2])
    elevation = axes[0].scatter(*xyz[order, :2].T, c=xyz[order, 2], s=0.3, cmap="viridis", linewidths=0)
    figure.colorbar(elevation, ax=axes[0], shrink=0.66, label="Local Z (m)")
    assigned = labels > 0
    axes[1].scatter(*xyz[~assigned, :2].T, color="#dce0e4", s=0.2, linewidths=0)
    colors = (labels[assigned] * 149 % 997) / 997
    axes[1].scatter(*xyz[assigned, :2].T, c=colors, cmap="turbo", s=0.4, linewidths=0)
    categories = [
        (~truth & ~predicted, "#dce0e4", "TN"), (truth & predicted, "#188476", "TP"),
        (truth & ~predicted, "#dc8c23", "FN"), (~truth & predicted, "#ae3670", "FP"),
    ]
    for mask, color, _ in categories:
        axes[2].scatter(*xyz[mask, :2].T, color=color, s=0.35, linewidths=0)
    axes[2].legend(handles=[Patch(color=color, label=label) for _, color, label in categories],
                   loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def reconstruct(tile, xyz, labels, origin, folder):
    counts = np.bincount(labels)
    for bid in sorted(range(1, len(counts)), key=lambda n: -counts[n])[:50]:
        points = xyz[labels == bid]
        if not 300 <= len(points) <= 60000:
            continue
        roof = roof_samples(points)
        if not 100 <= len(roof) <= 4500:
            continue
        if np.ptp(roof[:, 2]) > 20 or np.ptp(roof[:, :2], axis=0).min() < 6:
            continue
        fit = roof_fit(roof)
        if fit is None:
            continue
        center, level, direction, shift, coeff, holdout, val_lod1, val_lod2 = fit
        xy = np.floor(roof[:, :2] / CELL) * CELL
        cells = shapely.box(xy[:, 0], xy[:, 1], xy[:, 0] + CELL, xy[:, 1] + CELL)
        outline = shapely.union_all(cells).buffer(0.65, join_style="mitre").buffer(-0.65, join_style="mitre")
        if outline.geom_type == "MultiPolygon":
            outline = max(outline.geoms, key=lambda p: p.area)
        if outline.geom_type != "Polygon" or outline.area < 30:
            continue
        outline = shapely.Polygon(outline.exterior).simplify(0.4, preserve_topology=True)
        midpoint = center + shift * direction
        tangent = np.array([-direction[1], direction[0]])
        parts = list(split(outline, LineString([midpoint - tangent * 500, midpoint + tangent * 500])).geoms)
        if len(parts) != 2 or min(p.area for p in parts) < 8:
            continue
        lo = np.asarray(outline.bounds[:2])
        hi = np.asarray(outline.bounds[2:])
        near = ((xyz[:, :2] >= lo - 6) & (xyz[:, :2] <= hi + 6)).all(axis=1)
        roof_low = np.quantile(roof[:, 2], 0.10)
        nearby_ground = xyz[near & (labels != bid) & (xyz[:, 2] < roof_low - 2), 2]
        if len(nearby_ground) < 30:
            continue
        ground = float(np.quantile(nearby_ground, 0.1))

        def constant(coords):
            return np.full(len(coords), level)

        def two_planes(coords):
            xy = coords - center
            return coeff[0] + xy @ coeff[1:3] + np.maximum(xy @ direction - shift, 0) * coeff[3]

        if two_planes(np.asarray(outline.exterior.coords)).min() <= ground + 1:
            continue
        meshes = [mesh_for([outline], constant, ground), mesh_for(parts, two_planes, ground)]
        if not all(m.is_watertight and m.is_winding_consistent and m.volume > 0 for m in meshes):
            continue
        sample = roof[holdout]
        hit = shapely.covers(outline, shapely.points(sample[:, :2]))
        if not np.any(hit):
            continue
        sample = sample[hit]
        folder.mkdir(parents=True, exist_ok=True)
        for level_num, mesh in enumerate(meshes, 1):
            mesh.export(folder / f"lod{level_num}.obj")
        details = {
            "tile": tile, "building_id": bid, "instance_points": int(counts[bid]),
            "origin_m": origin.tolist(), "ground_z_m": ground + origin[2],
            "footprint_area_m2": outline.area, "holdout_points": int(len(holdout)),
            "holdout_covered_fraction": float(hit.mean()),
        }
        for level_num, (mesh, function, error) in enumerate(zip(meshes, (constant, two_planes), (val_lod1, val_lod2)), 1):
            residual = sample[:, 2] - function(sample[:, :2])
            details[f"lod{level_num}"] = {
                "validation_rmse_m": float(error), "holdout_rmse_m": float(np.sqrt(np.mean(residual ** 2))),
                "holdout_mae_m": float(np.mean(np.abs(residual))), "faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
            }
        (folder / "reconstruction.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        render_sample(folder / "lod_comparison.png", points, meshes)
        return details
    return None


def main(output):
    source, splits = read_manifest()
    summary = {}
    example = None
    for item in splits["test"]["files"]:
        path = source / item["file"]
        las = laspy.read(path)
        origin = np.asarray(las.header.mins)
        xyz = (np.column_stack((las.x, las.y, las.z)) - origin).astype(np.float32)
        pred = np.load(output / "predictions" / f"{path.stem}.npy").astype(bool)
        labels = instances(xyz, pred)
        las.add_extra_dim(laspy.ExtraBytesParams(name="BuildingPred", type=np.uint8))
        las.add_extra_dim(laspy.ExtraBytesParams(name="Building_ID", type=np.int32))
        las.BuildingPred = pred.astype(np.uint8)
        las.Building_ID = labels
        las.write(output / f"{path.stem}_instances.laz")
        render_tile(output / f"{path.stem}_extraction.png", xyz, np.asarray(las.classification) == 6, pred, labels)
        summary[path.stem] = {
            "point_count": len(labels),
            "predicted_building_points": int(pred.sum()),
            "instance_count": int(labels.max()),
            "instance_assigned_points": int((labels > 0).sum()),
        }
        print(path.stem, summary[path.stem], flush=True)
        if example is None:
            example = reconstruct(path.stem, xyz, labels, origin, output / "lod")
    summary["reconstruction"] = example
    (output / "instances.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if example is None:
        raise RuntimeError("No test instance satisfied the two-plane roof and watertightness criteria")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
