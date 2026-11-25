"""Visualize the trained Shenzhen LitePT encoder and its vegetation error example."""

import argparse
import json
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
from torch import nn

from litept.model import LitePT
from shenzhen_als import ROOT, SEED, Tile, model_input, read_manifest


class Segmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LitePT(in_channels=5)
        self.head = nn.Linear(72, 2)

    def forward(self, inputs):
        return self.head(self.backbone(inputs).feat)


def rgb_pca(embeddings):
    features = embeddings - embeddings.mean(axis=0)
    _, _, axes = np.linalg.svd(features, full_matrices=False)
    axes = axes[:3].copy()
    peak = axes[np.arange(3), np.abs(axes).argmax(axis=1)]
    axes *= np.where(peak < 0, -1, 1)[:, None]
    colors = features @ axes.T
    lo, hi = np.quantile(colors, [0.02, 0.98], axis=0)
    return np.clip((colors - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def encoder_features(model, tile, key):
    features, grid, _, inverse, _, _ = tile.sample(key, False, None)
    snapshots = []

    def collect(_, __, point):
        parent_index = point.get("pooling_inverse")
        snapshots.append((point.feat.detach().float().cpu().numpy(),
                          None if parent_index is None else parent_index.detach().cpu().numpy()))

    hooks = [getattr(model.backbone.enc, f"enc{i}").register_forward_hook(collect) for i in range(5)]
    try:
        with torch.inference_mode():
            model(model_input(torch, features, grid, "hag"))
    finally:
        for hook in hooks:
            hook.remove()
    assert len(snapshots) == 5
    # Each input representative point may map to a coarser encoder token.
    active = np.arange(len(features))
    colors = []
    for embedding, parent_index in snapshots:
        if parent_index is not None:
            active = parent_index[active]
        colors.append(rgb_pca(embedding)[active])
    core = np.unique(inverse)
    core = core[np.argsort(features[core, 2])]
    return features[core], [rgb[core] for rgb in colors], [len(snapshot[0]) for snapshot in snapshots]


def render_encoder(output, cache, checkpoint):
    torch.manual_seed(SEED)
    model = Segmentor().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    examples = [("0855925", (8, 9)), ("0855930", (2, 4))]
    fig = plt.figure(figsize=(7.08, 8.05))
    outer = fig.add_gridspec(2, 1, left=.055, right=.975, bottom=.066, top=.954, hspace=.19)
    normal = Normalize(0, 30)
    for row, (name, key) in enumerate(examples):
        features, colors, tokens = encoder_features(model, Tile(cache / name), key)
        print(name, key, "input representatives:", len(features), "encoder tokens:", tokens, flush=True)
        xy = features[:, :2] * 10
        h = features[:, 4] * 10
        inner = outer[row].subgridspec(2, 3, wspace=.10, hspace=.20)
        axes = [fig.add_subplot(spec) for spec in inner]
        axes[0].scatter(*xy.T, c=h, cmap="viridis", norm=normal, s=.9, linewidths=0)
        for stage, rgb in enumerate(colors, 1):
            axes[stage].scatter(*xy.T, c=rgb, s=.9, linewidths=0)
        for index, axis in enumerate(axes):
            axis.set(xlim=(-20, 20), ylim=(-20, 20), aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#9cabba")
            if index == 0:
                title, color = "Input HAG", "#263b52"
            else:
                operator = "Sparse conv" if index <= 3 else "PointROPE"
                title = f"E{index}  {operator}  ({tokens[index - 1]:,})"
                color = "#195e68" if index <= 3 else "#8c5519"
            axis.set_title(title, loc="left", color=color, fontsize=8.8, pad=4)
        axes[0].plot([-16, -6], [-15.5, -15.5], color="white", lw=2.3, solid_capstyle="butt",
                     path_effects=[patheffects.withStroke(linewidth=3.7, foreground="#263b52")])
        axes[0].text(-11, -14, "10 m", color="white", fontsize=8, ha="center",
                     path_effects=[patheffects.withStroke(linewidth=2, foreground="#263b52")])
        top = axes[0].get_position().y1
        left = axes[0].get_position().x0
        fig.text(left, top + .027, f"({chr(97 + row)})  Tile {name}  |  40 m × 40 m",
                 ha="left", va="bottom", fontsize=10.3, fontweight="bold", color="#263b52")
    bar_axis = fig.add_axes([.10, .031, .22, .013])
    fig.colorbar(plt.cm.ScalarMappable(norm=normal, cmap="viridis"), cax=bar_axis,
                 orientation="horizontal", ticks=[0, 15, 30], extend="both")
    bar_axis.set_xticklabels(["0", "15", "30 m"])
    bar_axis.tick_params(labelsize=8, length=2)
    fig.text(.39, .042, "Stage labels include context tokens. PCA colors are fitted per panel.",
             fontsize=8.3, color="#263b52", va="center")
    fig.savefig(output / "encoder_pca.png", dpi=350, facecolor="white")
    plt.close(fig)


def render_error(output, prediction):
    source, _ = read_manifest()
    las = laspy.read(source / "0855925.las")
    x0, y0, _ = las.header.mins
    x, y = np.asarray(las.x) - x0, np.asarray(las.y) - y0
    crop = (x >= 320) & (x < 360) & (y >= 360) & (y < 400)
    indices = np.flatnonzero(crop)
    truth = np.asarray(las.classification) == 6
    vegetation = np.asarray(las.classification) == 5
    pred = np.load(prediction, mmap_mode="r").astype(bool)
    cell = (x >= 330) & (x < 340) & (y >= 370) & (y < 380)
    assert int((vegetation & cell).sum()) == 5270
    assert int((vegetation & cell & pred).sum()) == 2245
    z = np.asarray(las.z)[indices]
    xyz = np.column_stack((x[indices], y[indices], z))
    # Retain every error point and use a deterministic display sample for background.
    label = np.select((truth[indices] & pred[indices], truth[indices] & ~pred[indices],
                       vegetation[indices] & pred[indices], ~truth[indices] & pred[indices]),
                      (1, 2, 3, 4), default=0)
    rng = np.random.default_rng(SEED)
    bg = np.flatnonzero(label == 0)
    kept = np.r_[rng.choice(bg, min(6500, len(bg)), replace=False), np.flatnonzero(label != 0)]
    kept = kept[np.argsort(xyz[kept, 2])]
    xyz, label = xyz[kept], label[kept]
    names = ["Background", "True building", "Missed building", "High vegetation FP", "Other FP"]
    colors = ["#b7c0cb", "#007e74", "#df962c", "#a63b73", "#5865a4"]
    fig = plt.figure(figsize=(3.52, 5.86))
    axes = [fig.add_subplot(211), fig.add_subplot(212, projection="3d")]
    fig.subplots_adjust(left=.16, right=.88, bottom=.14, top=.943, hspace=.37)
    for code, color in enumerate(colors):
        p = xyz[label == code]
        size = 1.1 if code == 0 else 2.4
        axes[0].scatter(*p[:, :2].T, c=color, s=size, linewidths=0, alpha=.76)
        axes[1].scatter(*p.T, c=color, s=size, linewidths=0, alpha=.76, depthshade=False)
    axes[0].add_patch(Rectangle((330, 370), 10, 10, fill=False, ec="#172d41", lw=1.35, ls="--"))
    axes[0].set(xlim=(320, 360), ylim=(360, 400), xlabel="Local X (m)", ylabel="Local Y (m)")
    axes[0].set_aspect("equal")
    axes[1].set(xlim=(320, 360), ylim=(360, 400), xlabel="Local X (m)", ylabel="Local Y (m)", zlabel="Elevation (m)")
    axes[1].set_xticks([320, 340, 360])
    axes[1].set_yticks([380, 400])
    axes[1].set_zticks([50, 100, 150, 200])
    axes[1].view_init(elev=25, azim=-53)
    for axis, title in zip(axes, ("a  Plan view", "b  Oblique view")):
        axis.set_title(title, loc="left", fontweight="semibold", fontsize=9)
        axis.tick_params(labelsize=8, pad=1)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    axes[1].zaxis.label.set_size(8)
    for axis in (axes[1].xaxis, axes[1].yaxis, axes[1].zaxis):
        axis.labelpad = 1
    handles = [Line2D([], [], marker="o", linestyle="none", color=c, label=n, markersize=5)
               for c, n in zip(colors, names)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.50, .017), ncol=2,
               frameon=False, fontsize=8, handletextpad=.25, columnspacing=.8)
    fig.savefig(output / "error_detail.png", dpi=330, facecolor="white")
    plt.close(fig)


def render_training(output, run):
    history = json.loads((run / "training.json").read_text(encoding="utf-8"))
    epochs = np.array([item["epoch"] for item in history])
    iou = np.array([item["validation"]["overall"]["iou"] for item in history])
    f1 = [item["validation"]["overall"]["f1"] for item in history]
    loss = [item["loss"] for item in history]
    rates = [1e4 * item.get("learning_rate", .001) for item in history]
    best = int(iou.argmax())
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.08, 2.48),
                                  gridspec_kw={"width_ratios": [1, 1.26, 1]})
    fig.subplots_adjust(left=.095, right=.975, bottom=.21, top=.81, wspace=.55)
    for label, axis in zip(("a", "b", "c"), (a, b, c)):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e3e8ee", linewidth=.6)
        axis.axvline(4.5, color="#a0aab7", ls="--", lw=.75)
        axis.set(xticks=[1, 5, 9, 13, 17], xlim=(.5, 17.5), xlabel="Epoch")
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(8)
        axis.set_title(f"({label})", loc="left", fontweight="bold", fontsize=9)
    a.plot(epochs, loss, color="#263b53", marker="o", ms=2.6, lw=1.2)
    a.set_ylabel("Cross-entropy loss", fontsize=8.5)
    b.plot(epochs, f1, color="#263b53", marker="s", ms=2.5, lw=1.15, label="F1")
    b.plot(epochs, iou, color="#037d78", marker="o", ms=2.6, lw=1.25, label="IoU")
    b.scatter([epochs[best]], [iou[best]], s=29, facecolor="white", edgecolor="#037d78", zorder=5)
    b.scatter([epochs[best]], [iou[best]], s=6, color="#037d78", zorder=6)
    b.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    b.set_yticks([.76, .84, .92])
    b.set_ylabel("Validation score", fontsize=8.5)
    b.legend(loc="lower center", bbox_to_anchor=(.55, 1.02), frameon=False, ncol=2, fontsize=8,
             handlelength=1.1, columnspacing=.9, borderaxespad=0)
    c.step(epochs, rates, where="mid", color="#ac542a", lw=1.25)
    c.scatter(epochs, rates, color="#ac542a", s=7)
    c.set_yticks([1.25, 2.5, 5, 10], labels=["1.25", "2.5", "5", "10"])
    c.set_ylabel("Learning rate (×0.0001)", fontsize=8.5)
    fig.savefig(output / "training_curves.png", dpi=350, facecolor="white")
    plt.close(fig)


def render_other_report_figures(output, run):
    import trimesh
    from shenzhen_lod import render_sample, render_tile

    detail = json.loads((run / "lod/reconstruction.json").read_text(encoding="utf-8"))
    for name in ("0855925", "0855930"):
        las = laspy.read(run / f"{name}_instances.laz")
        points = (np.column_stack((las.x, las.y, las.z)) - las.header.mins).astype(np.float32)
        truth = np.asarray(las.classification) == 6
        prediction = np.asarray(las.BuildingPred) == 1
        labels = np.asarray(las.Building_ID)
        render_tile(output / f"extract_{name}.png", points, truth, prediction, labels)
        if name == detail["tile"]:
            meshes = [trimesh.load(run / "lod" / f"lod{level}.obj", force="mesh") for level in (1, 2)]
            with plt.rc_context({"axes.titlesize": 14}):
                render_sample(output / "lod_comparison.png", points[labels == detail["building_id"]], meshes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--full-report", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.font_dir:
        for file in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
            font_manager.fontManager.addfont(args.font_dir / file)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "stix"
    run = ROOT / "exp/shenzhen_als/hag_stable"
    render_encoder(args.output, args.cache, run / "model_best.pt")
    render_error(args.output, run / "predictions/0855925.npy")
    if args.full_report:
        render_training(args.output, run)
        render_other_report_figures(args.output, run)
