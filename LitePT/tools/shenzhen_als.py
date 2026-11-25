"""Train and evaluate LitePT-S on the Shenzhen airborne LiDAR course tiles."""

import argparse
import json
import math
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "splits" / "shenzhen_als.json"
BLOCK_M = 40.0
HALO_M = 3.0
VOXEL_M = 0.5
GROUND_M = 10.0
SEED = 472
STABILITY_MIN_EPOCH = 16
STABILITY_PATIENCE = 8
STABILITY_MIN_GAIN = 0.002
STABILITY_MEDIAN_TOLERANCE = 0.006
STABILITY_START_LR = 5e-4
STABILITY_STOP_LR = 1.25e-4
STABILITY_LR_FLOOR = 1.5625e-5
STABILITY_LR_PATIENCE = 3


def read_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return (MANIFEST.parent / manifest["data_root_relative_to_manifest"]).resolve(), manifest["splits"]


def build_cache(output):
    source, splits = read_manifest()
    output.mkdir(parents=True, exist_ok=True)
    for split in splits.values():
        for item in split["files"]:
            path = source / item["file"]
            target = output / path.stem
            if (target / "index.json").is_file():
                continue
            las = laspy.read(path)
            if len(las.points) != item["point_count"]:
                raise ValueError(f"Unexpected point count in {path}")
            origin = np.asarray(las.header.mins, dtype=np.float64)
            xyz = np.column_stack((las.x, las.y, las.z)) - origin
            cells = np.floor(xyz[:, :2] / GROUND_M).astype(np.int32)
            shape = tuple(cells.max(axis=0) + 1)
            ground = np.full(shape, np.inf, dtype=np.float32)
            np.minimum.at(ground, tuple(cells.T), xyz[:, 2])
            missing = ~np.isfinite(ground)
            nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
            ground[missing] = ground[tuple(nearest[:, missing])]
            ground = ndimage.median_filter(ground, size=3, mode="nearest")
            hag = xyz[:, 2] - ground[tuple(cells.T)]
            blocks = np.floor(xyz[:, :2] / BLOCK_M).astype(np.int32)
            width = int(blocks[:, 1].max()) + 1
            codes = blocks[:, 0] * width + blocks[:, 1]
            order = np.argsort(codes, kind="stable")
            ids, starts, counts = np.unique(codes[order], return_index=True, return_counts=True)
            target.mkdir(exist_ok=True)
            np.save(target / "xyz.npy", xyz[order].astype(np.float32))
            np.save(target / "strength.npy", np.log1p(np.asarray(las.intensity)[order].astype(np.float32)) / np.log(65536.0))
            np.save(target / "hag.npy", hag[order].astype(np.float32))
            np.save(target / "class.npy", np.asarray(las.classification)[order])
            np.save(target / "original.npy", order.astype(np.int32))
            info = {
                "origin": origin.tolist(),
                "point_count": len(order),
                "blocks": {
                    f"{k // width},{k % width}": [int(start), int(start + count)]
                    for k, start, count in zip(ids.tolist(), starts, counts)
                },
            }
            index = target / "index.json"
            index.with_suffix(".tmp").write_text(json.dumps(info), encoding="utf-8")
            index.with_suffix(".tmp").replace(index)
            print(f"Prepared {path.stem}: {len(order):,} points, {len(ids)} blocks", flush=True)


class Tile:
    def __init__(self, path):
        self.name = path.name
        info = json.loads((path / "index.json").read_text(encoding="utf-8"))
        self.point_count = info["point_count"]
        self.blocks = {tuple(map(int, key.split(","))): tuple(span) for key, span in info["blocks"].items()}
        self.xyz = np.load(path / "xyz.npy", mmap_mode="r")
        self.strength = np.load(path / "strength.npy", mmap_mode="r")
        self.hag = np.load(path / "hag.npy", mmap_mode="r")
        self.classes = np.load(path / "class.npy", mmap_mode="r")
        self.original = np.load(path / "original.npy", mmap_mode="r")

    def sample(self, key, training, rng):
        start, end = self.blocks[key]
        center = np.asarray(key) * BLOCK_M
        core = np.arange(start, end)
        if training:
            rows = core
        else:
            rows = [core]
            for x in range(key[0] - 1, key[0] + 2):
                for y in range(key[1] - 1, key[1] + 2):
                    if (x, y) == key or (x, y) not in self.blocks:
                        continue
                    a, b = self.blocks[x, y]
                    local = self.xyz[a:b, :2] - center
                    keep = ((local >= -HALO_M) & (local < BLOCK_M + HALO_M)).all(axis=1)
                    rows.append(a + np.flatnonzero(keep))
            rows = np.concatenate(rows)
        xyz = self.xyz[rows].copy()
        xyz[:, :2] -= center + BLOCK_M / 2
        xyz[:, 2] -= np.median(xyz[:len(core), 2])
        if training:
            angle = rng.uniform(-np.pi, np.pi)
            sin, cos = np.sin(angle), np.cos(angle)
            xyz[:, :2] = xyz[:, :2] @ np.asarray([[cos, sin], [-sin, cos]], dtype=np.float32)
        grid = np.floor(xyz / VOXEL_M).astype(np.int32)
        grid -= grid.min(axis=0)
        code = np.ravel_multi_index(grid.T, tuple(grid.max(axis=0) + 1))
        order = rng.permutation(len(rows)) if training else np.arange(len(rows))
        _, first, inverse = np.unique(code[order], return_index=True, return_inverse=True)
        selected = order[first]
        ids = rows[selected]
        features = np.column_stack((
            xyz[selected] / 10.0,
            self.strength[ids],
            np.clip(self.hag[ids], -10, 150) / 10.0,
        )).astype(np.float32)
        return features, grid[selected], self.classes[ids], inverse[:len(core)], self.classes[core], self.original[core]


def split_blocks(cache, split, max_blocks):
    _, splits = read_manifest()
    tiles = [Tile(cache / Path(item["file"]).stem) for item in splits[split]["files"]]
    rng = np.random.default_rng(SEED)
    blocks = []
    for tile in tiles:
        # Ignore sub-metre edge slivers in training only; evaluation keeps all points.
        keys = [key for key, (start, end) in tile.blocks.items() if split != "train" or end - start >= 256]
        if max_blocks:
            count = min(len(keys), math.ceil(max_blocks / len(tiles)))
            keys = [keys[i] for i in sorted(rng.choice(len(keys), count, replace=False))]
        blocks.extend((tile, key) for key in keys)
    return blocks


def scores(counts):
    tp, fp, fn, tn, fp5, class5 = map(int, counts)
    divide = lambda num, den: num / den if den else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "iou": divide(tp, tp + fp + fn),
        "high_vegetation_false_positive": fp5,
        "high_vegetation_points": class5,
        "high_vegetation_false_positive_rate": divide(fp5, class5),
    }


def model_input(torch, features, grid, variant, offsets=None):
    if variant == "base":
        features[:, 4] = 0
    feat = torch.from_numpy(features).cuda()
    return {
        "feat": feat,
        "coord": (feat[:, :3] * 10).contiguous(),
        "grid_coord": torch.from_numpy(grid).cuda(),
        "offset": torch.tensor([len(features)] if offsets is None else offsets, device="cuda"),
    }


def evaluate(torch, model, blocks, variant, prediction_dir=None):
    model.eval()
    per_tile = {}
    output = {}
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for tile, key in blocks:
            feat, grid, _, inverse, cls, original = tile.sample(key, False, None)
            logits = model(model_input(torch, feat, grid, variant))
            predicted = logits.argmax(1).cpu().numpy()[inverse].astype(bool)
            truth = cls == 6
            high_vegetation = cls == 5
            counts = np.array([
                (predicted & truth).sum(), (predicted & ~truth).sum(),
                (~predicted & truth).sum(), (~predicted & ~truth).sum(),
                (predicted & high_vegetation).sum(), high_vegetation.sum(),
            ], dtype=np.int64)
            per_tile.setdefault(tile.name, np.zeros(6, dtype=np.int64))[:] += counts
            if prediction_dir:
                if tile.name not in output:
                    output[tile.name] = np.lib.format.open_memmap(
                        prediction_dir / f"{tile.name}.npy", mode="w+", dtype=np.uint8,
                        shape=(tile.point_count,),
                    )
                output[tile.name][original] = predicted
    for prediction in output.values():
        prediction.flush()
    overall = scores(sum(per_tile.values()))
    return {"overall": overall, "per_tile": {name: scores(c) for name, c in per_tile.items()}}


def train(args):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from litept.model import LitePT

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LitePT-S training")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(SEED)
    train_blocks = split_blocks(args.cache, "train", args.train_blocks)
    val_blocks = split_blocks(args.cache, "validation", args.val_blocks)
    args.output.mkdir(parents=True, exist_ok=True)

    class Segmentor(nn.Module):
        def __init__(self):
            super().__init__()
            # A zero fifth channel gives both variants identical initialization.
            self.backbone = LitePT(in_channels=5)
            self.head = nn.Linear(72, 2)

        def forward(self, inputs):
            return self.head(self.backbone(inputs).feat)

    model = Segmentor().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    checkpoint = args.output / "checkpoint.pt"
    report_file = args.output / "training.json"
    records = []
    start = 0
    best = -1.0
    convergence = None
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["variant"] != args.variant:
            raise ValueError("Checkpoint belongs to a different feature variant")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if state.get("batch_size", 1) != args.batch_size:
            raise ValueError("Batch size does not match the checkpoint")
        start, records, best = state["epoch"], state["records"], state["best"]
        convergence = state.get("convergence")
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    if start == 0:
        print(f"Training {args.variant} on {len(train_blocks)} blocks, validating on {len(val_blocks)} blocks, batch={args.batch_size}", flush=True)
    if args.until_stable and convergence is None:
        best_epoch = max(records, key=lambda r: r["validation"]["overall"]["iou"])["epoch"] if records else 0
        convergence = {"reference_iou": best, "last_gain_epoch": best_epoch, "last_lr_change_epoch": start, "stable": False}
        for group in opt.param_groups:
            group["lr"] = STABILITY_START_LR
    epoch = start
    while (args.until_stable and not convergence["stable"]) or (not args.until_stable and epoch < args.epochs):
        model.train()
        learning_rate = opt.param_groups[0]["lr"]
        losses = []
        shuffled = rng.permutation(len(train_blocks))
        for begin in range(0, len(shuffled), args.batch_size):
            batch = []
            for idx in shuffled[begin:begin + args.batch_size]:
                tile, key = train_blocks[idx]
                feat, grid, cls, *_ = tile.sample(key, True, rng)
                if len(feat) >= 96:
                    batch.append((feat, grid, cls))
            if not batch:
                continue
            features, grids, classes = zip(*batch)
            labels = torch.from_numpy((np.concatenate(classes) == 6).astype(np.int64)).cuda()
            offsets = np.cumsum([len(f) for f in features]).tolist()
            opt.zero_grad(set_to_none=True)
            logits = model(model_input(torch, np.concatenate(features), np.concatenate(grids), args.variant, offsets))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.detach().item())
            step = begin // args.batch_size + 1
            if step == 1 or step % 25 == 0:
                total_steps = math.ceil(len(train_blocks) / args.batch_size)
                print(f"{args.variant} epoch {epoch+1} step {step}/{total_steps} loss={losses[-1]:.4f}", flush=True)
        val = evaluate(torch, model, val_blocks, args.variant)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "updates": len(losses), "learning_rate": learning_rate, "validation": val}
        records.append(row)
        quality = val["overall"]["iou"]
        if quality > best:
            best = quality
            torch.save(model.state_dict(), args.output / "model_best.pt")
        if args.until_stable:
            if quality >= convergence["reference_iou"] + STABILITY_MIN_GAIN:
                convergence["reference_iou"] = quality
                convergence["last_gain_epoch"] = epoch + 1
            since_gain = epoch + 1 - convergence["last_gain_epoch"]
            since_decay = epoch + 1 - convergence["last_lr_change_epoch"]
            if min(since_gain, since_decay) >= STABILITY_LR_PATIENCE and learning_rate > STABILITY_LR_FLOOR:
                for group in opt.param_groups:
                    group["lr"] = max(learning_rate * 0.5, STABILITY_LR_FLOOR)
                convergence["last_lr_change_epoch"] = epoch + 1
            recent = [r["validation"]["overall"]["iou"] for r in records[-6:]]
            median_gap = abs(float(np.median(recent[-3:]) - np.median(recent[:3])))
            convergence.update(
                stable=epoch + 1 >= STABILITY_MIN_EPOCH and since_gain >= STABILITY_PATIENCE
                and learning_rate <= STABILITY_STOP_LR and median_gap <= STABILITY_MEDIAN_TOLERANCE,
                median_iou_gap=median_gap, epochs_since_meaningful_gain=since_gain,
                next_learning_rate=opt.param_groups[0]["lr"],
            )
            row["convergence"] = dict(convergence)
        state = dict(
            model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch + 1, records=records,
            variant=args.variant, batch_size=args.batch_size, best=best, rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), convergence=convergence,
        )
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        report_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        epoch += 1
    if args.test:
        model.load_state_dict(torch.load(args.output / "model_best.pt", weights_only=True))
        test = evaluate(torch, model, split_blocks(args.cache, "test", None), args.variant, args.output / "predictions")
        expected = read_manifest()[1]["test"]["point_count"]
        assert sum(test["overall"][k] for k in ("tp", "fp", "fn", "tn")) == expected
        (args.output / "test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
        print(json.dumps(test), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "shenzhen_als")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--variant", choices=("base", "hag"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-blocks", type=int)
    parser.add_argument("--val-blocks", type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--until-stable", action="store_true", help="Decay learning rate and stop on validation IoU stability")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.prepare:
        build_cache(args.cache)
    if args.variant:
        if args.output is None:
            parser.error("--output is required for training")
        train(args)
