"""Build the Dataflow Compiler's calibration sets for the face models from LFW, through Immich's own pipeline.

Both sets are raw uint8 NHWC pixels: the HEFs fold normalization in, so this is exactly what they will be fed.

rec_calib.npy  (N, 112, 112, 3)  faces found by the ONNX detector and aligned by `align_face`, one person each
det_calib.npy  (N, 640, 640, 3)  LFW is one centered close-up per photo, so a third are single photos and the rest
                                 2x2 / 4x4 mosaics, to also calibrate the small faces of a group shot

    python make_calib.py --lfw datasets/lfw --det-onnx det.onnx --out calib/buffalo_l
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from compare import DET_SIZE, OnnxModel, crops, detect, letterbox
from numpy.typing import NDArray
from PIL import Image


def lfw_by_person(root: Path) -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.jpg")):
        people.setdefault(path.parent.name, []).append(path)
    return people


def subject(faces: dict[str, NDArray[np.float32]], size: tuple[int, int]) -> int | None:
    """LFW photos are centered on their subject; any other face in the frame is a bystander."""
    if len(faces["boxes"]) == 0:
        return None
    centers = (faces["boxes"][:, :2] + faces["boxes"][:, 2:]) / 2
    return int(np.argmin(np.linalg.norm(centers - np.array(size) / 2, axis=1)))


def rec_set(paths: list[Path], detector: OnnxModel, count: int, min_score: float) -> tuple[NDArray[np.uint8], list]:
    out, used = [], []
    for path in paths:
        image = Image.open(path).convert("RGB")
        canvas, scale = letterbox(image, DET_SIZE)
        faces = detect(list(detector.run(canvas[None]).values()), scale, min_score)
        index = subject(faces, image.size)
        if index is None:
            continue
        out.append(crops(np.asarray(image), faces["landmarks"][index : index + 1])[0])
        used.append(path.as_posix())
        if len(out) == count:
            break
    return np.stack(out), used


def mosaic(paths: list[Path], grid: int) -> NDArray[np.uint8]:
    tile = DET_SIZE // grid
    canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
    for k, path in enumerate(paths):
        row, col = divmod(k, grid)
        pixels = np.asarray(Image.open(path).convert("RGB"))
        canvas[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile] = cv2.resize(
            pixels, (tile, tile), interpolation=cv2.INTER_AREA
        )
    return canvas


def det_set(paths: list[Path], count: int, rng: random.Random) -> tuple[NDArray[np.uint8], list]:
    out, used = [], []
    for k in range(count):
        grid = (1, 2, 4)[k % 3]
        chosen = rng.sample(paths, grid * grid)
        out.append(letterbox(Image.open(chosen[0]).convert("RGB"), DET_SIZE)[0] if grid == 1 else mosaic(chosen, grid))
        used.append({"grid": grid, "images": [p.as_posix() for p in chosen]})
    return np.stack(out), used


def save(out: Path, name: str, array: NDArray[np.uint8]) -> dict[str, object]:
    path = out / f"{name}.npy"
    np.save(path, array)
    return {"file": path.name, "shape": list(array.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lfw", type=Path, required=True)
    parser.add_argument(
        "--det-onnx", type=Path, required=True, help="Immich's detector, which finds the faces to align"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rec-count", type=int, default=1024)
    parser.add_argument("--det-count", type=int, default=512)
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    people = lfw_by_person(args.lfw)
    # one photo per person first, for as many identities as the set can hold
    firsts = [rng.choice(photos) for photos in people.values()]
    rng.shuffle(firsts)

    args.out.mkdir(parents=True, exist_ok=True)
    detector = OnnxModel(args.det_onnx, 127.5, 128.0)
    rec, rec_used = rec_set(firsts, detector, args.rec_count, args.min_score)
    det, det_used = det_set([p for photos in people.values() for p in photos], args.det_count, rng)

    manifest = {
        "source": "LFW (calibration only, not redistributed)",
        "seed": args.seed,
        "det_onnx": args.det_onnx.as_posix(),
        "min_score": args.min_score,
        "sets": {"rec": save(args.out, "rec_calib", rec), "det": save(args.out, "det_calib", det)},
        "rec_images": rec_used,
        "det_images": det_used,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["sets"], indent=2))


if __name__ == "__main__":
    main()
