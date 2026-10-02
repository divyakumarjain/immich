"""Build the Dataflow Compiler's calibration sets through Immich's own preprocessing.

Every set is raw uint8 NHWC pixels: the HEFs fold normalization in, so this is exactly what they will be fed.

faces (from LFW)
  rec_calib.npy   (N, 112, 112, 3)  faces found by the ONNX detector and aligned by `align_face`, one person each
  det_calib.npy   (N, 640, 640, 3)  LFW is one centered close-up per photo, so a third are single photos and the
                                    rest 2x2 / 4x4 mosaics, to also calibrate the small faces of a group shot
clip (from any photo collection, e.g. COCO val2017)
  clip_calib.npy  (N, S, S, 3)      resized as OpenClipVisualEncoder does, from the model's preprocess_cfg.json
ocr (from any photo collection with some text in it, through the ONNX TextDetector/TextRecognizer themselves)
  det/<variant>/<shape>.npy (N, H, W, 3)  per canvas, photos resized and padded as TextDetector._transform does
                                          (those Immich would send to that canvas first)
  rec/<shape>.npy           (N, 48, W, 3) per line width, text the detector finds, cropped and padded as
                                          TextRecognizer does at runtime

    python make_calib.py faces --lfw datasets/lfw --det-onnx det.onnx --out calib/buffalo_l
    python make_calib.py clip --images datasets/val2017 --model-dir ViT-B-16-SigLIP__webli/visual --out calib/siglip
    MACHINE_LEARNING_MODEL_REVISION=v2 python make_calib.py ocr --images datasets/val2017 \
        --model-repo PP-OCRv5_mobile --out calib/ocr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from compare import DET_SIZE, OnnxModel, crops, detect, letterbox
from numpy.typing import NDArray
from PIL import Image, ImageOps

from immich_ml.models.transforms import crop_pil, get_pil_resampling, resize_pil
from immich_ml.schemas import Shape


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


def rec_set(
    paths: list[Path], detector: OnnxModel, count: int, min_score: float
) -> tuple[NDArray[np.uint8], list[str]]:
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


def det_set(paths: list[Path], count: int, rng: random.Random) -> tuple[NDArray[np.uint8], list[dict[str, Any]]]:
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


def clip_resize(image: Image.Image, cfg: dict[str, Any]) -> NDArray[np.uint8]:
    """`OpenClipVisualEncoder._resize`, from the same preprocess_cfg.json the encoder loads."""
    size = cfg["size"][0] if isinstance(cfg["size"], list) else cfg["size"]
    resample = get_pil_resampling(cfg["interpolation"])
    match cfg.get("resize_mode", "shortest"):
        case "squash":
            image = image.resize((size, size), resample=resample)
        case "shortest":
            image = crop_pil(resize_pil(image, size, resample), size)
        case mode:
            raise SystemExit(f"unsupported resize_mode {mode!r}")
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def run_clip(args: argparse.Namespace) -> None:
    cfg = json.loads((args.model_dir / "preprocess_cfg.json").read_text())
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    chosen = random.Random(args.seed).sample(paths, min(args.count, len(paths)))
    frames = np.stack([clip_resize(ImageOps.exif_transpose(Image.open(p)), cfg) for p in chosen])
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": f"{args.images} (calibration only, not redistributed)",
        "seed": args.seed,
        "preprocess_cfg": cfg,
        "sets": {"clip": save(args.out, "clip_calib", frames)},
        "images": [p.as_posix() for p in chosen],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["sets"], indent=2))


def ocr_canvas(img: Image.Image, canvas: tuple[int, int], max_resolution: int) -> NDArray[np.uint8]:
    """`TextDetector._transform` with the canvas given rather than chosen: the same sizing, rounding and padding, for
    the photo at a library's resolution. A collection like COCO is smaller than max_resolution, which Immich never
    upscales, so as is it would calibrate every canvas on a 640px island in black; a phone photo fills it."""
    from immich_ml.models.ocr.detection import TextDetector

    r32 = TextDetector._round32
    ratio = max_resolution / min(img.height, img.width)
    content_h, content_w = max(r32(img.height * ratio), 32), max(r32(img.width * ratio), 32)
    scale = min(canvas[0] / content_h, canvas[1] / content_w, 1.0)
    content_h = min(max(r32(content_h * scale), 32), canvas[0])
    content_w = min(max(r32(content_w * scale), 32), canvas[1])
    out = np.zeros((*canvas, 3), dtype=np.uint8)  # what is left of the canvas stays black
    out[:content_h, :content_w] = np.asarray(img.resize((content_w, content_h), Image.Resampling.LANCZOS))
    return out


def run_ocr(args: argparse.Namespace) -> None:
    from immich_model.constants import dim_sets_of, dims_label, variant_dir

    from immich_ml.models.ocr.detection import TextDetector
    from immich_ml.models.ocr.recognition import REC_HEIGHT, TextRecognizer
    from immich_ml.schemas import ModelFormat, TextDetectionOptions

    rng = random.Random(args.seed)
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(paths)
    images = [(p, ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in paths[: args.pool]]
    sets: dict[str, Any] = {}

    # detection: per canvas, the photos Immich would route to it first, topped up with the rest
    det_sets = dim_sets_of(args.model_repo / "detection" / "model.onnx")
    for group in det_sets:
        variant = variant_dir(det_sets, group.dims[0])
        resolution = int(variant.removeprefix("res"))
        canvases = [(d["height"], d["width"]) for d in group.dims]
        routed: dict[tuple[int, int], list[Image.Image]] = {c: [] for c in canvases}
        for _, img in images:
            r32, ratio = TextDetector._round32, resolution / min(img.height, img.width)  # as ocr_canvas sizes it
            chosen = TextDetector._canvas(
                [Shape(1, height=h, width=w) for h, w in canvases],
                max(r32(img.height * ratio), 32),
                max(r32(img.width * ratio), 32),
            )
            routed[chosen].append(img)
        for canvas in canvases:
            picks = routed[canvas][: args.det_count]
            picks += [img for _, img in images if all(img is not other for other in picks)][
                : args.det_count - len(picks)
            ]
            frames = np.stack([ocr_canvas(img, canvas, resolution) for img in picks])
            name = f"det/{variant}/{dims_label({'height': canvas[0], 'width': canvas[1]})}"
            (args.out / "det" / variant).mkdir(parents=True, exist_ok=True)
            sets[name] = save(args.out, name, frames) | {"routed": min(len(routed[canvas]), args.det_count)}
            print(name, frames.shape, f"{sets[name]['routed']} routed here")

    # recognition: real text the detector finds, cropped and padded per width as TextRecognizer does
    detector = TextDetector(args.model_repo.name, cache_dir=args.model_repo, model_format=ModelFormat.ONNX)
    recognizer = TextRecognizer(args.model_repo.name, cache_dir=args.model_repo, model_format=ModelFormat.ONNX)
    lines: list[tuple[NDArray[np.uint8], float]] = []
    for _, img in images:
        found = detector.predict(img, options=TextDetectionOptions())
        if len(found["boxes"]) == 0:
            continue
        widths, heights, coeffs = recognizer._crop_geometry(found["boxes"])
        upright = heights * 2 >= widths * 3
        ratios = np.where(upright, heights / np.maximum(widths, 1), widths / np.maximum(heights, 1))
        for i in range(len(widths)):
            if widths[i] > 0 and heights[i] > 0:
                lines.append(
                    (recognizer._crop(img, coeffs[i], widths[i], heights[i], bool(upright[i])), float(ratios[i]))
                )
        if len(lines) >= args.rec_count:
            break
    rec_sets = dim_sets_of(args.model_repo / "recognition" / "model.onnx")
    for dims in (d for group in rec_sets for d in group.dims):
        width = dims["width"]
        frames = np.full((len(lines[: args.rec_count]), REC_HEIGHT, width, 3), 127, dtype=np.uint8)
        for frame, (crop, ratio) in zip(frames, lines):
            resized_w = max(1, min(width, int(np.ceil(REC_HEIGHT * ratio))))
            cv2.resize(crop, (resized_w, REC_HEIGHT), dst=frame[:, :resized_w])
        name = f"rec/{dims_label(dims)}"
        (args.out / "rec").mkdir(parents=True, exist_ok=True)
        sets[name] = save(args.out, name, frames)
        print(name, frames.shape)

    manifest = {
        "source": f"{args.images} (calibration only, not redistributed)",
        "seed": args.seed,
        "images": [p.as_posix() for p, _ in images],
        "text_lines": len(lines),
        "sets": sets,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    faces = commands.add_parser("faces")
    faces.add_argument("--lfw", type=Path, required=True)
    faces.add_argument("--det-onnx", type=Path, required=True, help="Immich's detector, which finds the faces to align")
    faces.add_argument("--out", type=Path, required=True)
    faces.add_argument("--rec-count", type=int, default=1024)
    faces.add_argument("--det-count", type=int, default=512)
    faces.add_argument("--min-score", type=float, default=0.7)
    faces.add_argument("--seed", type=int, default=0)
    clip = commands.add_parser("clip")
    clip.add_argument("--images", type=Path, required=True)
    clip.add_argument("--model-dir", type=Path, required=True, help="the visual submodel, with its preprocess_cfg.json")
    clip.add_argument("--out", type=Path, required=True)
    clip.add_argument("--count", type=int, default=1024)
    clip.add_argument("--seed", type=int, default=0)
    ocr = commands.add_parser("ocr")
    ocr.add_argument("--images", type=Path, required=True)
    ocr.add_argument("--model-repo", type=Path, required=True, help="a v2 OCR repo: detection/ and recognition/")
    ocr.add_argument("--out", type=Path, required=True)
    ocr.add_argument("--pool", type=int, default=2000, help="photos to draw both sets from")
    ocr.add_argument("--det-count", type=int, default=160, help="frames per canvas")
    ocr.add_argument("--rec-count", type=int, default=576, help="text lines per width")
    ocr.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "ocr":
        run_ocr(args)
    elif args.command == "clip":
        run_clip(args)
    else:
        run_faces(args)


def run_faces(args: argparse.Namespace) -> None:
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
