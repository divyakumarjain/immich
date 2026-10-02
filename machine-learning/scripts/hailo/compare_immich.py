"""Compare Immich's own model classes on Hailo against the same classes on ONNX: what Immich would store and return.

Each model is created twice from one cache: with its default format, which is Hailo whenever a device is present and
the model ships a Hailo build, and forced to ONNX. So this exercises the session routing, per-shape graphs and host
tails exactly as the server would. Run in an environment with immich_ml and HailoRT, on a Hailo host:

    MACHINE_LEARNING_CACHE_FOLDER=cache MACHINE_LEARNING_MODEL_REVISION=v2 \\
        python compare_immich.py ocr --model PP-OCRv5_mobile --images photos/
    ... python compare_immich.py clip --model ViT-B-16-SigLIP__webli --images photos/
    ... python compare_immich.py face --model buffalo_l --images photos/

ocr   text lines matched by box (IoU >= 0.5): recall/precision, exact text and character error rate on matches; and
      recognition alone, both recognizers reading the boxes ONNX detection found
clip  per-photo embedding cosine; text-to-image search agreement (top-1, top-5 overlap) for the same queries, with
      the query embedded once on ONNX, as the text encoder stays there
face  faces matched by box, cosine of the stored embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from numpy.typing import NDArray
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # machine-learning/

from immich_ml.models.base import InferenceModel  # noqa: E402
from immich_ml.schemas import ModelFormat  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
QUERIES = [
    "a dog",
    "a cat",
    "people playing tennis",
    "a plate of food",
    "a bus on a city street",
    "a person riding a horse",
    "a pizza",
    "a train at a station",
    "an airplane",
    "a bedroom",
    "a computer keyboard",
    "a baseball game",
    "birds",
    "a bicycle",
    "a beach",
    "snow",
    "a kitchen",
    "a giraffe",
    "an elephant",
    "a clock tower",
    "a group of people smiling",
    "a street sign",
    "a cake",
    "a boat on the water",
]


def photos(root: Path, limit: int | None) -> list[tuple[str, Image.Image]]:
    paths = sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES)
    # the server hands ML an upright preview, so a phone photo's EXIF rotation is applied here as well
    return [(p.name, ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in paths[:limit]]


def pair(model: type[InferenceModel[Any]], name: str) -> tuple[InferenceModel[Any], InferenceModel[Any]]:
    """The model as the server would create it, and the same forced to ONNX."""
    default, onnx = model(name), model(name, model_format=ModelFormat.ONNX)
    default.load()
    if default.model_format != ModelFormat.HAILO:
        raise SystemExit(f"{model.__name__} '{name}' loaded as {default.model_format}, not hailo: no Hailo build?")
    onnx.load()
    print(f"{model.__name__}: {default.model_path.relative_to(default.cache_dir)} ({type(default.session).__name__})")
    return default, onnx


def timed(fn: Callable[[], Any], into: list[float]) -> Any:
    start = time.perf_counter()
    result = fn()
    into.append((time.perf_counter() - start) * 1000)
    return result


def p50(values: list[float]) -> float:
    return float(np.percentile(values[1:] or values, 50))


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance over the reference's length."""
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(reference), 1)


def rects(quads: NDArray[np.float32]) -> NDArray[np.float32]:
    """Axis-aligned bounds of (N, 4, 2) quadrilaterals."""
    if not len(quads):
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate([quads.min(axis=1), quads.max(axis=1)], axis=1).astype(np.float32)


def match(a: NDArray[np.float32], b: NDArray[np.float32], threshold: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of (x1, y1, x2, y2) boxes by IoU, best overlaps first."""
    if len(a) == 0 or len(b) == 0:
        return []
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a, area_b = np.prod(a[:, 2:] - a[:, :2], axis=1), np.prod(b[:, 2:] - b[:, :2], axis=1)
    overlaps = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
    pairs, used_a, used_b = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-overlaps, axis=None), overlaps.shape)):
        if overlaps[i, j] < threshold:
            break
        if i not in used_a and j not in used_b:
            pairs.append((int(i), int(j)))
            used_a.add(i)
            used_b.add(j)
    return pairs


def run_ocr(args: argparse.Namespace) -> dict[str, Any]:
    from immich_ml.models.ocr.detection import TextDetector
    from immich_ml.models.ocr.recognition import TextRecognizer
    from immich_ml.schemas import TextDetectionOptions, TextRecognitionOptions

    hailo_det, onnx_det = pair(TextDetector, args.model)
    hailo_rec, onnx_rec = pair(TextRecognizer, args.model)
    det_opts, rec_opts = TextDetectionOptions(), TextRecognitionOptions()
    lines = {"onnx": 0, "hailo": 0, "matched": 0, "exact": 0}
    errors: list[float] = []
    rec_only = {"lines": 0, "exact": 0}
    rec_only_errors: list[float] = []
    ms: dict[str, list[float]] = {"hailo": [], "onnx": []}
    per_photo = []
    for name, image in photos(args.images, args.limit):
        found = {}
        for backend, det, rec in (("hailo", hailo_det, hailo_rec), ("onnx", onnx_det, onnx_rec)):
            boxes = timed(lambda: det.predict(image, options=det_opts), ms[backend])
            found[backend] = (boxes, rec.predict(image, {k: v.copy() for k, v in boxes.items()}, options=rec_opts))
        (_, hailo_out), (onnx_boxes, onnx_out) = found["hailo"], found["onnx"]
        quads = {k: np.asarray(out["box"]).reshape(-1, 4, 2) for k, out in (("hailo", hailo_out), ("onnx", onnx_out))}
        texts = {"hailo": list(hailo_out["text"]), "onnx": list(onnx_out["text"])}
        matched = match(rects(quads["onnx"]), rects(quads["hailo"]))
        lines["onnx"] += len(texts["onnx"])
        lines["hailo"] += len(texts["hailo"])
        lines["matched"] += len(matched)
        for i, j in matched:
            lines["exact"] += texts["onnx"][i] == texts["hailo"][j]
            errors.append(cer(texts["onnx"][i], texts["hailo"][j]))
        # recognition alone: both recognizers read the boxes ONNX detection found, with every line kept
        if len(onnx_boxes["boxes"]):
            keep_all = TextRecognitionOptions(min_score=0.0)
            alone = [
                list(rec.predict(image, {k: v.copy() for k, v in onnx_boxes.items()}, options=keep_all)["text"])
                for rec in (onnx_rec, hailo_rec)
            ]
            rec_only["lines"] += len(alone[0])
            rec_only["exact"] += sum(a == b for a, b in zip(*alone))
            rec_only_errors += [cer(a, b) for a, b in zip(*alone)]
        per_photo.append({"photo": name, "onnx": texts["onnx"], "hailo": texts["hailo"]})
    summary = {
        "lines": lines,
        "recall": lines["matched"] / max(lines["onnx"], 1),
        "precision": lines["matched"] / max(lines["hailo"], 1),
        "exact_text": lines["exact"] / max(lines["matched"], 1),
        "cer_mean": float(np.mean(errors)) if errors else 0.0,
        "recognition_alone": {
            "lines": rec_only["lines"],
            "exact_text": rec_only["exact"] / max(rec_only["lines"], 1),
            "cer_mean": float(np.mean(rec_only_errors)) if rec_only_errors else 0.0,
        },
        "graphs_configured": {
            "detection": sorted(p.name for p in hailo_det.session.graphs),  # type: ignore[attr-defined]
            "recognition": sorted(p.name for p in hailo_rec.session.graphs),  # type: ignore[attr-defined]
        },
        "detect_p50_ms": {k: p50(v) for k, v in ms.items()},
    }
    return {"task": "ocr", "model": args.model, "summary": summary, "per_photo": per_photo}


def run_clip(args: argparse.Namespace) -> dict[str, Any]:
    from immich_ml.models.clip.textual import OpenClipTextualEncoder
    from immich_ml.models.clip.visual import OpenClipVisualEncoder
    from immich_ml.schemas import TextualOptions, VisualOptions

    hailo_vis, onnx_vis = pair(OpenClipVisualEncoder, args.model)
    text = OpenClipTextualEncoder(args.model, model_format=ModelFormat.ONNX)
    ms: dict[str, list[float]] = {"hailo": [], "onnx": []}
    embeddings: dict[str, list[NDArray[np.float32]]] = {"hailo": [], "onnx": []}
    names = []
    for name, image in photos(args.images, args.limit):
        names.append(name)
        for backend, vis in (("hailo", hailo_vis), ("onnx", onnx_vis)):
            out = timed(lambda: vis.predict(image, options=VisualOptions()), ms[backend])
            embeddings[backend].append(np.array(orjson.loads(out), dtype=np.float32))
    h, o = (np.stack(embeddings[k]) for k in ("hailo", "onnx"))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    o /= np.linalg.norm(o, axis=1, keepdims=True)
    cosine = np.sum(h * o, axis=1)
    queries = np.stack(
        [np.array(orjson.loads(text.predict(q, options=TextualOptions())), dtype=np.float32) for q in QUERIES]
    )
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    k = min(5, len(names))
    top_h, top_o = np.argsort(-(queries @ h.T), axis=1), np.argsort(-(queries @ o.T), axis=1)
    overlap = [len(set(top_h[i, :k]) & set(top_o[i, :k])) / k for i in range(len(QUERIES))]
    summary = {
        "photos": len(names),
        "cosine": {"mean": float(cosine.mean()), "min": float(cosine.min()), "p5": float(np.percentile(cosine, 5))},
        "search_top1_agree": float((top_h[:, 0] == top_o[:, 0]).mean()),
        f"search_top{k}_overlap": float(np.mean(overlap)),
        "p50_ms": {k: p50(v) for k, v in ms.items()},
    }
    worst = [
        {"query": QUERIES[i], "onnx": [names[j] for j in top_o[i, :k]], "hailo": [names[j] for j in top_h[i, :k]]}
        for i in np.argsort(overlap)[:3]
    ]
    return {"task": "clip", "model": args.model, "summary": summary, "worst_queries": worst}


def run_face(args: argparse.Namespace) -> dict[str, Any]:
    from immich_ml.models.facial_recognition.detection import FaceDetector
    from immich_ml.models.facial_recognition.recognition import FaceRecognizer
    from immich_ml.schemas import FaceDetectionOptions, FaceRecognitionOptions

    hailo_det, onnx_det = pair(FaceDetector, args.model)
    hailo_rec, onnx_rec = pair(FaceRecognizer, args.model)
    total = found = 0
    cos: list[float] = []
    ms: dict[str, list[float]] = {"hailo": [], "onnx": []}
    for _, image in photos(args.images, args.limit):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        data = buffer.getvalue()
        results = {}
        for backend, det, rec in (("hailo", hailo_det, hailo_rec), ("onnx", onnx_det, onnx_rec)):
            start = time.perf_counter()
            faces = det.predict(data, options=FaceDetectionOptions())
            results[backend] = (faces, rec.predict(data, faces, options=FaceRecognitionOptions()))
            ms[backend].append((time.perf_counter() - start) * 1000)
        (hf, he), (of, oe) = results["hailo"], results["onnx"]
        total += len(of["boxes"])
        for i, j in match(of["boxes"].astype(np.float32), hf["boxes"].astype(np.float32)):
            found += 1
            a, b = np.array(orjson.loads(oe[i]["embedding"])), np.array(orjson.loads(he[j]["embedding"]))
            cos.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    summary = {
        "faces": {"onnx": total, "matched": found},
        "stored_embedding_cosine": {"mean": float(np.mean(cos)), "min": float(np.min(cos))} if cos else {},
        "p50_ms_per_photo": {k: p50(v) for k, v in ms.items()},
    }
    return {"task": "face", "model": args.model, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=("ocr", "clip", "face"))
    parser.add_argument("--model", required=True, help="e.g. PP-OCRv5_mobile, ViT-B-16-SigLIP__webli, buffalo_l")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    result = {"ocr": run_ocr, "clip": run_clip, "face": run_face}[args.task](args)
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / f"immich-{args.task}-{args.model}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["summary"], indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
