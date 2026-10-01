"""Compare a compiled .hef against the .onnx it should reproduce, on a Hailo host, without the ML server.

tensor: feed one preprocessed image to both and compare every output head (compile fidelity, any model)
face:   run Immich's own face pipeline (letterbox -> SCRFD decode -> NMS -> align -> ArcFace) on both and compare
        what Immich would store (task fidelity); any HEF left out runs on ONNX, so each stage can be isolated

    python compare.py tensor --onnx visual/model.onnx --hef siglip.hef --images images/ --mean 127.5 --std 127.5
    python compare.py face --det-onnx det.onnx --rec-onnx rec.onnx --det-hef det.hef --rec-hef rec.hef --images images/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # machine-learning/, for immich_ml's host-side helpers

from hef_runner import HefModel  # noqa: E402
from immich_model.constants import FACE_DETECTION_SIZE as DET_SIZE  # noqa: E402

from immich_ml.models.facial_recognition._ops import ALIGNED_SIZE, align_face, decode_scrfd, nms  # noqa: E402
from immich_ml.models.transforms import crop_pil, decode_pil, letterbox, normalize, resize_pil, widen  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
Outputs = dict[str, NDArray[np.float32]]


class OnnxModel:
    """CPU reference. A uint8 input is the fused (v2) graph that normalizes itself, as `OrtSession` detects it."""

    def __init__(self, path: Path, mean: float, std: float) -> None:
        self.path = path
        self.session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
        image = self.session.get_inputs()[0]
        self.input = image.name
        self.raw = image.type == "tensor(uint8)"
        self.outputs = [node.name for node in self.session.get_outputs()]
        self.mean, self.std = mean, std

    def run(self, nhwc: NDArray[np.uint8]) -> Outputs:
        blob = nhwc if self.raw else normalize(nhwc.astype(np.float32), self.mean, self.std).transpose(0, 3, 1, 2)
        return dict(zip(self.outputs, (widen(o) for o in self.session.run(None, {self.input: blob}))))


class HefFeed:
    """A HEF with normalization folded in takes raw pixels; one compiled without it takes them normalized."""

    def __init__(self, model: HefModel, normalized: bool, mean: float, std: float) -> None:
        self.model, self.normalized, self.mean, self.std = model, normalized, mean, std

    def run(self, nhwc: NDArray[np.uint8]) -> Outputs:
        blob = nhwc.astype(np.float32)
        if self.normalized:
            blob = normalize(blob, self.mean, self.std)
        return self.model.run({self.model.inputs[0].name: blob})


def pair_outputs(onnx: Outputs, hef: Outputs, forced: dict[str, str]) -> dict[str, str]:
    """onnx name -> hef name. HEF heads are the raw NHWC convs the ONNX flattens, so pair by element count and
    break ties (e.g. scores16 and boxes32 are both 3200 wide) by the channel ratio the unambiguous pairs share,
    which for SCRFD is its anchors per cell."""
    # both sides hold one frame here; a legacy ONNX output may carry no batch axis, so count every element
    size = {name: array.size for name, array in onnx.items()}
    hef_size = {name: array.size for name, array in hef.items()}
    pairs = dict(forced)
    candidates = {
        o: [h for h in hef if hef_size[h] == size[o] and h not in forced.values()] for o in onnx if o not in forced
    }
    ratio = lambda o, h: hef[h].shape[-1] / onnx[o].shape[-1]  # noqa: E731
    shared = Counter(ratio(o, hs[0]) for o, hs in candidates.items() if len(hs) == 1).most_common(1)
    for o, hs in candidates.items():
        chosen = hs if len(hs) == 1 else [h for h in hs if shared and ratio(o, h) == shared[0][0]]
        chosen = [h for h in chosen if h not in pairs.values()]
        if len(chosen) != 1:
            raise SystemExit(f"cannot pair ONNX output {o!r} (candidates {hs}); pass --output-map {o}=<hef output>")
        pairs[o] = chosen[0]
    return pairs


def aligned(onnx: Outputs, hef: Outputs, pairs: dict[str, str]) -> Outputs:
    """HEF outputs renamed and reshaped to the ONNX outputs they stand in for, in the ONNX's own order."""
    return {o: hef[pairs[o]].reshape(onnx[o].shape) for o in onnx}


def cosine_rows(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(a) == 0:  # a photo without faces
        return np.empty(0, dtype=np.float32)
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def fidelity(a: NDArray[np.float32], b: NDArray[np.float32]) -> dict[str, float]:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return {
        "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
        "max_abs": float(np.abs(a - b).max()),
        "rel_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)),
    }


def stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    v = np.asarray(values, dtype=np.float64)
    return {"n": len(v), "mean": float(v.mean()), "min": float(v.min()), "p5": float(np.percentile(v, 5))}


def timed(fn: Callable[[], Any], latencies: list[float]) -> Any:
    start = time.perf_counter()
    result = fn()
    latencies.append((time.perf_counter() - start) * 1000)
    return result


def latency(values: list[float]) -> dict[str, float]:
    """The first call pays for loading the network onto the device (and ORT's arena), so it is reported apart."""
    if not values:
        return {}
    steady = values[1:] or values
    return {
        "first_ms": values[0],
        "p50_ms": float(np.percentile(steady, 50)),
        "p95_ms": float(np.percentile(steady, 95)),
    }


def load_images(root: Path, limit: int | None) -> list[tuple[str, Image.Image]]:
    paths = sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES)
    # the server hands ML an upright preview, so a phone photo's EXIF rotation is applied here as well
    return [(p.name, decode_pil(ImageOps.exif_transpose(Image.open(p)))) for p in paths[:limit]]


# ── tensor ────────────────────────────────────────────────────────────


def run_tensor(args: argparse.Namespace) -> dict[str, Any]:
    hef = HefModel(args.hef)
    onnx = OnnxModel(args.onnx, args.mean, args.std)
    feed = HefFeed(hef, args.hef_normalized, args.mean, args.std)
    height, width = hef.inputs[0].shape[:2]
    forced = dict(item.split("=", 1) for item in args.output_map)
    pairs: dict[str, str] | None = None
    per_output: dict[str, list[dict[str, float]]] = {}
    onnx_ms: list[float] = []
    hef_ms: list[float] = []

    for name, image in load_images(args.images, args.limit):
        if args.resize == "letterbox":
            pixels, _ = letterbox(image, height)
        else:
            pixels = np.asarray(crop_pil(resize_pil(image, height), height))
        assert pixels.shape[:2] == (height, width), f"HEF takes {height}x{width}, {args.resize} made {pixels.shape}"
        reference = timed(lambda: onnx.run(pixels[None]), onnx_ms)
        compiled = timed(lambda: feed.run(pixels[None]), hef_ms)
        pairs = pairs or pair_outputs(reference, compiled, forced)
        for output, value in aligned(reference, compiled, pairs).items():
            per_output.setdefault(output, []).append(fidelity(reference[output], value) | {"image": name})

    hef.close()
    return {
        "task": "tensor",
        "hef": hef.describe(),
        "onnx": {"path": args.onnx.as_posix(), "raw_input": onnx.raw},
        "pairs": pairs,
        "outputs": {
            output: {metric: stats([row[metric] for row in rows]) for metric in ("cosine", "max_abs", "rel_l2")}
            for output, rows in per_output.items()
        },
        "latency": {"onnx": latency(onnx_ms), "hef": latency(hef_ms)},
        "per_image": per_output,
    }


# ── face ──────────────────────────────────────────────────────────────


def iou(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(a: NDArray[np.float32], b: NDArray[np.float32], threshold: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by IoU, best overlaps first."""
    if len(a) == 0 or len(b) == 0:
        return []
    overlaps = iou(a, b)
    pairs, used_a, used_b = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-overlaps, axis=None), overlaps.shape)):
        if overlaps[i, j] < threshold:
            break
        if i not in used_a and j not in used_b:
            pairs.append((int(i), int(j)))
            used_a.add(i)
            used_b.add(j)
    return pairs


class FaceStage:
    """One pipeline stage on ONNX, or on a HEF whose outputs are mapped back onto that ONNX's."""

    def __init__(self, onnx: OnnxModel, hef: HefModel | None, normalized: bool, forced: dict[str, str]) -> None:
        self.onnx = onnx
        self.hef = hef
        self.feed = HefFeed(hef, normalized, onnx.mean, onnx.std) if hef else None
        self.forced = forced
        self.pairs: dict[str, str] | None = None

    def run_onnx(self, nhwc: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        return list(self.onnx.run(nhwc).values())

    def run_hef(self, nhwc: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        assert self.feed is not None
        compiled = self.feed.run(nhwc)
        if self.pairs is None:  # pairing needs the ONNX's shapes once
            self.pairs = pair_outputs(self.onnx.run(nhwc[:1]), {k: v[:1] for k, v in compiled.items()}, self.forced)
        return [compiled[self.pairs[o]].reshape(len(nhwc), -1, *self._tail(o)) for o in self.onnx.outputs]

    def _tail(self, output: str) -> tuple[int, ...]:
        shape = next(node.shape for node in self.onnx.session.get_outputs() if node.name == output)
        return (shape[-1],) if len(shape) > 1 and isinstance(shape[-1], int) else ()


def detect(heads: list[NDArray[np.float32]], scale: float, min_score: float) -> dict[str, NDArray[np.float32]]:
    """`FaceDetector._predict` after the session call, verbatim."""
    scores, boxes, kps = decode_scrfd(heads, DET_SIZE)
    candidates = scores >= min_score
    scores, boxes, kps = scores[candidates], boxes[candidates] / scale, kps[candidates] / scale
    keep = nms(boxes, scores)
    return {"boxes": boxes[keep], "scores": scores[keep], "landmarks": kps[keep].reshape(-1, 5, 2)}


def crops(image: NDArray[np.uint8], landmarks: NDArray[np.float32]) -> NDArray[np.uint8]:
    out = np.empty((len(landmarks), ALIGNED_SIZE, ALIGNED_SIZE, 3), dtype=np.uint8)
    for crop, kps in zip(out, landmarks):
        align_face(image, kps, crop)
    return out


def embed(run: Callable[[NDArray[np.uint8]], list[NDArray[np.float32]]], faces: NDArray[np.uint8]) -> NDArray:
    if len(faces) == 0:
        return np.empty((0, 512), dtype=np.float32)
    embedding = run(faces)[0].reshape(len(faces), -1)
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)


def same_person(embeddings: NDArray[np.float32], max_distance: float) -> NDArray[np.bool_]:
    """Immich's clustering verdict for every face pair: cosine distance under `maxDistance`."""
    distance = 1 - embeddings @ embeddings.T
    return distance[np.triu_indices(len(embeddings), k=1)] < max_distance


def run_face(args: argparse.Namespace) -> dict[str, Any]:
    forced = dict(item.split("=", 1) for item in args.output_map)
    det = FaceStage(
        OnnxModel(args.det_onnx, 127.5, 128.0),
        HefModel(args.det_hef) if args.det_hef else None,
        args.hef_normalized,
        forced,
    )
    rec = FaceStage(
        OnnxModel(args.rec_onnx, 127.5, 127.5),
        HefModel(args.rec_hef) if args.rec_hef else None,
        args.hef_normalized,
        forced,
    )
    ms: dict[str, list[float]] = {"det_onnx": [], "det_hef": [], "rec_onnx": [], "rec_hef": []}
    det_rows, rec_cos, e2e_cos, all_onnx, all_e2e = [], [], [], [], []

    for name, image in load_images(args.images, args.limit):
        pixels = np.asarray(image, dtype=np.uint8)
        canvas, scale = letterbox(image, DET_SIZE)
        reference = detect(timed(lambda: det.run_onnx(canvas[None]), ms["det_onnx"]), scale, args.min_score)
        onnx_faces = crops(pixels, reference["landmarks"])
        onnx_emb = embed(lambda x: timed(lambda: rec.run_onnx(x), ms["rec_onnx"]), onnx_faces)

        if rec.hef:  # recognition alone, on the very crops ONNX detection aligned
            hef_emb = embed(lambda x: timed(lambda: rec.run_hef(x), ms["rec_hef"]), onnx_faces)
            rec_cos += cosine_rows(onnx_emb, hef_emb).tolist()

        found = reference
        if det.hef:
            found = detect(timed(lambda: det.run_hef(canvas[None]), ms["det_hef"]), scale, args.min_score)
            pairs = match(reference["boxes"], found["boxes"])
            # landmark error as a fraction of the face's width, which a 12 MP photo and a 640 px one share
            width = reference["boxes"][:, 2] - reference["boxes"][:, 0]
            landmark = [
                float(np.abs(reference["landmarks"][i] - found["landmarks"][j]).max() / width[i]) for i, j in pairs
            ]
            det_rows.append(
                {
                    "image": name,
                    "onnx_faces": len(reference["boxes"]),
                    "hef_faces": len(found["boxes"]),
                    "matched": len(pairs),
                    "landmark_err_max": max(landmark, default=0.0),
                    "score_abs_diff_max": max(
                        (float(abs(reference["scores"][i] - found["scores"][j])) for i, j in pairs), default=0.0
                    ),
                }
            )

        # end to end: whatever runs on Hailo, faces matched back to ONNX's by box
        e2e_emb = embed(rec.run_hef if rec.hef else rec.run_onnx, crops(pixels, found["landmarks"]))
        for i, j in match(reference["boxes"], found["boxes"]):
            e2e_cos.append(float(cosine_rows(onnx_emb[i : i + 1], e2e_emb[j : j + 1])[0]))
            all_onnx.append(onnx_emb[i])
            all_e2e.append(e2e_emb[j])

    summary: dict[str, Any] = {"rec_cosine": stats(rec_cos), "e2e_cosine": stats(e2e_cos)}
    if det_rows:
        onnx_total = sum(r["onnx_faces"] for r in det_rows)
        hef_total = sum(r["hef_faces"] for r in det_rows)
        matched = sum(r["matched"] for r in det_rows)
        summary["det"] = {
            "recall": matched / onnx_total if onnx_total else 1.0,
            "precision": matched / hef_total if hef_total else 1.0,
            "landmark_err": stats([r["landmark_err_max"] for r in det_rows if r["matched"]]),
            "score_abs_diff": stats([r["score_abs_diff_max"] for r in det_rows]),
        }
    if len(all_onnx) > 1:
        same_onnx = same_person(np.stack(all_onnx), args.max_distance)
        same_e2e = same_person(np.stack(all_e2e), args.max_distance)
        summary["cluster_agreement"] = {
            "pairs": int(same_onnx.size),
            "agree": float((same_onnx == same_e2e).mean()),
            "same_onnx": int(same_onnx.sum()),
            "same_hef": int(same_e2e.sum()),
        }
    for stage in (det, rec):
        if stage.hef:
            stage.hef.close()
    return {
        "task": "face",
        "hef": {"det": det.hef.describe() if det.hef else None, "rec": rec.hef.describe() if rec.hef else None},
        "onnx": {"det": args.det_onnx.as_posix(), "rec": args.rec_onnx.as_posix()},
        "pairs": {"det": det.pairs, "rec": rec.pairs},
        "summary": summary,
        "latency": {key: latency(values) for key, values in ms.items() if values},
        "per_image": det_rows,
    }


# ── cli ───────────────────────────────────────────────────────────────


def report(result: dict[str, Any], out_dir: Path) -> None:
    hefs = [result["hef"]] if "sha256" in result["hef"] else [h for h in result["hef"].values() if h]
    tag = "-".join(h["sha256"][:12] for h in hefs) or "onnx-only"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['task']}-{tag}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    for hef in hefs:
        print(f"hef {Path(hef['path']).name} sha256={hef['sha256'][:12]}")
        for node in hef["inputs"]:
            quant = f"scale={node['qp_scale']:.6g} zp={node['qp_zp']:g}"
            print(f"  in  {node['name']} {node['shape']} {node['dtype']} {quant}")
    if result["task"] == "tensor":
        print(f"pairs {result['pairs']}")
        for output, metrics in result["outputs"].items():
            c, m, r = metrics["cosine"], metrics["max_abs"], metrics["rel_l2"]
            cosine = f"cos mean={c['mean']:.4f} min={c['min']:.4f}"
            print(f"  {output:>12}  {cosine}  rel_l2={r['mean']:.4f}  max_abs={m['mean']:.4g}")
    else:
        print(json.dumps(result["summary"], indent=2))
    print(f"latency {json.dumps(result['latency'])}")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--images", type=Path, required=True, help="an image or a directory searched recursively")
    common.add_argument("--limit", type=int)
    common.add_argument("--hef-normalized", action="store_true", help="HEF has no folded normalization layer")
    common.add_argument("--output-map", nargs="*", default=[], metavar="ONNX=HEF", help="override output pairing")
    common.add_argument("--results", type=Path, default=Path("results"))
    commands = parser.add_subparsers(dest="command", required=True)

    tensor = commands.add_parser("tensor", parents=[common])
    tensor.add_argument("--onnx", type=Path, required=True)
    tensor.add_argument("--hef", type=Path, required=True)
    tensor.add_argument("--mean", type=float, default=127.5, help="in pixel units, for a float-input ONNX or HEF")
    tensor.add_argument("--std", type=float, default=127.5, help="in pixel units, for a float-input ONNX or HEF")
    tensor.add_argument(
        "--resize", choices=("crop", "letterbox"), default="crop", help="crop is CLIP's, letterbox SCRFD's"
    )

    face = commands.add_parser("face", parents=[common])
    face.add_argument("--det-onnx", type=Path, required=True)
    face.add_argument("--rec-onnx", type=Path, required=True)
    face.add_argument("--det-hef", type=Path)
    face.add_argument("--rec-hef", type=Path)
    face.add_argument("--min-score", type=float, default=0.7, help="Immich's default minScore")
    face.add_argument("--max-distance", type=float, default=0.5, help="Immich's default maxDistance")

    args = parser.parse_args()
    report(run_tensor(args) if args.command == "tensor" else run_face(args), args.results)


if __name__ == "__main__":
    main()
