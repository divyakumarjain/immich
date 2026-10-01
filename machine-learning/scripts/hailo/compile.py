"""Compile one of Immich's fused (v2) ONNX models to a Hailo .hef with the Dataflow Compiler (DFC).

Runs inside the Hailo AI Software Suite container. The steps:

1. prepare  cut the graph between its preprocess and its host-side tail, pin the shape, inline the weights:
            uint8 NHWC -> Cast -> Transpose -> Sub(mean) [-> Mul/Div(1/std)] -> ...body... -> cut tensors -> tail
            The preprocess becomes the HEF's `normalization` layer, so the HEF takes raw uint8 pixels; the tail
            (reshape/split/sigmoid, L2 norm) stays on the host, where compare.py's HostTail replays it from the ONNX
2. optimize quantize against the calibration set (raw uint8 NHWC, from make_calib.py)
3. check    run the quantized emulator against ONNX on held-out images before anything reaches a device
4. compile  write model.hef plus model.json: what was compiled from what, with which settings

    python compile.py --onnx buffalo_l/v2/detection/model.onnx --calib calib/buffalo_l/det_calib.npy --out out/
    python compile.py --onnx ... --prepare-only   # step 1 alone; needs only onnx, not the DFC
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import numpy_helper

# where Immich's v2 face graphs hand over to the host: the per-stride fused SCRFD heads (score|box|kps,
# 15 channels per anchor) before their Transpose/Reshape/Split/Sigmoid, and ArcFace's Gemm before its L2 norm
DEFAULT_CUTS = {
    "detection": {"end": ["val_0", "val_4", "val_8"], "hw": (640, 640)},
    "recognition": {"end": ["683"], "hw": (112, 112)},
}


def constant(model: onnx.ModelProto, name: str) -> NDArray[np.float32] | None:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t)
    return None


def preprocess(model: onnx.ModelProto) -> tuple[str, list[float], list[float]]:
    """The tensor the body starts from, and the mean/std the preprocess applied to get there (pixel units)."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def sole(tensor: str, op: str) -> onnx.NodeProto | None:
        nodes = consumers.get(tensor, [])
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op else None

    image = model.graph.input[0]
    cast = sole(image.name, "Cast")
    transpose = cast and sole(cast.output[0], "Transpose")
    sub = transpose and sole(transpose.output[0], "Sub")
    if sub is None:
        raise SystemExit("not a fused uint8 graph: expected Cast -> Transpose -> Sub at its input")
    channels = image.type.tensor_type.shape.dim[3].dim_value
    mean = constant(model, sub.input[1])
    std = np.ones(1, dtype=np.float32)
    start = sub.output[0]
    scale = sole(start, "Mul") or sole(start, "Div")
    if scale is not None and (factor := constant(model, scale.input[1])) is not None and factor.size in (1, channels):
        std = 1 / factor if scale.op_type == "Mul" else factor
        start = scale.output[0]
    assert mean is not None, "the Sub's mean is not a constant"
    return start, np.broadcast_to(mean.ravel(), channels).tolist(), np.broadcast_to(std.ravel(), channels).tolist()


def prepare(source: Path, work: Path, end: list[str], hw: tuple[int, int]) -> tuple[Path, dict[str, Any]]:
    model = onnx.load(source.as_posix())  # pulls the safetensors sidecar into memory
    start, mean, std = preprocess(model)
    body = onnx.utils.Extractor(model).extract_model([start], end)
    # pin the start tensor to one NCHW frame; the DFC wants static shapes
    dims = body.graph.input[0].type.tensor_type.shape.dim
    for dim, value in zip(dims, (1, len(mean), *hw)):
        dim.Clear()
        dim.dim_value = value
    body = onnx.shape_inference.infer_shapes(body)
    path = work / f"{source.parent.name}.cut.onnx"
    onnx.save(body, path.as_posix())  # weights inline: one self-contained file for the parser
    return path, {"start": start, "end": end, "mean": mean, "std": std, "hw": list(hw)}


def alls(spec: dict[str, Any], args: argparse.Namespace, calib_size: int) -> str:
    lines = [
        f"normalization1 = normalization({spec['mean']}, {spec['std']})",
        f"model_optimization_config(calibration, batch_size={args.calib_batch}, calibset_size={calib_size})",
        f"model_optimization_flavor(optimization_level={args.opt_level}, compression_level=0)",
        "performance_param(compiler_optimization_level=max)",
        *args.alls,
    ]
    return "\n".join(lines) + "\n"


def emulator_check(runner: Any, cut: Path, holdout: NDArray[np.uint8], spec: dict[str, Any]) -> dict[str, Any]:
    """Quantized emulator vs ONNX over the same body: what quantization alone costs, before any hardware."""
    import onnxruntime as ort
    from hailo_sdk_client import InferenceContext

    session = ort.InferenceSession(cut.as_posix(), providers=["CPUExecutionProvider"])
    mean, std = np.array(spec["mean"], np.float32), np.array(spec["std"], np.float32)
    nchw = ((holdout.astype(np.float32) - mean) / std).transpose(0, 3, 1, 2)
    reference = [
        np.concatenate([session.run(None, {session.get_inputs()[0].name: x[None]})[i] for x in nchw])
        for i in range(len(session.get_outputs()))
    ]
    results = {}
    for context in ("SDK_FP_OPTIMIZED", "SDK_QUANTIZED"):
        with runner.infer_context(getattr(InferenceContext, context)) as ctx:
            emulated = runner.infer(ctx, holdout.astype(np.float32))
        emulated = emulated if isinstance(emulated, list) else [emulated]
        rows = {}
        for name, ref in zip(spec["end"], reference):
            # outputs come back NHWC (or flat); pair each ONNX tensor with the one of matching size and channels
            nhwc = ref.transpose(0, 2, 3, 1) if ref.ndim == 4 else ref.reshape(len(ref), -1)
            match = next(e for e in emulated if e.size == nhwc.size and e.shape[-1] == nhwc.shape[-1])
            a, b = nhwc.reshape(len(nhwc), -1).astype(np.float64), match.reshape(len(match), -1).astype(np.float64)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            rows[name] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min())}
        results[context] = rows
        print(context, json.dumps(rows))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--onnx", type=Path, required=True, help="a v2 submodel: <repo>/<detection|recognition>/model.onnx"
    )
    parser.add_argument("--calib", type=Path, help="(N, H, W, 3) uint8 from make_calib.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", nargs="*", help="cut tensors; defaults by submodel")
    parser.add_argument("--hw", type=int, nargs=2, help="input height width; defaults by submodel")
    parser.add_argument("--arch", default="hailo8", choices=("hailo8", "hailo8l", "hailo8r"))
    parser.add_argument("--opt-level", type=int, default=2, help="2+ fine-tunes on GPU; 0-1 run on CPU")
    parser.add_argument("--calib-batch", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=64, help="calibration frames kept out for the emulator check")
    parser.add_argument("--alls", nargs="*", default=[], help="extra model-script lines, e.g. 16-bit outputs")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    submodel = args.onnx.parent.name
    defaults = DEFAULT_CUTS.get(submodel, {})
    end, hw = args.end or defaults.get("end"), tuple(args.hw or defaults.get("hw", ()))
    if not end or len(hw) != 2:
        raise SystemExit(f"no default cut for {submodel!r}; pass --end and --hw")
    args.out.mkdir(parents=True, exist_ok=True)
    cut, spec = prepare(args.onnx, args.out, end, hw)
    print(f"prepared {cut}: {json.dumps(spec)}")
    if args.prepare_only:
        return

    from hailo_sdk_client import ClientRunner

    frames = np.load(args.calib)
    assert frames.dtype == np.uint8 and frames.shape[1:3] == hw, f"calibration is {frames.shape} {frames.dtype}"
    holdout, calib = frames[: args.holdout], frames[args.holdout :]
    script = alls(spec, args, len(calib))
    name = f"{args.onnx.parents[1].name}_{submodel}".replace("-", "_").replace(".", "_")
    started = time.time()

    runner = ClientRunner(hw_arch=args.arch)
    runner.translate_onnx_model(cut.as_posix(), name)
    runner.load_model_script(script)
    runner.optimize(calib.astype(np.float32))
    check = emulator_check(runner, cut, holdout, spec)
    runner.save_har((args.out / "model.har").as_posix())
    hef = runner.compile()
    (args.out / "model.hef").write_bytes(hef)

    import hailo_sdk_client

    (args.out / "model.json").write_text(
        json.dumps(
            {
                "source": args.onnx.as_posix(),
                "source_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
                "calib": {"path": args.calib.as_posix(), "frames": len(calib), "holdout": len(holdout)},
                "arch": args.arch,
                "dfc": getattr(hailo_sdk_client, "__version__", "unknown"),
                "cut": spec,
                "alls": script,
                "emulator": check,
                "hef_sha256": hashlib.sha256(hef).hexdigest(),
                "minutes": round((time.time() - started) / 60, 1),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out / 'model.hef'}")


if __name__ == "__main__":
    main()
