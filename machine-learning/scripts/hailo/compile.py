"""Compile one of Immich's fused (v2) ONNX models to a Hailo .hef with the Dataflow Compiler (DFC).

Runs inside the Hailo AI Software Suite container. The steps:

1. prepare  cut the graph between its preprocess and its host-side tail, pin the shape, inline the weights:
            uint8 NHWC -> Cast -> Transpose [-> Sub(mean) [-> Mul/Div(1/std)]] -> ...body... -> cut tensors -> tail
            The preprocess becomes the HEF's `normalization` layer (none where the export folded it into the first
            conv), so the HEF takes raw uint8 pixels; the tail (reshape/split/sigmoid, L2 norm) stays on the host,
            where compare.py's HostTail replays it from the ONNX. Opset-23 fusions the DFC cannot parse (Attention,
            Gelu) are decomposed with immich_model's own lowerings, and the body is converted down to opset 17
2. optimize quantize against the calibration set (raw uint8 NHWC, from make_calib.py)
3. check    run the quantized emulator against ONNX on held-out images before anything reaches a device
4. compile  write model.hef plus model.json: what was compiled from what, with which settings

    python compile.py --onnx buffalo_l/v2/detection/model.onnx --calib calib/buffalo_l/det_calib.npy --out out/
    python compile.py --onnx ... --prepare-only   # step 1 alone; needs onnx and immich_model, not the DFC
    python compile.py --prepared out/visual --calib ... --out out/visual   # steps 2-4 on a step-1 directory

A graph that needs lowering is prepared on the host (the suite's onnx is too old for opset 23), then compiled with
--prepared inside the suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import numpy_helper

# where Immich's v2 face graphs hand over to the host: the per-stride fused SCRFD heads (score|box|kps,
# 15 channels per anchor) before their Transpose/Reshape/Split/Sigmoid, and ArcFace's Gemm before its L2 norm
DEFAULT_CUTS: dict[str, dict[str, Any]] = {
    "detection": {"end": ["val_0", "val_4", "val_8"], "hw": (640, 640)},
    "recognition": {"end": ["683"], "hw": (112, 112)},
}
DFC_OPSET = 17  # the newest default-domain opset the suite's parser and onnx 1.16 both take


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
    if transpose is None:
        raise SystemExit("not a fused uint8 graph: expected Cast -> Transpose at its input")
    channels = image.type.tensor_type.shape.dim[3].dim_value
    sub = sole(transpose.output[0], "Sub")
    if sub is None:  # the export folded the whole normalization into the first conv
        return transpose.output[0], [0.0] * channels, [1.0] * channels
    mean = constant(model, sub.input[1])
    std: NDArray[Any] = np.ones(1, dtype=np.float32)
    start = sub.output[0]
    scale = sole(start, "Mul") or sole(start, "Div")
    if scale is not None and (factor := constant(model, scale.input[1])) is not None and factor.size in (1, channels):
        std = 1 / factor if scale.op_type == "Mul" else factor
        start = scale.output[0]
    assert mean is not None, "the Sub's mean is not a constant"
    return start, np.broadcast_to(mean.ravel(), channels).tolist(), np.broadcast_to(std.ravel(), channels).tolist()


def l2_tail_start(model: onnx.ModelProto) -> str | None:
    """Where an embedding's trailing L2 normalization (and any squeeze before it) begins, walking back from the
    single output: x -> [Squeeze/Reshape/Flatten] -> ReduceL2 -> Clip -> Div. A CLIP encoder's natural cut."""
    produces = {output: node for node in model.graph.node for output in node.output}
    if len(model.graph.output) != 1 or (div := produces.get(model.graph.output[0].name)) is None:
        return None
    if div.op_type != "Div":
        return None
    tensor: str = div.input[0]
    while (node := produces.get(tensor)) is not None and node.op_type in ("Squeeze", "Reshape", "Flatten"):
        tensor = node.input[0]
    return tensor


def lower(body: onnx.ModelProto) -> onnx.ModelProto:
    """Decompose the fusions the DFC cannot parse, then label the body with an opset it can."""
    ops = {node.op_type for node in body.graph.node if node.domain in ("", "ai.onnx")}
    opset = next(o.version for o in body.opset_import if o.domain in ("", "ai.onnx"))
    if opset <= DFC_OPSET:
        return body
    if "Attention" in ops:
        import onnx_ir as ir
        from immich_model.onnx.lowering import DecomposeAttentionPass

        body = ir.to_proto(DecomposeAttentionPass()(ir.from_proto(body)).model)
    decompose_gelu(body)
    downgrade(body)
    print(f"lowered opset {opset} -> {DFC_OPSET}: {sorted({n.op_type for n in body.graph.node})}")
    return body


def decompose_gelu(body: onnx.ModelProto) -> None:
    """Gelu -> Div(x, sqrt 2) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5): the exact chain the DFC folds back into its
    GELU activation (an Erf in any other arrangement, e.g. immich_model's RKNN-ordered 0.5x-first form, is rejected
    as an unsupported activation). approximate=tanh takes the erf form too."""
    nodes = list(body.graph.node)
    for node in nodes:
        if node.op_type != "Gelu":
            continue
        x, (y,) = node.input[0], node.output
        stem = node.name or y

        def const(suffix: str, value: float) -> str:
            name = f"{stem}_{suffix}"
            body.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.float32), name))
            return name

        chain = [
            onnx.helper.make_node("Div", [x, const("sqrt2", math.sqrt(2.0))], [f"{stem}_div"], name=f"{stem}_div"),
            onnx.helper.make_node("Erf", [f"{stem}_div"], [f"{stem}_erf"], name=f"{stem}_erf"),
            onnx.helper.make_node("Add", [f"{stem}_erf", const("one", 1.0)], [f"{stem}_add"], name=f"{stem}_add"),
            onnx.helper.make_node("Mul", [x, f"{stem}_add"], [f"{stem}_mul"], name=f"{stem}_mul"),
            onnx.helper.make_node("Mul", [f"{stem}_mul", const("half", 0.5)], [y], name=f"{stem}_half"),
        ]
        at = list(body.graph.node).index(node)
        body.graph.node.remove(node)
        for offset, part in enumerate(chain):
            body.graph.node.insert(at + offset, part)


# ops whose axes moved from an attribute to an input at opset 18 (ReduceSum already did at 13)
_REDUCE_AXES_18 = {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSumExp"}


def downgrade(body: onnx.ModelProto) -> None:
    """Relabel an opset 18+ body as DFC_OPSET. onnx's version converter has no 18 -> 17 adapter for Split, and
    past 17 only Split (num_outputs) and the Reduce ops (axes as input) changed meaning; every other op in these
    graphs only widened its types. Those two are rewritten to their opset-17 form, and the full checker proves
    what is left is valid opset 17."""
    for node in body.graph.node:
        if node.op_type == "Split":
            # equal parts, one per output, is opset 17's default when no `split` input is given
            kept = [a for a in node.attribute if a.name != "num_outputs"]
            del node.attribute[:]
            node.attribute.extend(kept)
        elif node.op_type in _REDUCE_AXES_18 and len(node.input) > 1 and node.input[1]:
            axes = constant(body, node.input[1])
            if axes is None:
                raise SystemExit(f"{node.name}: axes are computed at runtime, which opset {DFC_OPSET} cannot take")
            del node.input[1:]
            node.attribute.append(onnx.helper.make_attribute("axes", axes.astype(np.int64).tolist()))
    for opset in body.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = DFC_OPSET
    spell_out_convs(body)
    body.ir_version = 8  # what opset 17 shipped with; the suite's onnx 1.16 reads up to 10
    onnx.checker.check_model(body, full_check=True)


def spell_out_convs(body: onnx.ModelProto) -> None:
    """Write out every optional Conv attribute: the DFC parser reads kernel_shape and friends off the node, where
    ONNX lets an exporter leave them to the weights and defaults (and the dynamo exporter does)."""
    for node in body.graph.node:
        if node.op_type != "Conv":
            continue
        weight = constant(body, node.input[1])
        if weight is None:
            raise SystemExit(f"{node.name}: weights are computed at runtime")
        spatial = weight.ndim - 2
        have = {a.name for a in node.attribute}
        defaults = {
            "kernel_shape": list(weight.shape[2:]),
            "strides": [1] * spatial,
            "dilations": [1] * spatial,
            "pads": [0] * (2 * spatial),
            "group": 1,
        }
        if "auto_pad" in have:
            defaults.pop("pads")
        for name, value in defaults.items():
            if name not in have:
                node.attribute.append(onnx.helper.make_attribute(name, value))


def fold_batch_plumbing(model: onnx.ModelProto, hw: tuple[int, int]) -> None:
    """A branch off the image that ends in a Mul by zero is how an export broadcasts a learned query over the batch
    (SigLIP's attention pool): its value never depends on the pixels, so for one pinned frame it is a constant."""
    import onnxruntime as ort

    image = model.graph.input[0]
    for node in list(model.graph.node):
        if node.op_type != "Mul":
            continue
        zero = next((c for i in node.input if (c := constant(model, i)) is not None and not c.any()), None)
        if zero is None:
            continue
        branch = onnx.utils.Extractor(model).extract_model([image.name], [node.output[0]])
        session = ort.InferenceSession(branch.SerializeToString(), providers=["CPUExecutionProvider"])
        channels = image.type.tensor_type.shape.dim[3].dim_value
        (value,) = session.run(None, {image.name: np.zeros((1, *hw, channels), dtype=np.uint8)})
        folded = onnx.helper.make_node(
            "Constant", [], [node.output[0]], name=f"{node.name}_folded", value=numpy_helper.from_array(value)
        )
        model.graph.node.insert(list(model.graph.node).index(node), folded)
        model.graph.node.remove(node)
        print(f"folded {node.name} to a constant {list(value.shape)}")
    # what fed only the folded branches now feeds nothing
    while dead := [
        n
        for n in model.graph.node
        if not any(o in {o.name for o in model.graph.output} for o in n.output)
        and not any(o in i for o in n.output for m in model.graph.node for i in [m.input])
    ]:
        for n in dead:
            model.graph.node.remove(n)


def prepare(source: Path, work: Path, end: list[str], hw: tuple[int, int]) -> tuple[Path, dict[str, Any]]:
    model = onnx.load(source.as_posix())  # pulls the safetensors sidecar into memory
    fold_batch_plumbing(model, hw)
    start, mean, std = preprocess(model)
    body = onnx.utils.Extractor(model).extract_model([start], end)
    # pin the start tensor to one NCHW frame; the DFC wants static shapes
    dims = body.graph.input[0].type.tensor_type.shape.dim
    for dim, value in zip(dims, (1, len(mean), *hw)):
        dim.Clear()
        dim.dim_value = value
    body = lower(onnx.shape_inference.infer_shapes(body))
    body = onnx.shape_inference.infer_shapes(body, strict_mode=True)
    path = work / f"{source.parent.name}.cut.onnx"
    onnx.save(body, path.as_posix())  # weights inline: one self-contained file for the parser
    # what the session replays on the host past the cut, from the cut tensors to the graph's own outputs
    tail = onnx.utils.Extractor(model).extract_model(end, [output.name for output in model.graph.output])
    onnx.save(tail, (work / "tail.onnx").as_posix())
    image = model.graph.input[0]
    # a source graph that left height/width free declares the size it was pinned to, as an rknn binary does
    free = any(dim.dim_param for dim in image.type.tensor_type.shape.dim[1:3])
    spec = {
        "start": start,
        "end": end,
        "mean": mean,
        "std": std,
        "hw": list(hw),
        "input": image.name,
        "dims": [{"height": hw[0], "width": hw[1]}] if free else [{}],
        "source": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cut_onnx": path.name,
    }
    (work / "spec.json").write_text(json.dumps(spec, indent=2))
    return path, spec


def alls(spec: dict[str, Any], args: argparse.Namespace, calib_size: int) -> str:
    identity = all(m == 0 for m in spec["mean"]) and all(s == 1 for s in spec["std"])
    lines = [
        *([] if identity else [f"normalization1 = normalization({spec['mean']}, {spec['std']})"]),
        f"model_optimization_config(calibration, batch_size={args.calib_batch}, calibset_size={calib_size})",
        f"model_optimization_flavor(optimization_level={args.opt_level}, compression_level=0)",
        # an exhaustive search for the fastest placement: worth it for a release binary, but it can take hours
        # on a multi-context model, so a build that only has to prove the pipeline leaves it at the default
        *(["performance_param(compiler_optimization_level=max)"] if args.max_performance else []),
        *args.alls,
    ]
    return "\n".join(lines) + "\n"


def output_map(hef_path: Path, cut: Path) -> dict[str, str]:
    """HEF output vstream -> the cut tensor it carries, from the parser's record of the nodes behind each output."""
    import hailo_platform as hpf

    hef = hpf.HEF(hef_path.as_posix())
    produces = {node.name: node.output[0] for node in onnx.load(cut.as_posix(), load_external_data=False).graph.node}
    mapping = {}
    for info in hef.get_output_vstream_infos():
        found = [produces[name] for name in hef.get_original_names_from_vstream_name(info.name) if name in produces]
        if not found:
            raise RuntimeError(f"{info.name} names no node of {cut.name}")
        mapping[info.name] = found[-1]
    return mapping


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
    parser.add_argument("--onnx", type=Path, help="a v2 submodel: <repo>/<detection|recognition|visual>/model.onnx")
    parser.add_argument("--prepared", type=Path, help="a directory --prepare-only wrote, to compile as is")
    parser.add_argument("--calib", type=Path, help="(N, H, W, 3) uint8 from make_calib.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", nargs="*", help="cut tensors; defaults by submodel")
    parser.add_argument("--hw", type=int, nargs=2, help="input height width; defaults by submodel")
    parser.add_argument("--arch", default="hailo8", choices=("hailo8", "hailo8l", "hailo8r"))
    parser.add_argument("--opt-level", type=int, default=2, help="2+ fine-tunes on GPU; 0-1 run on CPU")
    parser.add_argument("--calib-batch", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=64, help="calibration frames kept out for the emulator check")
    parser.add_argument("--alls", nargs="*", default=[], help="extra model-script lines, e.g. 16-bit outputs")
    parser.add_argument("--max-performance", action="store_true", help="search exhaustively for the fastest layout")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepared:
        spec = json.loads((args.prepared / "spec.json").read_text())
        cut, hw = args.prepared / spec["cut_onnx"], tuple(spec["hw"])
        source = Path(spec["source"])
    else:
        if args.onnx is None:
            raise SystemExit("pass --onnx, or --prepared with a directory --prepare-only wrote")
        source = args.onnx
        submodel = source.parent.name
        defaults = DEFAULT_CUTS.get(submodel, {})
        full = onnx.load(source.as_posix(), load_external_data=False)
        end: list[str] = args.end or defaults.get("end", []) or [t for t in [l2_tail_start(full)] if t]
        static = [d.dim_value for d in full.graph.input[0].type.tensor_type.shape.dim[1:3]]
        hw = tuple(args.hw or defaults.get("hw", ()) or (static if all(static) else ()))
        if not end or len(hw) != 2:
            raise SystemExit(f"no default cut for {submodel!r}; pass --end and --hw")
        cut, spec = prepare(source, args.out, end, hw)
        print(f"prepared {cut}: {json.dumps(spec)}")
    if args.prepare_only:
        return

    from hailo_sdk_client import ClientRunner

    frames = np.load(args.calib)
    assert frames.dtype == np.uint8 and frames.shape[1:3] == hw, f"calibration is {frames.shape} {frames.dtype}"
    holdout, calib = frames[: args.holdout], frames[args.holdout :]
    script = alls(spec, args, len(calib))
    name = f"{source.parents[1].name}_{source.parent.name}".replace("-", "_").replace(".", "_")
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

    # the session's contract first (immich_ml.sessions.hailo reads dims/input/cut), the build record under it
    (args.out / "model.json").write_text(
        json.dumps(
            {
                "dims": spec["dims"],
                "input": spec["input"],
                "cut": output_map(args.out / "model.hef", cut),
                "build": {
                    "source": spec["source"],
                    "source_sha256": spec["source_sha256"],
                    "calib": {"path": args.calib.as_posix(), "frames": len(calib), "holdout": len(holdout)},
                    "arch": args.arch,
                    "dfc": getattr(hailo_sdk_client, "__version__", "unknown"),
                    "cut": spec,
                    "alls": script,
                    "emulator": check,
                    "hef_sha256": hashlib.sha256(hef).hexdigest(),
                    "minutes": round((time.time() - started) / 60, 1),
                },
            },
            indent=2,
        )
    )
    print(f"wrote {args.out / 'model.hef'}")


if __name__ == "__main__":
    main()
