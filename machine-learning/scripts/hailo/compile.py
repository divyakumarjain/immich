"""Compile one of Immich's fused (v2) ONNX models to Hailo .hef files with the Dataflow Compiler (DFC).

Runs inside the Hailo AI Software Suite container. The steps:

1. prepare  for every graph the model ships (immich_model's dim sets: OCR detection compiles a canvas per aspect
            ratio for each max resolution, OCR recognition a graph per line width), pin the shape, cut the graph
            between its preprocess and its host-side tail, and inline the weights:
            uint8 NHWC -> Cast -> Transpose [-> Sub(mean) [-> Mul/Div(1/std)]] -> ...body... -> cut tensors -> tail
            The preprocess becomes the HEF's `normalization` layer (none where the export folded it into the first
            conv), so the HEF takes raw uint8 pixels. The tail is whatever only reshapes, activates, normalizes or
            decodes the body's last real layer (SCRFD's split/sigmoid, an embedding's L2 norm, DBNet's sigmoid, CTC's
            argmax): it stays on the host, where the session replays it from tail.onnx. Opset-23 fusions the DFC
            cannot parse (Attention, Gelu) are decomposed, and the body is relabeled as opset 17
2. optimize quantize against the calibration set (raw uint8 NHWC, from make_calib.py)
3. check    run the quantized emulator against ONNX on held-out images before anything reaches a device
4. compile  write model.hef plus model.json: the session's contract and what was compiled from what

    python compile.py --onnx buffalo_l/v2/detection/model.onnx --calib calib/buffalo_l/det_calib.npy --out out/
    python compile.py --onnx PP-OCRv5_mobile/detection/model.onnx --out out/ocr-det --prepare-only
    python compile.py --prepared out/ocr-det/res736/height736_width736 --calib ... --out ...

A model with one graph prepares into --out itself; one with several into --out/<variant>/<shape>/, and each is then
compiled on its own with --prepared. Preparing needs immich_model and an onnx new enough to read the export, so a
graph past opset 17 is prepared on the host and compiled with --prepared inside the suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import numpy_helper

DFC_OPSET = 17  # the newest default-domain opset the suite's parser and onnx 1.16 both take

# what only reshapes, activates, normalizes or decodes: the host replays these past the cut
HOST_OPS = {
    "Cast",
    "Squeeze",
    "Unsqueeze",
    "Reshape",
    "Flatten",
    "Transpose",
    "Split",
    "Sigmoid",
    "ReduceL2",
    "Clip",
    "Div",
    "ArgMax",
    "ReduceMax",
    "ReduceSum",
    "Sub",
    "Exp",
    "Reciprocal",
    "Softmax",
    "Identity",
}


def constant(model: onnx.ModelProto, name: str) -> NDArray[np.float32] | None:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t)
    return None


def host_cut(model: onnx.ModelProto) -> list[str]:
    """The tensors where the body hands over to the host: walking back from every output through HOST_OPS, the
    first tensors some other op produced (constants aside)."""
    produces = {output: node for node in model.graph.node for output in node.output}
    cut: list[str] = []
    seen: set[str] = set()
    frontier = [output.name for output in model.graph.output]
    while frontier:
        tensor = frontier.pop()
        if tensor in seen or not tensor or constant(model, tensor) is not None:
            continue
        seen.add(tensor)
        node = produces.get(tensor)
        if node is not None and node.op_type in HOST_OPS:
            frontier.extend(node.input)
        elif tensor not in cut:
            cut.append(tensor)
    return sorted(cut)


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


def lower(body: onnx.ModelProto) -> onnx.ModelProto:
    """Decompose the fusions the DFC cannot parse, then label the body with an opset it can."""
    ops = {node.op_type for node in body.graph.node if node.domain in ("", "ai.onnx")}
    opset = next(o.version for o in body.opset_import if o.domain in ("", "ai.onnx"))
    if opset <= DFC_OPSET:
        spell_out_convs(body)
        return body
    if "Attention" in ops:
        import onnx_ir as ir
        from immich_model.onnx.lowering import DecomposeAttentionPass

        body = ir.to_proto(DecomposeAttentionPass()(ir.from_proto(body)).model)
    decompose_gelu(body)
    downgrade(body)
    print(f"lowered opset {opset} -> {DFC_OPSET}: {sorted({n.op_type for n in body.graph.node})}")
    return body


def canonical_layouts(body: onnx.ModelProto) -> None:
    """Every Squeeze/Unsqueeze in the body as a Reshape to its (pinned, inferred) output shape: the DFC parser
    follows a tensor's layout through a Reshape, but loses it through a Squeeze, as where PP-OCR's SVTR neck turns
    its feature map into tokens and back (Hailo's own PP-OCR export writes both as Reshape)."""
    shapes = {
        value.name: [d.dim_value for d in value.type.tensor_type.shape.dim]
        for value in [*body.graph.value_info, *body.graph.output]
    }
    for node in list(body.graph.node):
        if node.op_type not in ("Squeeze", "Unsqueeze"):
            continue
        shape = shapes.get(node.output[0])
        if not shape or not all(shape):
            raise SystemExit(f"{node.name}: no static output shape to reshape to")
        target = f"{node.name or node.output[0]}_shape"
        body.graph.initializer.append(numpy_helper.from_array(np.array(shape, dtype=np.int64), target))
        reshape = onnx.helper.make_node(
            "Reshape", [node.input[0], target], list(node.output), name=f"{node.name}_reshape"
        )
        at = list(body.graph.node).index(node)
        body.graph.node.remove(node)
        body.graph.node.insert(at, reshape)
        print(f"rewrote {node.op_type} {node.name} as Reshape{shape}")


def decompose_gelu(body: onnx.ModelProto) -> None:
    """Gelu -> Div(x, sqrt 2) -> Erf -> Add(1) -> Mul(x) -> Mul(0.5): the exact chain the DFC folds back into its
    GELU activation (an Erf in any other arrangement, e.g. immich_model's RKNN-ordered 0.5x-first form, is rejected
    as an unsupported activation). approximate=tanh takes the erf form too."""
    for node in list(body.graph.node):
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
# attributes Resize gained at 18, each with the value that keeps opset 17's behavior
_RESIZE_18_DEFAULTS: dict[str, Any] = {"antialias": 0, "keep_aspect_ratio_policy": b"stretch"}


def downgrade(body: onnx.ModelProto) -> None:
    """Relabel an opset 18+ body as DFC_OPSET. onnx's version converter has no 18 -> 17 adapter for Split, and past
    17 only Split (num_outputs), the Reduce ops (axes as input) and Resize (antialias, aspect policy, axes) changed
    meaning; every other op in these graphs only widened its types. Those are rewritten to their opset-17 form, a
    use of anything opset 17 cannot express is refused, and the full checker proves the rest is valid opset 17."""
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
        elif node.op_type == "Resize":
            kept = []
            for attribute in node.attribute:
                value = onnx.helper.get_attribute_value(attribute)
                if attribute.name == "axes" or (
                    attribute.name in _RESIZE_18_DEFAULTS and value != _RESIZE_18_DEFAULTS[attribute.name]
                ):
                    raise SystemExit(f"{node.name}: Resize {attribute.name}={value!r} has no opset-17 form")
                if attribute.name == "coordinate_transformation_mode" and value == b"half_pixel_symmetric":
                    raise SystemExit(f"{node.name}: half_pixel_symmetric has no opset-17 form")
                if attribute.name not in _RESIZE_18_DEFAULTS:
                    kept.append(attribute)
            del node.attribute[:]
            node.attribute.extend(kept)
    for opset in body.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = DFC_OPSET
    spell_out_convs(body)
    body.ir_version = 8  # what opset 17 shipped with; the suite's onnx 1.16 reads up to 10
    onnx.checker.check_model(body, full_check=True)


def spell_out_convs(body: onnx.ModelProto) -> None:
    """Write out every optional Conv/ConvTranspose attribute: the DFC parser reads kernel_shape and friends off the
    node, where ONNX lets an exporter leave them to the weights and defaults (and the dynamo exporter does)."""
    for node in body.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        weight = constant(body, node.input[1])
        if weight is None:
            raise SystemExit(f"{node.name}: weights are computed at runtime")
        spatial = weight.ndim - 2
        have = {a.name for a in node.attribute}
        defaults: dict[str, Any] = {
            "kernel_shape": list(weight.shape[2:]),
            "strides": [1] * spatial,
            "dilations": [1] * spatial,
            "pads": [0] * (2 * spatial),
            "group": 1,
        }
        if node.op_type == "ConvTranspose":
            defaults["output_padding"] = [0] * spatial
        if have & {"auto_pad", "output_shape"}:
            defaults.pop("pads")
        for name, value in defaults.items():
            if name not in have:
                node.attribute.append(onnx.helper.make_attribute(name, value))


def fold_batch_plumbing(model: onnx.ModelProto, frame: tuple[int, ...]) -> None:
    """A branch off the image that ends in a Mul by zero is how an export broadcasts a learned query over the batch
    (SigLIP's attention pool): its value never depends on the pixels, so for one pinned frame it is a constant."""
    import onnxruntime as ort

    image = model.graph.input[0]
    folded_any = False
    for node in list(model.graph.node):
        if node.op_type != "Mul":
            continue
        zero = next((c for i in node.input if (c := constant(model, i)) is not None and not c.any()), None)
        if zero is None:
            continue
        branch = onnx.utils.Extractor(model).extract_model([image.name], [node.output[0]])
        session = ort.InferenceSession(branch.SerializeToString(), providers=["CPUExecutionProvider"])
        (value,) = session.run(None, {image.name: np.zeros(frame, dtype=np.uint8)})
        folded = onnx.helper.make_node(
            "Constant", [], [node.output[0]], name=f"{node.name}_folded", value=numpy_helper.from_array(value)
        )
        model.graph.node.insert(list(model.graph.node).index(node), folded)
        model.graph.node.remove(node)
        folded_any = True
        print(f"folded {node.name} to a constant {list(value.shape)}")
    if not folded_any:
        return
    # what fed only the folded branches now feeds nothing
    outputs = {output.name for output in model.graph.output}
    while dead := [
        node
        for node in model.graph.node
        if not any(o in outputs for o in node.output)
        and not any(o in other.input for o in node.output for other in model.graph.node)
    ]:
        for node in dead:
            model.graph.node.remove(node)


def graphs(source: Path) -> list[tuple[str, dict[str, int]]]:
    """(variant, dims) for every graph the model ships, as immich_model's RKNN exporter enumerates them."""
    try:
        from immich_model.constants import dim_sets_of, variant_dir
    except ImportError:  # inside the suite: only a graph with nothing left to pin can be prepared here
        free = [d.dim_param for d in onnx.load(source.as_posix(), load_external_data=False).graph.input[0].type
                .tensor_type.shape.dim[1:3] if d.dim_param]  # fmt: skip
        if free:
            raise SystemExit(f"{source} leaves {free} free: prepare it on the host, where immich_model is installed")
        return [("", {})]
    sets = dim_sets_of(source)
    return [(variant_dir(sets, group.dims[0]), dict(dims)) for group in sets for dims in group.dims]


def graph_dir(root: Path, every: list[tuple[str, dict[str, int]]], variant: str, dims: Mapping[str, int]) -> Path:
    if len(every) == 1:
        return root
    from immich_model.constants import dims_label

    # the layout the session reads: <variant>/model.json indexes a <variant>/<shape>/ per graph (no variant level
    # for a model that has one set, as rknpu/<soc>/model.rknn has none)
    return root / variant / (dims_label(dims) or "default")


def prepare(
    source: Path, work: Path, dims: Mapping[str, int], end: list[str] | None, indexed: bool = False
) -> tuple[Path, dict[str, Any]]:
    model = onnx.load(source.as_posix())  # pulls the safetensors sidecar into memory
    image = model.graph.input[0]
    named = image.type.tensor_type.shape.dim
    frame = tuple(1 if axis == 0 else (d.dim_value or dims[d.dim_param]) for axis, d in enumerate(named))
    fold_batch_plumbing(model, frame)
    start, mean, std = preprocess(model)
    end = end or host_cut(model)
    body = onnx.utils.Extractor(model).extract_model([start], end)
    # pin the start tensor to one NCHW frame; the DFC wants static shapes
    for dim, value in zip(body.graph.input[0].type.tensor_type.shape.dim, (1, frame[3], frame[1], frame[2])):
        dim.Clear()
        dim.dim_value = value
    body = lower(onnx.shape_inference.infer_shapes(body))
    canonical_layouts(body)
    body = onnx.shape_inference.infer_shapes(body, strict_mode=True)
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{source.parent.name}.cut.onnx"
    onnx.save(body, path.as_posix())  # weights inline: one self-contained file for the parser
    # what the session replays on the host past the cut, from the cut tensors to the graph's own outputs
    tail = onnx.utils.Extractor(model).extract_model(end, [output.name for output in model.graph.output])
    onnx.save(tail, (work / "tail.onnx").as_posix())
    spec = {
        "start": start,
        "end": end,
        "mean": mean,
        "std": std,
        "hw": [frame[1], frame[2]],
        "input": image.name,
        "dims": [dict(dims)],  # what the session offers, as an rknn binary declares the dims it was pinned to
        "metadata": {prop.key: prop.value for prop in model.metadata_props},  # e.g. OCR's CTC character set
        "source": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cut_onnx": path.name,
        "indexed": indexed,  # one of several shapes: compiling it also lists it in the parent's model.json index
    }
    (work / "spec.json").write_text(json.dumps(spec, indent=2))
    print(f"prepared {path}: {json.dumps({k: v for k, v in spec.items() if k != 'metadata'})}")
    return path, spec


def alls(spec: dict[str, Any], args: argparse.Namespace, calib_size: int) -> str:
    identity = all(m == 0 for m in spec["mean"]) and all(s == 1 for s in spec["std"])
    lines = [
        # a name no parsed layer takes: the parser already calls batch norms normalization1..N (PP-OCR has them)
        *([] if identity else [f"input_normalization = normalization({spec['mean']}, {spec['std']})"]),
        f"model_optimization_config(calibration, batch_size={args.calib_batch}, calibset_size={calib_size})",
        f"model_optimization_flavor(optimization_level={args.opt_level}, compression_level=0)",
        # an exhaustive search for the fastest placement: worth it for a release binary, but it can take hours
        # on a multi-context model, so a build that only has to prove the pipeline leaves it at the default
        *(["performance_param(compiler_optimization_level=max)"] if args.max_performance else []),
        *args.alls,
    ]
    return "\n".join(lines) + "\n"


def output_map(runner: Any, cut: Path) -> dict[str, str]:
    """HEF output vstream -> the cut tensor it carries, from the parser's record (in the compiled HN) of the nodes
    behind each output layer. The same names HailoRT reports from the .hef, without needing its C library here."""
    hn = runner.get_hn_dict()
    layers = hn["layers"]
    produces = {node.name: node.output[0] for node in onnx.load(cut.as_posix(), load_external_data=False).graph.node}
    mapping = {}
    for layer in layers.values():
        if layer["type"] != "output_layer":
            continue
        (source,) = layer["input"]
        vstream = source if "/" in source else f"{hn['name']}/{source}"
        found = [produces[name] for name in layers[source].get("original_names", []) if name in produces]
        if not found:
            raise RuntimeError(f"{vstream} names no node of {cut.name}")
        mapping[vstream] = found[-1]
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
    parser.add_argument("--onnx", type=Path, help="a v2 submodel: <repo>/<submodel>/model.onnx")
    parser.add_argument("--prepared", type=Path, help="a graph directory --prepare-only wrote, to compile as is")
    parser.add_argument("--calib", type=Path, help="(N, H, W, 3) uint8 from make_calib.py, sized for the graph")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", nargs="*", help="cut tensors; by default where the host tail begins")
    parser.add_argument("--variant", help="prepare only this variant (e.g. res736)")
    parser.add_argument("--arch", default="hailo8", choices=("hailo8", "hailo8l", "hailo8r"))
    parser.add_argument("--opt-level", type=int, default=2, help="2+ fine-tunes on GPU; 0-1 run on CPU")
    parser.add_argument("--calib-batch", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=64, help="calibration frames kept out for the emulator check")
    parser.add_argument("--alls", nargs="*", default=[], help="extra model-script lines, e.g. 16-bit outputs")
    parser.add_argument("--max-performance", action="store_true", help="search exhaustively for the fastest layout")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.prepared:
        spec = json.loads((args.prepared / "spec.json").read_text())
        cut = args.prepared / spec["cut_onnx"]
    elif args.onnx is None:
        raise SystemExit("pass --onnx, or --prepared with a graph directory --prepare-only wrote")
    else:
        every = graphs(args.onnx)
        chosen = [(v, d) for v, d in every if args.variant is None or v == args.variant]
        indexed = len(every) > 1
        prepared = [prepare(args.onnx, graph_dir(args.out, every, v, d), d, args.end, indexed) for v, d in chosen]
        if args.prepare_only:
            return
        if len(prepared) != 1:
            raise SystemExit(f"{len(prepared)} graphs prepared: compile each with --prepared and its own --calib")
        cut, spec = prepared[0]
    if args.prepare_only:
        return
    compile_graph(args, cut, spec)


def compile_graph(args: argparse.Namespace, cut: Path, spec: dict[str, Any]) -> None:
    from hailo_sdk_client import ClientRunner

    args.out.mkdir(parents=True, exist_ok=True)
    frames = np.load(args.calib)
    hw = tuple(spec["hw"])
    assert frames.dtype == np.uint8 and frames.shape[1:3] == hw, f"calibration is {frames.shape} {frames.dtype}"
    holdout, calib = frames[: args.holdout], frames[args.holdout :]
    script = alls(spec, args, len(calib))
    source = Path(spec["source"])
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
    if not (args.out / "tail.onnx").exists():  # compiled somewhere other than where it was prepared
        (args.out / "tail.onnx").write_bytes((cut.parent / "tail.onnx").read_bytes())

    import hailo_sdk_client

    # the session's contract first (immich_ml.sessions.hailo reads it), the build record under it
    (args.out / "model.json").write_text(
        json.dumps(
            {
                "dims": spec["dims"],
                "input": spec["input"],
                "cut": output_map(runner, cut),
                "metadata": spec.get("metadata", {}),
                "build": {
                    "source": spec["source"],
                    "source_sha256": spec["source_sha256"],
                    "calib": {"path": args.calib.as_posix(), "frames": len(calib), "holdout": len(holdout)},
                    "arch": args.arch,
                    "dfc": getattr(hailo_sdk_client, "__version__", "unknown"),
                    "cut": {k: v for k, v in spec.items() if k != "metadata"},
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
    if spec.get("indexed"):
        index_graph(args.out)


def index_graph(graph: Path) -> None:
    """List a compiled shape in its variant's model.json, the entry point the session routes shapes from."""
    entry = graph.parent / "model.json"
    graphs = set(json.loads(entry.read_text()).get("graphs", [])) if entry.exists() else set()
    entry.write_text(json.dumps({"graphs": sorted(graphs | {graph.name})}, indent=2))
    print(f"indexed {graph.name} in {entry}")


if __name__ == "__main__":
    main()
