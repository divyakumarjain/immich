"""A minimal HailoRT wrapper that runs a .hef the way an onnxruntime session runs a .onnx; no immich_ml imports."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HefNode:
    name: str
    shape: tuple[int, ...]  # per frame, as HailoRT reports it (HWC for images, C for vectors)
    dtype: str  # the on-chip type, before HailoRT (de)quantizes to the float32 this wrapper exchanges
    qp_scale: float
    qp_zp: float
    original: tuple[str, ...]  # the source graph's nodes fused into this one; the last produces its tensor


@cache
def _vdevice() -> Any:
    """One VDevice per process; the scheduler lets several configured HEFs share it without manual activation."""
    import hailo_platform as hpf

    params = hpf.VDevice.create_params()
    params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
    return hpf.VDevice(params)


def _node(hef: Any, info: Any) -> HefNode:
    try:
        original = tuple(hef.get_original_names_from_vstream_name(info.name))
    except Exception:  # inputs, and HEFs built without the parser's name records
        original = ()
    return HefNode(
        name=info.name,
        shape=tuple(info.shape),
        dtype=str(info.format.type).rsplit(".", 1)[-1],
        qp_scale=float(info.quant_info.qp_scale),
        qp_zp=float(info.quant_info.qp_zp),
        original=original,
    )


class HefModel:
    """Float32 in, float32 out by default, HailoRT (de)quantizing with the HEF's own quant_info.

    `batch_size` is how many frames the device runs per pass. A multi-context HEF (one too big to stay resident)
    reloads every context per pass, so batch 1 pays that for each frame. `raw_input` hands HailoRT the uint8
    pixels a folded-normalization HEF takes natively; `raw_output` takes the quantized outputs and dequantizes
    them here, skipping HailoRT's per-element float conversion."""

    def __init__(
        self, path: Path, batch_size: int | None = None, raw_input: bool = False, raw_output: bool = False
    ) -> None:
        import hailo_platform as hpf

        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        hef = hpf.HEF(path.as_posix())
        configure = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        if batch_size:
            for name in hef.get_network_group_names():
                configure[name].batch_size = batch_size
        self.network_group = _vdevice().configure(hef, configure)[0]
        self.contexts = hef.get_network_groups_infos()[0].is_multi_context
        self.inputs = [_node(hef, info) for info in hef.get_input_vstream_infos()]
        self.outputs = [_node(hef, info) for info in hef.get_output_vstream_infos()]
        self.raw_input = raw_input and all(node.dtype == "UINT8" for node in self.inputs)
        self.raw_output = raw_output
        in_format = hpf.FormatType.UINT8 if self.raw_input else hpf.FormatType.FLOAT32
        out_format = hpf.FormatType.AUTO if raw_output else hpf.FormatType.FLOAT32
        self._pipeline = hpf.InferVStreams(
            self.network_group,
            hpf.InputVStreamParams.make(self.network_group, format_type=in_format),
            hpf.OutputVStreamParams.make(self.network_group, format_type=out_format),
        )
        self._pipeline.__enter__()

    def run(self, feeds: dict[str, NDArray[Any]]) -> dict[str, NDArray[np.float32]]:
        """`feeds` are batched (N, *node.shape); each output comes back batched the same way."""
        dtype = np.uint8 if self.raw_input else np.float32
        outputs: dict[str, NDArray[np.float32]] = self._pipeline.infer(
            {name: np.ascontiguousarray(a, dtype=dtype) for name, a in feeds.items()}
        )
        if not self.raw_output:
            return outputs
        quant = {node.name: node for node in self.outputs}
        return {
            name: (value.astype(np.float32) - quant[name].qp_zp) * quant[name].qp_scale
            for name, value in outputs.items()
        }

    def close(self) -> None:
        self._pipeline.__exit__(None, None, None)

    def describe(self) -> dict[str, Any]:
        """Identity of the artifact: two downloads with equal sha/quant are the same compile, not a new one."""
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "inputs": [vars(node) for node in self.inputs],
            "outputs": [vars(node) for node in self.outputs],
        }
