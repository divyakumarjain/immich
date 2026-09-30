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


@cache
def _vdevice() -> Any:
    """One VDevice per process; the scheduler lets several configured HEFs share it without manual activation."""
    import hailo_platform as hpf

    params = hpf.VDevice.create_params()
    params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
    return hpf.VDevice(params)


def _node(info: Any) -> HefNode:
    return HefNode(
        name=info.name,
        shape=tuple(info.shape),
        dtype=str(info.format.type).rsplit(".", 1)[-1],
        qp_scale=float(info.quant_info.qp_scale),
        qp_zp=float(info.quant_info.qp_zp),
    )


class HefModel:
    """Float32 in, float32 out: HailoRT quantizes inputs and dequantizes outputs with the HEF's own quant_info."""

    def __init__(self, path: Path) -> None:
        import hailo_platform as hpf

        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        hef = hpf.HEF(path.as_posix())
        configure = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        self.network_group = _vdevice().configure(hef, configure)[0]
        self.inputs = [_node(info) for info in hef.get_input_vstream_infos()]
        self.outputs = [_node(info) for info in hef.get_output_vstream_infos()]
        self._pipeline = hpf.InferVStreams(
            self.network_group,
            hpf.InputVStreamParams.make(self.network_group, format_type=hpf.FormatType.FLOAT32),
            hpf.OutputVStreamParams.make(self.network_group, format_type=hpf.FormatType.FLOAT32),
        )
        self._pipeline.__enter__()

    def run(self, feeds: dict[str, NDArray[Any]]) -> dict[str, NDArray[np.float32]]:
        """`feeds` are batched (N, *node.shape); each output comes back batched the same way."""
        return self._pipeline.infer({name: np.ascontiguousarray(a, dtype=np.float32) for name, a in feeds.items()})

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
