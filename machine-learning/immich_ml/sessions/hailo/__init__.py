"""Hailo-8/8L sessions. The .hef runs the model's body on the NPU; what the compiler left to the host (the ONNX past
its cut, e.g. SCRFD's head flattening or ArcFace's L2 norm) runs on CPU from tail.onnx, so outputs reach the model
classes named and shaped exactly as the ONNX would return them."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple

import numpy as np
import onnxruntime as ort
import orjson
from numpy.typing import NDArray

from immich_ml.config import log, settings
from immich_ml.schemas import ModelInput, SessionNode, Shape


class HailoNode(NamedTuple):
    name: str
    shape: tuple[int, ...]


def _architecture() -> str | None:
    """The chip a binary must be compiled for, or None without the runtime or a device."""
    try:
        import hailo_platform as hpf

        devices = hpf.Device.scan()
        if not devices:
            return None
        device = hpf.Device(devices[0])
        try:
            return str(device.control.identify().device_architecture.name).lower()
        finally:
            device.release()
    except Exception as e:  # no hailort, no driver, or a device another process holds
        log.debug(f"Hailo is not available: {e}")
        return None


architecture = _architecture() if settings.hailo else None
is_available = architecture is not None


def model_path(model_dir: Path, variant: str = "") -> Path:
    return model_dir / "hailo" / (architecture or "") / variant / "model.hef"


@cache
def _vdevice() -> Any:
    """One per process. The scheduler time-shares the device between every configured binary, so a detector and a
    recognizer stay loaded together instead of being activated and deactivated around each request."""
    import hailo_platform as hpf

    params = hpf.VDevice.create_params()
    params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
    return hpf.VDevice(params)


def _layout(value: NDArray[np.float32], rank: int) -> NDArray[np.float32]:
    """HailoRT returns NHWC feature maps and flat vectors; the tail reads NCHW, or rows past a Gemm."""
    return value.transpose(0, 3, 1, 2) if rank == 4 else value.reshape(len(value), -1)


class HailoSession:
    def __init__(self, model_path: Path) -> None:
        import hailo_platform as hpf

        log.info(f"Loading Hailo model from {model_path}")
        contract = orjson.loads(model_path.with_suffix(".json").read_bytes())
        hef = hpf.HEF(model_path.as_posix())
        # a binary too big to stay resident reloads its contexts every pass, which a device batch amortizes;
        # one that fits gains nothing from it, but still takes any number of frames in one call
        multi_context = hef.get_network_groups_infos()[0].is_multi_context
        configure = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        for name in hef.get_network_group_names():
            configure[name].batch_size = settings.hailo_batch_size if multi_context else 1
        self.network_group = _vdevice().configure(hef, configure)[0]
        # raw uint8 in (the binary normalizes), float32 out: both measured fastest on a Raspberry Pi 5
        self.pipeline = hpf.InferVStreams(
            self.network_group,
            hpf.InputVStreamParams.make(self.network_group, format_type=hpf.FormatType.UINT8),
            hpf.OutputVStreamParams.make(self.network_group, format_type=hpf.FormatType.FLOAT32),
        )
        self.pipeline.__enter__()
        self.lock = Lock()  # one pipeline per binary, which takes one call at a time

        (compiled,) = hef.get_input_vstream_infos()
        self.compiled = HailoNode(compiled.name, tuple(compiled.shape))
        self.input = HailoNode(contract.get("input", "image"), (1, *self.compiled.shape))
        self.cut: dict[str, str] = contract.get("cut", {})  # binary output -> tail input
        tail = model_path.parent / "tail.onnx"
        self.tail = ort.InferenceSession(tail.as_posix(), providers=["CPUExecutionProvider"]) if self.cut else None
        if self.tail is not None:
            self.ranks = {node.name: len(node.shape) for node in self.tail.get_inputs()}
            self.outputs: Sequence[SessionNode] = self.tail.get_outputs()
        else:
            self.outputs = [HailoNode(info.name, tuple(info.shape)) for info in hef.get_output_vstream_infos()]

        self.shapes = tuple(Shape(1, **dims) for dims in contract.get("dims", [{}]))
        # any count of frames is one call, so a run never splits into smaller ones than it has to
        self.batches = tuple(range(settings.hailo_batch_size, 0, -1))
        self.metadata: dict[str, str] = contract.get("metadata", {})
        log.info(f"Loaded Hailo model from {model_path} ({'multi' if multi_context else 'single'} context)")

    def for_shape(self, shape: Shape) -> HailoSession:
        return self

    def warm(self) -> None:
        self.run(None, {self.input.name: np.zeros(self.input.shape, dtype=np.uint8)})

    def get_inputs(self) -> Sequence[SessionNode]:
        return [self.input]

    def get_outputs(self) -> Sequence[SessionNode]:
        return self.outputs

    def get_metadata(self) -> dict[str, str]:
        return self.metadata

    @property
    def normalizes_input(self) -> bool:
        return True  # the preprocess is the binary's normalization layer, so it takes the raw pixels

    def run(
        self,
        output_names: list[str] | None,
        input_feed: ModelInput,
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        (frames,) = input_feed.values()
        if frames.dtype != np.uint8 or tuple(frames.shape[1:]) != self.compiled.shape:
            raise ValueError(
                f"binary takes uint8 {list(self.compiled.shape)} frames, fed {frames.dtype} {list(frames.shape)}"
            )
        with self.lock:
            compiled = self.pipeline.infer({self.compiled.name: np.ascontiguousarray(frames)})
        if self.tail is None:
            names = output_names or [node.name for node in self.outputs]
            return [compiled[name] for name in names]
        feeds = {tensor: _layout(compiled[output], self.ranks[tensor]) for output, tensor in self.cut.items()}
        outputs: list[NDArray[np.float32]] = self.tail.run(output_names, feeds)
        return outputs

    def __del__(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.__exit__(None, None, None)


__all__ = ["HailoSession", "HailoNode", "architecture", "is_available", "model_path"]
