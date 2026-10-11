"""NVIDIA Jetson deployment utilities (F3)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JetsonDeployer:
    """Deploy PyGeoVision models to NVIDIA Jetson (Nano, Orin, Xavier).

    Converts PyTorch models to TensorRT for maximum inference speed
    on Jetson's integrated GPU.

    Example::

        deployer = JetsonDeployer()
        trt_path = deployer.convert(model, "model.onnx", "model.trt",
                                     precision="fp16", input_shape=(1,4,512,512))
        latency = deployer.benchmark("model.trt")
    """

    PRECISION_MODES = ["fp32", "fp16", "int8"]

    def __init__(self, device: str = "jetson") -> None:
        self.device = device

    def convert(
        self,
        model: Any,
        onnx_path: str | Path,
        trt_path: str | Path,
        precision: str = "fp16",
        input_shape: tuple = (1, 4, 512, 512),
        workspace_gb: int = 4,
    ) -> dict[str, Any]:
        """Export ``model`` to a fixed-shape ONNX file and build a TensorRT engine from it.

        The ONNX file is always written. The engine is built here only when the
        ``tensorrt`` Python package is importable (on a Jetson, or a PC with TensorRT);
        otherwise the result has ``success=False``, ``stage="tensorrt_not_installed"``,
        the ``onnx_path`` and a ready-to-run ``trtexec`` command for the device.

        The input shape is fixed to ``input_shape`` (TensorRT needs an optimisation
        profile for dynamic dimensions, which this class does not create). ``int8`` is
        requested without a calibration set, so its accuracy is not guaranteed: validate
        the engine's output against the PyTorch model before using it. The TensorRT
        build path has not been run in PyGeoVision's own test environment (no TensorRT).
        """
        if precision not in self.PRECISION_MODES:
            raise ValueError(f"precision must be one of {self.PRECISION_MODES}")

        onnx_path = Path(onnx_path); trt_path = Path(trt_path)
        trt_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Export to ONNX
        try:
            import torch
            model.eval()
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            dummy = torch.randn(*input_shape, device=device)
            from pygeovision._onnx import export_onnx
            export_onnx(model, dummy, onnx_path,
                        opset_version=17, input_names=["input"],
                        output_names=["output"])
            logger.info("ONNX exported: %s", onnx_path)
        except Exception as exc:
            return {"success": False, "stage": "onnx_export", "error": str(exc)}

        # Step 2: Convert to TensorRT
        try:
            import tensorrt as trt
            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
            network = builder.create_network(1 << int(flag) if flag is not None else 0)
            parser = trt.OnnxParser(network, trt_logger)

            with open(str(onnx_path), "rb") as f:
                if not parser.parse(f.read()):
                    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    return {"success": False, "stage": "trt_parse", "errors": errors}

            config = builder.create_builder_config()
            if hasattr(config, "set_memory_pool_limit"):                 # TensorRT >= 8.4
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * 1024**3)
            else:                                                        # older TensorRT
                config.max_workspace_size = workspace_gb * 1024**3
            used = "fp32"
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                used = "fp16"
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                used = "int8"

            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                return {"success": False, "stage": "trt_build",
                        "error": "TensorRT could not build an engine (see its log above)"}
            with open(str(trt_path), "wb") as f:
                f.write(serialized)

            logger.info("TensorRT engine saved: %s", trt_path)
            result = {"success": True, "trt_path": str(trt_path), "precision": used,
                      "requested_precision": precision, "onnx_path": str(onnx_path)}
            if used != precision:
                result["note"] = f"this device has no fast {precision}; the engine was built in {used}"
            elif used == "int8":
                result["note"] = "INT8 without calibration data: validate accuracy against the PyTorch model"
            return result

        except ImportError:
            # TensorRT not installed — provide trtexec command instead
            flag_txt = {"fp16": " --fp16", "int8": " --int8", "fp32": ""}[precision]
            trt_cmd = (
                f"trtexec --onnx={onnx_path} --saveEngine={trt_path}{flag_txt} "
                f"--memPoolSize=workspace:{workspace_gb * 1024}M"
            )
            logger.info("TensorRT not installed. Run on Jetson:\n  %s", trt_cmd)
            return {
                "success": False,
                "stage": "tensorrt_not_installed",
                "onnx_path": str(onnx_path),
                "jetson_command": trt_cmd,
                "note": "Run the trtexec command on the Jetson device to complete conversion "
                        "(TensorRT older than 8.4 uses --workspace=<MiB> instead of --memPoolSize)",
            }

    @staticmethod
    def _output_numel(engine: Any, trt: Any, batch: int) -> int:
        """Number of elements in the engine's first output, from the engine itself."""
        if hasattr(engine, "num_io_tensors"):                              # TensorRT >= 8.5
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            shape = tuple(engine.get_tensor_shape(outs[0]))
        else:
            shape = tuple(engine.get_binding_shape(1))
        shape = tuple(batch if d == -1 and i == 0 else d for i, d in enumerate(shape))
        if any(d <= 0 for d in shape):
            raise ValueError(f"engine output has a dynamic shape {shape}; build a fixed-shape engine")
        n = 1
        for d in shape:
            n *= d
        return n

    def benchmark(self, trt_path: str, input_shape: tuple = (1, 4, 512, 512), n_runs: int = 100) -> dict[str, Any]:
        """Time a TensorRT engine with random input (needs ``tensorrt`` and ``pycuda`` on a CUDA device).

        The output buffer is sized from the engine, whatever its class count. Returns
        ``mean_ms``, ``median_ms``, ``p95_ms``, ``fps`` (``mean_ms`` and ``fps`` are per call,
        i.e. per batch of ``input_shape[0]``) or ``{"error": ...}``. Each run copies the input to the GPU
        outside the timer and times only ``execute_v2``. Not run in PyGeoVision's own tests (no TensorRT).
        """
        try:
            import statistics
            import time

            import numpy as np
            import tensorrt as trt
            if n_runs < 1:
                raise ValueError("n_runs must be at least 1")
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(trt_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                return {"error": f"could not load {trt_path} as a TensorRT engine "
                                 "(built with another TensorRT version or on another GPU?)"}
            context = engine.create_execution_context()

            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            dummy = np.random.randn(*input_shape).astype(np.float32)
            d_input = cuda.mem_alloc(dummy.nbytes)
            d_output = cuda.mem_alloc(self._output_numel(engine, trt, input_shape[0]) * 4)

            times = []
            for i in range(n_runs + 5):                                     # 5 warm-up runs
                cuda.memcpy_htod(d_input, dummy)
                t0 = time.perf_counter()
                context.execute_v2([int(d_input), int(d_output)])
                if i >= 5:
                    times.append((time.perf_counter() - t0) * 1000)

            return {
                "mean_ms":   round(statistics.mean(times), 2),
                "median_ms": round(statistics.median(times), 2),
                "p95_ms":    round(sorted(times)[min(n_runs - 1, int(n_runs * 0.95))], 2),
                "fps":       round(1000 / statistics.mean(times), 1),
                "n_runs":    n_runs,
            }
        except ImportError:
            return {"error": "TensorRT + PyCUDA required. Install on Jetson: pip install tensorrt pycuda"}
        except Exception as exc:
            return {"error": str(exc)}
