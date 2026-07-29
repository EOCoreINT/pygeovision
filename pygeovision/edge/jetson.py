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
        """Convert a model to TensorRT for Jetson deployment."""
        if precision not in self.PRECISION_MODES:
            raise ValueError(f"precision must be one of {self.PRECISION_MODES}")

        onnx_path = Path(onnx_path); trt_path = Path(trt_path)
        trt_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Export to ONNX
        try:
            import torch
            model.eval()
            dummy = torch.randn(*input_shape)
            torch.onnx.export(model, dummy, str(onnx_path),
                               opset_version=17, input_names=["input"],
                               output_names=["output"],
                               dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            logger.info("ONNX exported: %s", onnx_path)
        except Exception as exc:
            return {"success": False, "stage": "onnx_export", "error": str(exc)}

        # Step 2: Convert to TensorRT
        try:
            import tensorrt as trt
            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)

            with open(str(onnx_path), "rb") as f:
                if not parser.parse(f.read()):
                    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    return {"success": False, "stage": "trt_parse", "errors": errors}

            config = builder.create_builder_config()
            config.max_workspace_size = workspace_gb * 1024**3
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)

            serialized = builder.build_serialized_network(network, config)
            with open(str(trt_path), "wb") as f:
                f.write(serialized)

            logger.info("TensorRT engine saved: %s", trt_path)
            return {"success": True, "trt_path": str(trt_path), "precision": precision}

        except ImportError:
            # TensorRT not installed — provide trtexec command instead
            trt_cmd = (
                f"trtexec --onnx={onnx_path} --saveEngine={trt_path} "
                f"--{'fp16' if precision == 'fp16' else 'int8' if precision == 'int8' else 'noFP16'} "
                f"--workspace={workspace_gb * 1024}"
            )
            logger.info("TensorRT not installed. Run on Jetson:\n  %s", trt_cmd)
            return {
                "success": False,
                "stage": "tensorrt_not_installed",
                "onnx_path": str(onnx_path),
                "jetson_command": trt_cmd,
                "note": "Run the trtexec command on the Jetson device to complete conversion",
            }

    def benchmark(self, trt_path: str, input_shape: tuple = (1, 4, 512, 512)) -> dict[str, Any]:
        """Benchmark TensorRT engine inference speed."""
        try:
            import time

            import numpy as np
            import tensorrt as trt
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(trt_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

            import pycuda.autoinit
            import pycuda.driver as cuda
            dummy = np.random.randn(*input_shape).astype(np.float32)
            d_input = cuda.mem_alloc(dummy.nbytes)
            n_out = input_shape[0] * input_shape[2] * input_shape[3] * 2   # assume 2 classes
            d_output = cuda.mem_alloc(n_out * 4)

            times = []
            for _ in range(100):
                cuda.memcpy_htod(d_input, dummy)
                t0 = time.perf_counter()
                context.execute_v2([int(d_input), int(d_output)])
                times.append((time.perf_counter() - t0) * 1000)

            import statistics
            return {
                "mean_ms":   round(statistics.mean(times), 2),
                "p95_ms":    round(sorted(times)[95], 2),
                "fps":       round(1000 / statistics.mean(times), 1),
                "precision": "detected from engine",
            }
        except ImportError:
            return {"error": "TensorRT + PyCUDA required. Install on Jetson: pip install tensorrt pycuda"}
        except Exception as exc:
            return {"error": str(exc)}