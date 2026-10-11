"""ONNX Runtime inference — cross-platform, CPU/GPU/Jetson (F2, F3)."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ONNXRuntimeInference:
    """High-performance ONNX Runtime inference engine for geospatial models (F2).

    Supports CPU, CUDA, TensorRT (on Jetson), and CoreML (on Apple Silicon).
    Drop-in replacement for PyTorch inference — 2-4x faster on CPU,
    comparable speed on GPU with reduced memory.

    Example::

        engine = ONNXRuntimeInference("model.onnx", device="cuda")
        pred = engine.infer(image_array)   # numpy array in, numpy array out
        result = engine.infer_geotiff("./data/scene.tif", "./output/pred.tif")
        bench = engine.benchmark(n_runs=100)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cpu",       # cpu | cuda | tensorrt | coreml
        optimization_level: str = "all",
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.device = device
        self.optimization_level = optimization_level
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self._session = None

    def _load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime required:\n"
                "  CPU:  pip install onnxruntime\n"
                "  GPU:  pip install onnxruntime-gpu"
            )

        opt_levels = {
            "none":     ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }

        opts = ort.SessionOptions()
        opts.graph_optimization_level = opt_levels.get(self.optimization_level,
                                                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        opts.intra_op_num_threads = self.intra_op_threads
        opts.inter_op_num_threads = self.inter_op_threads

        # Select execution providers by device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "tensorrt":
            providers = [
                ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif self.device == "coreml":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        self._output_name = self._session.get_outputs()[0].name
        wanted = {"cuda": "CUDAExecutionProvider", "tensorrt": "TensorrtExecutionProvider",
                  "coreml": "CoreMLExecutionProvider"}.get(self.device)
        if wanted and wanted not in self._session.get_providers():
            logger.warning(
                "device=%r was requested but %s is not available in this onnxruntime build "
                "(active: %s); running on the CPU. Install onnxruntime-gpu (CUDA/TensorRT) on a "
                "machine with the matching drivers.", self.device, wanted, self._session.get_providers())
        logger.info("ONNXRuntime loaded: %s | providers=%s",
                    Path(self.onnx_path).name,
                    [p if isinstance(p, str) else p[0] for p in self._session.get_providers()])

    def infer(self, input_array: Any) -> Any:
        """Run inference on a numpy array.

        Args:
            input_array: numpy array (C, H, W) or (B, C, H, W)

        Returns:
            Output numpy array from the model
        """
        import numpy as np
        self._load()
        if hasattr(input_array, "numpy"):
            arr = input_array.numpy()
        else:
            arr = np.asarray(input_array)
        if arr.ndim == 3:
            arr = arr[np.newaxis]
        arr = arr.astype(np.float32)
        outputs = self._session.run([self._output_name], {self._input_name: arr})
        return outputs[0]

    def infer_geotiff(
        self,
        image_path: str | Path,
        output_path: str | Path = "./output/onnx_pred.tif",
        chip_size: int = 512,
        overlap: int = 64,
        num_classes: int | None = None,
        blend_mode: str = "gaussian",
        normalise: bool = True,
    ) -> dict[str, Any]:
        """Run tiled ONNX inference over a full GeoTIFF (B1, B4).

        Wraps TiledInference with an ONNX-backed forward pass — handles
        arbitrarily large GeoTIFFs with Gaussian tile blending.

        ``num_classes`` defaults to the model's output channel count (2 when the
        model does not declare one). A model exported with fixed spatial
        dimensions can only take chips of exactly that size, so ``chip_size`` is
        replaced by it (with a warning) and ``overlap`` is reduced to fit; export
        with ``dynamic_spatial=True`` to allow any chip size.
        """
        import numpy as np
        self._load()
        hw = self._input_shape[2:4]
        if len(hw) == 2 and all(isinstance(d, int) and d > 0 for d in hw):
            if hw[0] != hw[1]:
                return {"success": False,
                        "error": f"the model accepts only {hw[0]}x{hw[1]} (non-square) inputs; tiled "
                                 f"inference needs square chips. Re-export with dynamic_spatial=True."}
            if hw[0] != chip_size:
                new_overlap = min(overlap, hw[0] // 4)
                logger.warning("the model was exported for %dx%d inputs: using chip_size=%d, overlap=%d "
                               "(instead of %d, %d). Export with dynamic_spatial=True for other sizes.",
                               hw[0], hw[1], hw[0], new_overlap, chip_size, overlap)
                chip_size, overlap = hw[0], new_overlap

        # One test chip tells us how many classes the model really outputs.
        in_ch = self._input_shape[1] if len(self._input_shape) > 1 else None
        out_classes = None
        if isinstance(in_ch, int) and in_ch > 0:
            try:
                probe = self.infer(np.zeros((1, in_ch, chip_size, chip_size), dtype=np.float32))
            except Exception as exc:  # noqa: BLE001
                return {"success": False,
                        "error": f"the model failed on a {chip_size}x{chip_size} test chip with {in_ch} "
                                 f"bands: {exc}"}
            out_classes = int(probe.shape[1]) if probe.ndim == 4 else None
        if num_classes is None:
            num_classes = out_classes or 2
        elif out_classes is not None and out_classes != num_classes:
            return {"success": False,
                    "error": f"num_classes={num_classes} but the model outputs {out_classes} channels; "
                             f"pass num_classes={out_classes} (or leave it out)"}
        try:
            import rasterio  # noqa: F401
            import torch
            import torch.nn as nn
        except ImportError as exc:
            return {"success": False, "error": f"rasterio + torch required: {exc}"}

        # Wrap ONNX session as a PyTorch-compatible callable
        session = self
        class ONNXWrapper(nn.Module):
            def forward(self, x):
                import numpy as np
                arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
                result = session.infer(arr)
                return torch.tensor(result)

        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(
            model=ONNXWrapper(),
            chip_size=chip_size,
            overlap=overlap,
            blend_mode=blend_mode,
            num_classes=num_classes,
        )
        return inf.infer(image_path, output_path, normalise=normalise)

    def benchmark(self, input_shape: tuple | None = None, n_runs: int = 100) -> dict[str, Any]:
        """Benchmark inference speed.

        Args:
            input_shape: Override input shape (default: from model)
            n_runs: Number of inference runs

        Returns:
            Dict with ``mean_ms``, ``median_ms``, ``p95_ms``, ``fps`` (from the mean),
            ``n_runs``, ``input_shape``, ``device`` and the active ``providers``.
            Five untimed warm-up runs come first; the median is the steadier figure.
        """
        import numpy as np
        if n_runs < 1:
            raise ValueError("n_runs must be at least 1")
        self._load()

        if input_shape is None:
            # Build shape from model input — replace dynamic dims with 1 or 512
            raw_shape = self._input_shape
            shape = tuple(d if isinstance(d, int) and d > 0 else (1 if i == 0 else 512)
                          for i, d in enumerate(raw_shape))
        else:
            shape = input_shape

        dummy = np.random.randn(*shape).astype(np.float32)
        times = []

        # Warm-up
        for _ in range(5):
            self._session.run([self._output_name], {self._input_name: dummy})

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self._session.run([self._output_name], {self._input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        import statistics
        return {
            "mean_ms":   round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "p95_ms":    round(sorted(times)[min(n_runs - 1, int(n_runs * 0.95))], 2),
            "fps":       round(1000 / statistics.mean(times), 1),
            "n_runs":    n_runs,
            "input_shape": shape,
            "device":    self.device,
            "providers": self._session.get_providers(),
        }

    def model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded ONNX model."""
        self._load()
        inputs  = [{"name": i.name, "shape": i.shape, "dtype": i.type}
                   for i in self._session.get_inputs()]
        outputs = [{"name": o.name, "shape": o.shape, "dtype": o.type}
                   for o in self._session.get_outputs()]
        return {
            "onnx_path": self.onnx_path,
            "inputs":    inputs,
            "outputs":   outputs,
            "providers": self._session.get_providers(),
        }

    @staticmethod
    def from_pytorch(
        model: Any,
        output_path: str | Path,
        input_shape: tuple = (1, 4, 512, 512),
        opset: int = 17,
        simplify: bool = True,
        dynamic_batch: bool = True,
        dynamic_spatial: bool = False,
        # alias accepted by tests
        opset_version: int = 0,
    ) -> ONNXRuntimeInference:
        """Export a PyTorch model to ONNX and return an ONNXRuntimeInference wrapper.

        Args:
            model: PyTorch nn.Module in eval mode
            output_path: Output .onnx file path
            input_shape: Example input shape (B, C, H, W)
            opset: ONNX opset version (17 recommended)
            simplify: Run onnxsim to optimise the graph
            dynamic_batch: Allow dynamic batch dimension
            dynamic_spatial: Also allow any height and width at run time. Off by default
                because some architectures (fixed reshapes, resizes to a constant size)
                export wrongly with it; check parity on a second size before relying on it.

        Returns:
            Ready-to-use ONNXRuntimeInference instance
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for ONNX export")

        # opset_version kwarg takes precedence (passed by some callers)
        effective_opset = opset_version if opset_version else opset

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        dummy = torch.randn(*input_shape, device=device)
        axes: dict[int, str] = {}
        if dynamic_batch:
            axes[0] = "batch"
        if dynamic_spatial:
            axes.update({2: "height", 3: "width"})
        dynamic_axes = {"input": dict(axes), "output": dict(axes)} if axes else None

        # Use the legacy torch.onnx.export API (compatible with torch<2.6 and >=2.6)
        # torch>=2.6 moved to a new exporter that requires onnxscript; the legacy
        # path is still available via torch.onnx.export with dynamo=False.
        _export_kwargs: dict[str, Any] = dict(
            opset_version=effective_opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        try:
            # torch>=2.6: pass dynamo=False to force legacy path (no onnxscript dep)
            torch.onnx.export(
                model, dummy, str(output_path),
                dynamo=False,
                **_export_kwargs,
            )
        except TypeError:
            # torch<2.6 doesn't accept dynamo=
            torch.onnx.export(
                model, dummy, str(output_path),
                **_export_kwargs,
            )

        logger.info("ONNX exported: %s (opset=%d)", output_path, effective_opset)

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnxsim
                model_onnx = onnx.load(str(output_path))
                simplified, ok = onnxsim(model_onnx)
                if ok:
                    onnx.save(simplified, str(output_path))
                    logger.info("ONNX graph simplified")
            except ImportError:
                logger.debug("pip install onnxsim for graph simplification")

        return ONNXRuntimeInference(output_path)
