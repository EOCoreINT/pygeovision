# Deploying models: edge and cloud

A trained `.pth` file is not yet something a field laptop, a drone computer or a web service can run. This section covers
the three ways PyGeoVision takes a model out of the training environment, and how far each has been verified.

| You want to run the model… | Use | Page | Verified here? |
|---|---|---|---|
| On any machine without PyTorch (CPU server, laptop, Raspberry-class board) | Export to **ONNX**, run with **ONNX Runtime** | [Edge deployment](edge.md#onnx-export-and-runtime) | **Yes**: export, numeric parity with PyTorch, tiled GeoTIFF inference and timings were run |
| On an NVIDIA **Jetson** | ONNX → **TensorRT** engine | [Edge deployment](edge.md#nvidia-jetson-and-tensorrt) | ONNX export and the `trtexec` command: yes. TensorRT build and timing: **no** (no TensorRT/GPU available) |
| As a hosted **endpoint** on AWS, Azure or Google Cloud | `AWSDeployer`, `AzureDeployer`, `GCPDeployer` | [Cloud deployment](cloud.md) | Packaging, the SageMaker request handler and request building: yes. **No live cloud call has been made** |

```{admonition} Read this before you deploy anything
:class: important
1. **An exported model is not a better model.** Export only changes the file format. Always compare the exported model's
   output with the original on a few real chips before trusting it ([how](edge.md#check-the-export)).
2. **The pre- and post-processing are not in the file.** Scaling, band order, tiling and class names must be reproduced
   exactly as in training ([Inference](../inference/index.md#matching-the-input-to-training)).
3. **Cloud endpoints cost money while they exist.** Delete them when you are done ([`delete_endpoint`](cloud.md#cleaning-up)).
4. **Latency numbers are for the machine that measured them.** Benchmark on the target hardware.
```

```{toctree}
:maxdepth: 1
:hidden:

edge
cloud
```

Related: [Exporting and shrinking a model (training)](../training/advanced.md#exporting-and-shrinking-a-model) ·
[Benchmarking](../benchmark/index.md) · [Installation extras](../getting-started/installation.md)
