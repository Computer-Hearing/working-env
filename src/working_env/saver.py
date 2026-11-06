import os
import torch
import torch.nn as nn
import onnx
from onnx import TensorProto


class ModelSaver:
    """
    Сохраняет веса, экспортирует модель в ONNX (opset 18) и генерирует config.pbtxt для Triton.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def save_weights(self, save_path: str):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Веса сохранены: {save_path}")

    def export_onnx(
            self,
            onnx_path: str,
            dummy_input: torch.Tensor,
            input_names: list[str] = ["input"],
            output_names: list[str] = ["target_output"],
            dynamic_axes: dict = None,
            opset_version: int = 18,
            export_params: bool = True,
            do_constant_folding: bool = True
    ):
        os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
        self.model.eval()

        if dynamic_axes is None:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"}
            }

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=export_params,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        print(f"ONNX модель сохранена: {onnx_path}")
        return onnx_path

    def generate_triton_pbtxt(
            self,
            onnx_path: str,
            pbtxt_path: str,
            model_name: str = "my_model",
            max_batch_size: int = 32,
            backend: str = "onnxruntime",
    ):
        onnx_model = onnx.load(onnx_path)

        inputs = []
        for inp in onnx_model.graph.input:
            name = inp.name
            dims = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
            dtype = inp.type.tensor_type.elem_type
            data_type = "TYPE_FP32" if dtype == TensorProto.FLOAT else "TYPE_INT64"
            input_dims = dims[1:] if len(dims) > 1 else dims
            inputs.append((name, data_type, input_dims))

        outputs = []
        for out in onnx_model.graph.output:
            name = out.name
            dims = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim]
            dtype = out.type.tensor_type.elem_type
            data_type = "TYPE_FP32" if dtype == TensorProto.FLOAT else "TYPE_INT64"
            output_dims = dims[1:] if len(dims) > 1 else dims
            outputs.append((name, data_type, output_dims))

        lines = [
            f'name: "{model_name}"',
            f'backend: "{backend}"',
            f'max_batch_size: {max_batch_size}',
            '',
            'input [',
        ]
        for name, dtype, dims in inputs:
            dims_str = ", ".join(str(d) for d in dims)
            lines.append(f'  {{ name: "{name}" data_type: {dtype} dims: [ {dims_str} ] }}')
        lines.append(']')
        lines.append('')
        lines.append('output [')
        for name, dtype, dims in outputs:
            dims_str = ", ".join(str(d) for d in dims)
            lines.append(f'  {{ name: "{name}" data_type: {dtype} dims: [ {dims_str} ] }}')
        lines.append(']')

        os.makedirs(os.path.dirname(pbtxt_path) or ".", exist_ok=True)
        with open(pbtxt_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Triton config сохранен: {pbtxt_path}")

    def save_all(
            self,
            save_dir: str,
            model_name: str,
            dummy_input: torch.Tensor,
            input_names: list[str] = ["input"],
            output_names: list[str] = ["target_output"],
            dynamic_axes: dict = None,
            max_batch_size: int = 32,
            opset_version: int = 18,
            export_params: bool = True,
            do_constant_folding: bool = True
    ):
        os.makedirs(save_dir, exist_ok=True)

        # 1. Веса
        self.save_weights(os.path.join(save_dir, f"{model_name}.pth"))

        # 2. ONNX
        onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
        self.export_onnx(onnx_path, dummy_input, input_names, output_names, dynamic_axes,
                         opset_version, export_params, do_constant_folding)

        # 3. pbtxt
        pbtxt_path = os.path.join(save_dir, "config.pbtxt")
        self.generate_triton_pbtxt(onnx_path, pbtxt_path, model_name, max_batch_size)