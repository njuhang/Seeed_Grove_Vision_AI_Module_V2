from pathlib import Path


def artifact_dir(model_name: str, output_root: str | Path = "artifacts") -> Path:
    return Path(output_root) / model_name


def float_model_path(model_name: str, output_root: str | Path = "artifacts") -> Path:
    """Path of the unquantized float .tflite produced by the PT->LiteRT export.

    Sits next to ``quantized_model_path`` so callers can locate the float model
    that feeds ai-edge-quantizer without re-deriving the path.
    """
    return artifact_dir(model_name, output_root) / f"{model_name}_float.tflite"


def quantized_model_path(model_name: str, output_root: str | Path = "artifacts") -> Path:
    return artifact_dir(model_name, output_root) / f"{model_name}_int8.tflite"


def vela_model_path(model_name: str, output_root: str | Path = "artifacts") -> Path:
    return artifact_dir(model_name, output_root) / f"{model_name}_vela.tflite"
