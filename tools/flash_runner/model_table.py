import struct

from tools.model_converter.model_registry import ModelSpec

MAGIC = b"MODL"
VERSION = 1
NAME_BYTES = 32
TASK_BYTES = 32
ENTRY_STRUCT = struct.Struct("<32s32sIIIIII")


def build_model_table(models: list[tuple[ModelSpec, int]]) -> bytes:
    header = bytearray()
    header.extend(MAGIC)
    header.extend(struct.pack("<I", VERSION))
    header.extend(struct.pack("<I", len(models)))
    header.extend(struct.pack("<I", 0))

    entries = bytearray()
    for model, model_size in models:
        name = model.name.encode("ascii")
        input_channels, input_height, input_width = _table_input_dims(model.input_shape)
        entries.extend(
            ENTRY_STRUCT.pack(
                name.ljust(NAME_BYTES, b"\x00"),
                model.task.encode("ascii").ljust(TASK_BYTES, b"\x00"),
                model.flash_address,
                model_size,
                input_channels,
                input_height,
                input_width,
                model.tier,
            )
        )
    return bytes(header + entries)


def _table_input_dims(input_shape: list[int]) -> tuple[int, int, int]:
    if len(input_shape) == 4:
        return int(input_shape[1]), int(input_shape[2]), int(input_shape[3])
    if len(input_shape) == 3:
        return 1, int(input_shape[1]), int(input_shape[2])
    raise ValueError(f"model table expects 3D or 4D input_shape, got {input_shape!r}")
