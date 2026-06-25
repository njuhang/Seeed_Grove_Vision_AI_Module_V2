from pathlib import Path


def test_op_optimization_log_contains_required_debug_fields() -> None:
    log_path = Path("docs/model_optimizer/op_optimization_log.md")

    text = log_path.read_text(encoding="utf-8")

    for heading in (
        "Op/Fallback Type",
        "Models",
        "Failure Symptom",
        "Root Cause Hypothesis",
        "Implementation Method",
        "Debug Process",
        "Vela Before/After",
        "Board Verification",
        "Remaining Risk",
    ):
        assert heading in text
