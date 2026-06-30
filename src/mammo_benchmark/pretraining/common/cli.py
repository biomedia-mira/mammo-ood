from __future__ import annotations


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value from '{value}'.")


def parse_limit_batches(value):
    if value is None:
        return None
    raw = str(value).strip()
    parsed = float(raw)
    if parsed.is_integer() and parsed >= 1:
        return int(parsed)
    return parsed
