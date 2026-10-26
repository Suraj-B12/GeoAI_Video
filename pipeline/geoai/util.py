"""Small shared helpers: hashing, JSON, crash-tolerant CSV reading, logging."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("geoai")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Byte-identical to the app's canonical JSON (sorted keys, no spaces, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def to_json(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=_json_default, allow_nan=False)


def write_json_atomic(path: Path, obj: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(to_json(obj, indent))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read an app CSV. After a crash the last line may be incomplete (spec §1): drop it.

    Missing file -> empty DataFrame.
    """
    if not path.exists():
        return pd.DataFrame()
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        data = data[: cut + 1] if cut >= 0 else b""
    if not data.strip():
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(data), keep_default_na=True, **kwargs)


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based standard deviation (1.4826 x MAD), ignoring NaNs."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, mad
