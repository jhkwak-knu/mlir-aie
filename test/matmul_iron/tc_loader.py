# tc_loader.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union
import json


@dataclass
class TcJson:
    """
    Container for holding the raw tc.json.

    Attributes:
        data: The raw JSON content as loaded (kept unchanged).
    """
    data: Dict[str, Any]


def load_tc_json(path: Union[str, Path]) -> TcJson:
    """
    Load tc.json from disk and return a TcJson object.

    Args:
        path: Path to the JSON file.

    Returns:
        TcJson with raw data preserved.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] tc.json not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tc = TcJson(data=data)

    return tc
