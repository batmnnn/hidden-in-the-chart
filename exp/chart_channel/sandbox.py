"""Run decoder measurements against one image. No network, no imports beyond PIL and numpy."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

_BLOCKED = re.compile(r"__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\bcompile\b|\bglobals\b|\blocals\b")

_RUNNER = textwrap.dedent(
    """
    import sys
    from PIL import Image
    import numpy as np

    im = Image.open(sys.argv[1]).convert("RGB")
    ims = [Image.open(p).convert("RGB") for p in sys.argv[2:]]
    code = sys.stdin.read()
    code = "\\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith(("import ", "from "))
    )
    safe = {
        "range": range, "len": len, "int": int, "float": float, "str": str,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "enumerate": enumerate, "zip": zip, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "sorted": sorted, "map": map,
        "any": any, "all": all, "print": print, "bool": bool, "pow": pow,
    }
    try:
        exec(code, {"__builtins__": safe, "im": im, "ims": ims, "np": np, "Image": Image}, {})
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}")
    """
)


def measure(image: Path, code: str, timeout: float = 12.0, extra: list[Path] | None = None) -> str:
    # The runner already provides im, ims, np and Image and drops import lines, so drop them
    # before the safety check too (a bare "import numpy as np" used to block the whole snippet).
    code = "\n".join(line for line in code.splitlines() if not line.strip().startswith(("import ", "from ")))
    # Decoders often reload the picture themselves; the runner already loaded it as im.
    code = re.sub(r"Image\.open\([^()]*\)(\.convert\([^()]*\))?", "im.copy()", code)
    if _BLOCKED.search(code):
        return "ERROR blocked syntax"
    env = {k: v for k, v in os.environ.items() if not any(s in k for s in ("TOKEN", "KEY", "SECRET"))}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER, str(image), *[str(p) for p in (extra or [])]],
            input=code,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "ERROR timeout"
    out = (proc.stdout or "") + (proc.stderr or "")
    return out[-4000:]
