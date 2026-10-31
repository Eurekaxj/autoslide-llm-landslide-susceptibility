"""Compatibility entry; supersedes the archived historical implementation."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("26_pslsa_common.py")), run_name="__main__")
