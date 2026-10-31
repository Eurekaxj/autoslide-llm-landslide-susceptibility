"""Independent same-definition terrain computation; old GDAL results are historical."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("27_plot_source.py")))["terrain"]()
