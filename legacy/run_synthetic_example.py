"""Generate a deterministic smoke-test dataset and run AutoSlide."""

from pathlib import Path

import numpy as np
import pandas as pd

from autoslide_core import FEATURES, LABEL, run_benchmark, save_results


def main() -> None:
    rng = np.random.default_rng(42)
    rows = 120
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    score = 1.4 * frame["slope"] + 0.8 * frame["rainfall"] - 0.7 * frame["NDVI"]
    frame[LABEL] = (score + rng.normal(scale=0.8, size=rows) > 0).astype(int)
    input_path = Path("examples") / "synthetic_dataset_generated.csv"
    output_path = Path("examples") / "synthetic_results"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(input_path, index=False)
    metrics, predictions = run_benchmark(frame, "random_forest", rounds=3, seed=42)
    summary = save_results(metrics, predictions, output_path)
    print(f"Input written to {input_path}")
    print(f"Results written to {output_path}")
    print(summary)


if __name__ == "__main__":
    main()

