# Configuration for the pipeline

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    random_state: int = 42
    n_jobs: int = -1
    output_dir: Path = Path("outputs")
    figures_dir: Path = Path("outputs/figures")
    results_path: Path = Path("outputs/results.csv")
    n_splits_outer: int = 5
    n_splits_inner: int = 3
    n_iter_search: int = 20
    test_size: float = 0.2

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
