from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH  = ROOT / "dataset" / "tx_survival.csv"
RESULTS_DIR = ROOT / "results"

TAU = 1.0

DROP_COLS = ["id", "obs_time", "event", "split", "ipcw", "y"]

RESULTS_DIR.mkdir(exist_ok=True)
