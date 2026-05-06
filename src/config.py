from pathlib import Path

# Project root is two levels up from this file (src/config.py → src/ → root)
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH  = ROOT / "dataset" / "tx_survival.csv"
RESULTS_DIR = ROOT / "results"

# Survival horizon in years
TAU = 1.0

# Columns that are metadata/target — not used as model features
DROP_COLS = ["id", "obs_time", "event", "split", "ipcw", "y"]

# Ensure results directory exists on import
RESULTS_DIR.mkdir(exist_ok=True)
