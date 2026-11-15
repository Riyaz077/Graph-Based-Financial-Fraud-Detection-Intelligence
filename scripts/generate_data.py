from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import generate_transactions

out = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
out.parent.mkdir(exist_ok=True)
generate_transactions().to_csv(out, index=False)
print(f"Generated {out}")
