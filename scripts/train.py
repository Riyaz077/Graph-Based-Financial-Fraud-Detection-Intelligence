import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import generate_transactions, load_transactions
from src.pipeline import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/transactions.csv")
args = parser.parse_args()
path = Path(args.data)
df = load_transactions(path) if path.exists() else generate_transactions()
metrics, _ = train_and_evaluate(df)
print(metrics)
