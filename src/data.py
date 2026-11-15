from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ["transaction_id", "sender_id", "receiver_id", "amount", "hour", "channel", "is_fraud"]

def generate_transactions(n=5000, seed=42):
    """Create reproducible synthetic transactions for demos and tests."""
    rng = np.random.default_rng(seed)
    accounts = [f"ACC{i:04d}" for i in range(350)]
    sender = rng.choice(accounts, n)
    receiver = rng.choice(accounts, n)
    same = sender == receiver
    while same.any():
        receiver[same] = rng.choice(accounts, same.sum())
        same = sender == receiver
    amount = np.round(rng.lognormal(mean=3.2, sigma=1.0, size=n), 2)
    hour = rng.integers(0, 24, n)
    channel = rng.choice(["UPI", "CARD", "BANK_TRANSFER", "WALLET"], n,
                         p=[0.48, 0.22, 0.20, 0.10])
    # Synthetic label mechanism, intentionally imperfect and not a real fraud model.
    logit = (-4.2 + 0.0018 * amount + 0.75 * ((hour <= 4) | (hour >= 23))
             + 0.65 * (channel == "WALLET") + 0.35 * (channel == "UPI"))
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)
    return pd.DataFrame({
        "transaction_id": [f"TXN{i:07d}" for i in range(n)],
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": amount,
        "hour": hour,
        "channel": channel,
        "is_fraud": fraud
    })

def load_transactions(path):
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[REQUIRED].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df = df[(df["amount"] >= 0) & df["hour"].between(0, 23) & df["is_fraud"].isin([0, 1])]
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df.reset_index(drop=True)
