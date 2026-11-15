from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from src.graph import transaction_feature_matrix
from src.data import generate_transactions
from src.pipeline import train_and_evaluate

app = FastAPI(title="GraphShield API", version="1.0.0",
              description="Educational graph-based transaction fraud intelligence prototype.")
class Transaction(BaseModel):
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    channel: str
    is_fraud: int = Field(ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "ok", "project": "GraphShield"}

@app.post("/demo/train")
def demo_train():
    try:
        metrics, scored = train_and_evaluate(generate_transactions(1800, seed=7),
                                              artifact_dir="/tmp/graphshield_artifacts",
                                              report_dir="/tmp/graphshield_reports")
        return {"metrics": metrics, "scored_sample": scored.head(10).to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/demo/score")
def demo_score(tx: Transaction):
    # Explicitly illustrative score proxy; model training endpoint provides evaluation.
    amount_component = min(tx.amount / 5000, 0.45)
    late_component = 0.25 if tx.hour <= 4 or tx.hour >= 23 else 0.0
    channel_component = 0.10 if tx.channel.upper() == "WALLET" else 0.0
    score = min(0.05 + amount_component + late_component + channel_component, 0.95)
    band = "High" if score >= 0.7 else "Medium" if score >= 0.35 else "Low"
    return {"transaction_id": tx.transaction_id, "illustrative_risk_score": round(score, 4),
            "risk_band": band, "note": "Rule-based demo score, not a trained-model prediction."}
