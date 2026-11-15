from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.graph import transaction_feature_matrix, FEATURE_NAMES
from src.model import train_graph_model, predict_proba

def train_and_evaluate(df, artifact_dir="artifacts", report_dir="reports"):
    X = transaction_feature_matrix(df)
    y = df.is_fraud.to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both fraud and non-fraud labels.")
    Xtr, Xte, ytr, yte, idxtr, idxte = train_test_split(
        X, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)
    graph_model = train_graph_model(Xtr_s, ytr)
    gp = predict_proba(graph_model, Xte_s)
    rf = RandomForestClassifier(n_estimators=180, max_depth=10, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rp = rf.predict_proba(Xte)[:, 1]
    def metrics(prob):
        pred = (prob >= 0.5).astype(int)
        return {"accuracy": float(accuracy_score(yte, pred)),
                "precision": float(precision_score(yte, pred, zero_division=0)),
                "recall": float(recall_score(yte, pred, zero_division=0)),
                "f1": float(f1_score(yte, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(yte, prob))}
    artifact = Path(artifact_dir); artifact.mkdir(parents=True, exist_ok=True)
    report = Path(report_dir); report.mkdir(parents=True, exist_ok=True)
    torch.save(graph_model.state_dict(), artifact / "graphshield_model.pt")
    joblib.dump(scaler, artifact / "scaler.joblib")
    joblib.dump(rf, artifact / "random_forest.joblib")
    with open(artifact / "model_config.json", "w") as f:
        json.dump({"input_dim": int(X.shape[1]), "feature_names": FEATURE_NAMES}, f, indent=2)
    result = {"GraphSAGE-style": metrics(gp), "Random Forest": metrics(rp)}
    with open(report / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    scored = df.iloc[idxte].copy()
    scored["graphshield_probability"] = gp
    scored["risk_band"] = pd.cut(gp, [-0.001, 0.35, 0.70, 1.0],
                                 labels=["Low", "Medium", "High"]).astype(str)
    scored.to_csv(report / "scored_transactions.csv", index=False)
    return result, scored
