import networkx as nx
import numpy as np
import pandas as pd

def build_transaction_graph(df):
    graph = nx.DiGraph()
    for row in df.itertuples(index=False):
        graph.add_edge(str(row.sender_id), str(row.receiver_id),
                       amount=float(row.amount), channel=str(row.channel),
                       is_fraud=int(row.is_fraud))
    return graph

def account_features(df):
    accounts = sorted(set(df.sender_id.astype(str)) | set(df.receiver_id.astype(str)))
    out = df.groupby("sender_id").agg(sent_count=("transaction_id", "count"),
        sent_amount=("amount", "sum"), mean_sent_amount=("amount", "mean"),
        fraud_sent_rate=("is_fraud", "mean")).rename_axis("account_id")
    inc = df.groupby("receiver_id").agg(received_count=("transaction_id", "count"),
        received_amount=("amount", "sum")).rename_axis("account_id")
    feat = pd.DataFrame(index=accounts).join(out).join(inc).fillna(0)
    feat["total_activity"] = feat.sent_count + feat.received_count
    feat["total_volume"] = feat.sent_amount + feat.received_amount
    feat["account_id"] = feat.index
    return feat.reset_index(drop=True)

def transaction_feature_matrix(df):
    f = account_features(df).set_index("account_id")
    graph = build_transaction_graph(df)
    rows = []
    for row in df.itertuples(index=False):
        s = f.loc[str(row.sender_id)]
        r = f.loc[str(row.receiver_id)]
        rows.append([
            np.log1p(float(row.amount)), int(row.hour),
            int(row.hour <= 4 or row.hour >= 23),
            *[int(row.channel == c) for c in ["UPI", "CARD", "BANK_TRANSFER", "WALLET"]],
            np.log1p(float(s.total_activity)), np.log1p(float(r.total_activity)),
            np.log1p(float(s.total_volume)), np.log1p(float(r.total_volume)),
            float(s.fraud_sent_rate),
            graph.in_degree(str(row.sender_id)), graph.out_degree(str(row.sender_id)),
            graph.in_degree(str(row.receiver_id)), graph.out_degree(str(row.receiver_id))
        ])
    return np.asarray(rows, dtype=np.float32)

FEATURE_NAMES = ["log_amount", "hour", "late_night", "channel_upi", "channel_card",
    "channel_bank_transfer", "channel_wallet", "sender_activity", "receiver_activity",
    "sender_volume", "receiver_volume", "sender_fraud_history",
    "sender_in_degree", "sender_out_degree", "receiver_in_degree", "receiver_out_degree"]
