from src.data import generate_transactions
from src.graph import build_transaction_graph, transaction_feature_matrix

def test_synthetic_data_schema_and_labels():
    df = generate_transactions(120, 3)
    assert {"transaction_id","sender_id","receiver_id","amount","hour","channel","is_fraud"} <= set(df.columns)
    assert set(df.is_fraud.unique()) <= {0,1}

def test_graph_and_feature_matrix():
    df = generate_transactions(100, 4)
    graph = build_transaction_graph(df)
    X = transaction_feature_matrix(df)
    assert graph.number_of_nodes() > 0
    assert X.shape == (len(df), 16)
