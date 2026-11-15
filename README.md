# GraphShield
## Graph Neural Network-Based Financial Transaction Network and Fraud Intelligence

GraphShield is an educational fraud-intelligence prototype that models financial transactions as a graph. It builds a transaction network, engineers node and edge features, trains a graph-neural-network-style classifier, and presents transaction risk scores with network-level insights.

### Key capabilities
- Generate or load a transaction dataset with sender, receiver, amount, timestamp, channel and fraud label.
- Build a directed transaction graph using NetworkX.
- Derive account-level network features: degree, weighted transaction volume, fraud-neighbour ratio and account activity.
- Train a GraphSAGE-style message-passing classifier using PyTorch.
- Compare graph model results with a tabular Random Forest baseline.
- Inspect high-risk transactions, suspicious account connections and model metrics.
- Explore results in a Streamlit dashboard and query predictions through FastAPI.

### Architecture
```text
Synthetic / CSV Transactions
          |
          v
Validation + Preprocessing
          |
          v
Directed Account Transaction Graph
          |
          +--> Network Features
          |
          v
GraphSAGE-style Message Passing
          |
          v
Fraud Probability + Risk Band
          |
          v
Dashboard / API / Network Intelligence
```

### Quick start
Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py
streamlit run app/streamlit_app.py
```

Start the API separately:
```bash
uvicorn api.main:app --reload
```

The training script writes model artifacts to `artifacts/` and evaluation output to `reports/`.

### Use your own data
Provide a CSV containing these columns:
- `transaction_id`
- `sender_id`
- `receiver_id`
- `amount`
- `hour`
- `channel`
- `is_fraud` (0 or 1; training/evaluation only)

Run:
```bash
python scripts/train.py --data path/to/transactions.csv
```

### Project structure
```text
GraphShield/
├── api/                  # FastAPI endpoints
├── app/                  # Streamlit dashboard
├── data/                 # Sample synthetic transactions
├── docs/                 # Project documentation
├── notebooks/            # Exploratory analysis notebook
├── scripts/              # Data generation and training
├── src/                  # Graph construction, features and models
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

### Important limitations
This repository is a research/portfolio prototype. The included dataset is entirely synthetic and does not represent real customers, accounts, banks or payment networks. The graph model is a compact educational GraphSAGE-style implementation, not a production-grade, large-scale GNN. Results are not evidence of real-world fraud-detection performance. Do not use this project to block payments, accuse individuals, or make financial decisions without independent validation, governance, human review and appropriate regulatory controls. No live banking or UPI integration is included.

### Author

**Mohamed Riyaz S.**  
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai
