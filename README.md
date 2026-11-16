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

### Quick Start

- Ensure Python 3.10 or later is installed.
- Install the required dependencies using `requirements.txt`.
- Generate synthetic transaction data.
- Train the fraud detection model using the training script.
- Launch the interactive Streamlit dashboard.
- Start the FastAPI backend separately for API-based fraud detection.
- Access trained model artifacts in `artifacts/` and evaluation reports in `reports/`.

### Using Custom Datasets

- Prepare a transaction dataset in CSV format.
- Include transaction ID, sender ID, receiver ID, transaction amount, timestamp hour, payment channel, and fraud labels.
- Ensure `is_fraud` contains binary values (0 or 1) for model training and evaluation.
- Execute the training pipeline with your custom dataset path.
- Review the generated model artifacts and evaluation reports.

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

### Limitations

- This repository is a research and portfolio prototype, not a production-ready fraud detection system.
- Uses entirely synthetic transaction data with no real banking, customer, or UPI records.
- Implements a compact educational GraphSAGE-style model rather than a production-scale Graph Neural Network.
- Model performance does not establish real-world fraud detection effectiveness.
- No live banking, UPI, payment gateway, or financial network integration is included.
- Real-world deployment requires independent validation, scalability testing, and regulatory assessment.

### Responsible Use

- Use this project strictly for educational, research, and portfolio demonstration purposes.
- Do not use predictions as the sole basis for blocking transactions or restricting financial accounts.
- Never use model outputs to accuse individuals of fraudulent activity without verified evidence and human investigation.
- Protect sensitive financial information and follow applicable data privacy regulations.
- Evaluate potential model bias, false positives, and false negatives before practical application.
- Require human oversight, independent validation, and appropriate governance for financial decisions.
- Do not upload real customer information, banking credentials, UPI IDs, or confidential transaction records into this demo.

**Disclaimer:** This project demonstrates AI-driven financial transaction analytics and graph-based fraud intelligence. It is not a certified financial security solution and should not replace established fraud investigation, banking controls, or regulatory compliance procedures.

### Author

**Mohamed Riyaz S.**  
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai
