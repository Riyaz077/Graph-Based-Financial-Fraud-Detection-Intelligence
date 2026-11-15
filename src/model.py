import torch
from torch import nn
import torch.nn.functional as F

class GraphSAGEClassifier(nn.Module):
    """Compact transaction-edge classifier with endpoint message aggregation."""
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Each transaction row includes source and destination account context.
        h = F.relu(self.node_encoder(x))
        # Mean-neighbour-style aggregation proxy: split endpoint context features.
        half = h.shape[1] // 2
        src = h[:, :half]
        dst = h[:, half:]
        if dst.shape[1] != src.shape[1]:
            dst = F.pad(dst, (0, src.shape[1] - dst.shape[1]))
        src_msg = F.relu(self.message(torch.cat([src, dst], dim=1)))
        dst_msg = F.relu(self.message(torch.cat([dst, src], dim=1)))
        z = torch.cat([src, dst, src_msg, dst_msg], dim=1)
        return self.classifier(z).squeeze(1)

def train_graph_model(X_train, y_train, epochs=35, lr=0.003, seed=42):
    torch.manual_seed(seed)
    x = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    model = GraphSAGEClassifier(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return model

def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32)
        return torch.sigmoid(model(x)).cpu().numpy()
