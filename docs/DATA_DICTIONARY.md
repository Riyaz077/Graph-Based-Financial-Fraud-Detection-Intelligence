# Data Dictionary

| Field | Description |
|---|---|
| transaction_id | Unique synthetic transaction identifier |
| sender_id | Sending account identifier |
| receiver_id | Receiving account identifier |
| amount | Transaction amount in arbitrary demo units |
| hour | Hour of day, 0–23 |
| channel | UPI, CARD, BANK_TRANSFER or WALLET |
| is_fraud | Synthetic binary target label (0/1) |

All included records are generated locally. No real personal or banking data is included.
