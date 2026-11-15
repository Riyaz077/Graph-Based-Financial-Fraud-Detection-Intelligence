# API

Run `uvicorn api.main:app --reload`.

- `GET /health` — health check.
- `POST /demo/train` — trains the demo models on a fresh synthetic sample and returns metrics.
- `POST /demo/score` — returns a transparent illustrative rule-based score for one transaction. It is not a trained-model prediction.

Interactive API documentation is available at `/docs`.
