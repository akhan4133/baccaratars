Baccarat Reconstructed Backend (Render-friendly)

How it works:
- Endpoints:
  - GET /ping
  - POST /upload-history  (json {history: [...] } )
  - POST /train           (json { history: [...], config: {...} })
  - POST /predict         (json { history: [...], seq_len: N, model: optional })
  - GET  /models
  - GET  /train-history   (latest model's training history)

Deploy to Render:
- Push this repo to GitHub and create a Render Web Service using this repo.
- Build command: pip install -r requirements.txt
- Start command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1
- Make sure to set environment variable RENDER=true if you want Render-light behavior (optional).

Notes:
- On Render, the backend will use small defaults (epochs <=4, units <=48) to avoid OOM/timeouts.
- For heavy training, run this locally (without RENDER env) on a machine with more resources.
