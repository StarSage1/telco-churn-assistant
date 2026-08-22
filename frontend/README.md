# ChurnSignal frontend

The frontend is the local marketing interface for conversational churn assessment. It shows all 19 required customer fields, profile readiness, the final churn score and threshold, marketing signals, the recommended action, professional report download, and CSV/Excel bulk scoring.

## Run locally

Start the FastAPI backend from the repository root first:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn src.Api:app --host 127.0.0.1 --port 8000
```

Then start the frontend in this directory:

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`.

The API URL defaults to `http://127.0.0.1:8000`. Set `NEXT_PUBLIC_API_URL` only when the backend runs somewhere else.

## Verification

```powershell
npm run build
npm test
npm run lint
```

The application is intentionally local-first. Customer conversation data is sent only to the local FastAPI service and local Ollama model.
