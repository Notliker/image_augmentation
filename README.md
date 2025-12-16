# Image augmentation service

Small FastAPI backend wrapping the existing augmentation pipeline plus a Vue UI to upload images, compose steps and preview results.

## Backend
- Create a venv and install deps: `pip install -r requirements.txt`.
- Run the API: `uvicorn app:app --reload --port 8000` (run from repo root so `data/` is reachable).
- Endpoints:
  - `GET /health` – simple liveness check.
  - `GET /augmentations` – metadata for all available operations and their parameters.
  - `POST /process` – multipart form with `file` (image) and `config` (JSON list of steps). Example config:
    ```json
    [
      {"name": "gaussian_noise", "params": {"sigma": 18}},
      {"name": "rotation", "params": {"angle": 25}}
    ]
    ```
    Returns a base64 encoded PNG plus image shape.

## Frontend (Vue + Vite)
- `cd frontend`
- `cp .env.example .env` (edit `VITE_API_BASE` if backend runs elsewhere)
- `npm install`
- `npm run dev` and open the shown URL (default `http://localhost:5173`).

The UI lets you upload an image, stack augmentations, and send the payload to the backend. Defaults for dataset-dependent augmentations point to files in `data/`.
