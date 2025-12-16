# Image augmentation UI (Vue + Vite)

A small UI for the FastAPI backend in the repo root. Upload an image, compose a pipeline of augmentations and preview the processed result.

## Quick start

```bash
cd frontend
cp .env.example .env       # optional: change VITE_API_BASE
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and expects the backend at `http://localhost:8000` unless `VITE_API_BASE` is set.
