# Frontend (Web Preview) — Setup

This is the Vite web preview of the app. Environment variables are supported via `.env` files.

1. Copy `.env.example` -> `.env` and fill values (optional):
```
GOOGLE_MAPS_API_KEY=YOUR_REAL_API_KEY
BACKEND_URL=http://127.0.0.1:5000
```

2. Install deps and start the dev server:
```bash
npm install
npm run dev
```

3. Notes:
- The web app uses `process.env.BACKEND_URL` with a fallback to `http://127.0.0.1:5000`.
- For backend running locally, keep backend.py running on port 5000.
