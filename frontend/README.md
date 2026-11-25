# Batik Checker — React frontend

Web UI for the Batik Checker: upload a batik photo (or take one with the phone
camera) and see the predicted motif with confidence bars.

## Setup

Requires Node.js 18+.

```powershell
cd frontend
npm install
npm run dev
```

The dev server runs on http://localhost:5173 and proxies `/api/*` to the FastAPI
backend on http://localhost:8000 (see `vite.config.js`), so start the backend too.

## Structure

```
frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx                  # React entry point
    ├── App.jsx                   # main page: upload + results
    ├── api.js                    # calls the FastAPI backend
    ├── config.js                 # API base URL, allowed types, max size
    ├── index.css                 # styles
    └── components/
        ├── PredictionList.jsx    # top-k predictions with confidence bars
        └── ImagePreview.jsx      # input + intermediate images
```

## Next steps (to continue later)

- Add a **motif description card** using `GET /motifs`.
- Show the **LBP map** image (add it to the backend response).
- Nicer styling / responsive layout.
- Handle the "model not loaded" (503) state gracefully.
- Deploy: build with `npm run build`, serve `dist/` behind the FastAPI app.
