# Cheel Nazar (चील नज़र) 🦅

> **Tactical Drone Reconnaissance & 3D Geospatial Intelligence Engine**
> 
> A web-based prototype designed to transform single-pass aerial drone video into a georeferenced 3D point cloud/mesh with AI-driven damage and hazard analytics.

---

## Project Structure

```text
cheel-nazar/
├── frontend/             # React + Vite + TypeScript tactical dark dashboard
│   ├── src/
│   │   ├── components/   # Modular dashboard placeholder panels
│   │   │   ├── Header.tsx           # Mission status, telemetry & backend connectivity
│   │   │   ├── VideoUploader.tsx    # Drone video & telemetry ingestion dropzone
│   │   │   ├── ProcessingStatus.tsx # End-to-end photogrammetry pipeline stepper
│   │   │   ├── MapViewer3D.tsx      # 3D viewport canvas placeholder (Three.js ready)
│   │   │   ├── AIAnalysis.tsx       # AI damage, vehicle & obstacle detection
│   │   │   ├── Measurements.tsx     # Distance, area, volume & elevation tools
│   │   │   └── ConfidenceMetric.tsx # Photogrammetric uncertainty & QA/QC metrics
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── backend/              # Python FastAPI REST API server
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints/
│   │   │       │   ├── health.py         # GET /health & /api/v1/health
│   │   │       │   ├── video.py          # Video upload & frame extraction stubs
│   │   │       │   ├── reconstruction.py # 3D SfM & photogrammetry stubs
│   │   │       │   └── analysis.py       # YOLO & Depth model stubs
│   │   │       └── router.py             # Modular router aggregator
│   │   ├── core/
│   │   │   └── config.py                 # Paths, uploads dir, CORS configuration
│   │   └── main.py                       # FastAPI application entrypoint
│   └── requirements.txt                  # Minimal lightweight backend dependencies
├── ai/                   # Modular package for future YOLO/Depth models
├── reconstruction/       # Modular package for future COLMAP/Open3D/georeferencing
├── data/                 # Sample datasets, camera calibration & flight logs
├── uploads/              # Raw drone footage staging area
├── .gitignore            # Git ignore rules for React + Python + photogrammetry
└── README.md             # Project documentation & run manual
```

---

## Prerequisites

- **Node.js**: v18.0.0 or higher (v20+ recommended)
- **npm**: v9.0.0 or higher
- **Python**: 3.10, 3.11, or 3.12

---

## Running the Backend (FastAPI)

### 1. Open Terminal and Navigate to Backend
```bash
cd backend
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *(If using MSYS2 or Git Bash: `source .venv/bin/activate`)*

- **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Development Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Verify Backend
- **Health Check**: Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) in your browser. Expected response:
  ```json
  {
    "status": "ok",
    "app": "Cheel Nazar Backend",
    "version": "0.1.0",
    "timestamp": "2026-10-03T18:30:00.000000+00:00"
  }
  ```
- **Interactive OpenAPI Documentation**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Running the Frontend (React + Vite + TypeScript)

### 1. Open a Separate Terminal and Navigate to Frontend
```bash
cd frontend
```

### 2. Install Frontend Dependencies
```bash
npm install
```

### 3. Start Vite Development Server
```bash
npm run dev
```

### 4. Access the Tactical Dashboard
Open [http://localhost:5173](http://localhost:5173) in your browser.

The dashboard will automatically connect to the backend at `http://127.0.0.1:8000/health` and display a live `API: ONLINE` indicator in the header bar.

### 5. Build for Production (Optional)
```bash
npm run build
```
The compiled output will be generated in `frontend/dist/`.

---

## Pipeline Roadmap (Upcoming Phases)

1. **Drone Video Ingestion & Telemetry Parser**:
   - Extraction of keyframes via OpenCV with blur rejection.
   - Parsing of DJI / PX4 subtitle telemetry (`.srt`) and GPX/CSV tracks.
2. **3D Reconstruction & Photogrammetry**:
   - Feature detection & matching using SIFT / SuperPoint / LightGlue.
   - Structure-from-Motion (SfM) via COLMAP / pycolmap.
   - Dense point cloud generation and Poisson surface meshing via Open3D.
3. **AI Hazard & Damage Inference**:
   - Object detection and damage segmentation with YOLOv11 and Depth Anything v2.
   - 3D bounding box projection onto georeferenced coordinate space.
4. **Three.js Tactical 3D Map Viewer**:
   - Interactive point cloud rendering with LOD (Level of Detail).
   - Volumetric measurement tools, distance pickers, and uncertainty heatmaps.
