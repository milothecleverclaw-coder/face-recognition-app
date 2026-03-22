# Face Recognition App

A mobile-first face recognition web application built with Vite.js and FastAPI.

## Features

- **Face Registration**: Capture and store face descriptors in browser localStorage
- **Live Monitoring**: Real-time face detection and matching with visual alerts
- **Progress UI**: Visual feedback during model loading from CDN
- **Mobile-First**: Responsive dark theme design
- **Tunnel Ready**: Works behind Cloudflare tunnel or any reverse proxy

## Tech Stack

- **Backend**: FastAPI, uvicorn
- **Frontend**: Vite.js (vanilla JS), face-api.js
- **Face Models**: Downloaded from GitHub CDN on first visit, cached in browser

## Quick Start

```bash
cd /home/node/.openclaw/workspace-milo/face-app
./start.sh
```

This will:
1. Install frontend dependencies (if needed)
2. Build the Vite.js frontend
3. Install backend dependencies (if needed)
4. Start FastAPI server on port 80

## Manual Build & Run

```bash
# Build frontend
cd frontend
npm install --production=false
npm run build

# Install backend dependencies
cd ../backend
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Start server
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 80
```

## Pages

- **/** (Register): Camera view to capture and register your face
- **/#/monitor** (Monitor): Live face recognition and matching

## How It Works

1. **Model Loading**: On first visit, face-api.js models are downloaded from GitHub CDN with progress indication
2. **Registration**: Detect face, extract 128-dimensional descriptor, save to localStorage
3. **Monitoring**: Continuously detect faces and compare with saved descriptor using Euclidean distance
4. **Matching**: Threshold of 0.6 (lower = stricter) determines MATCH vs ALERT

## File Structure

```
face-app/
├── backend/
│   ├── main.py              # FastAPI app serving static files
│   ├── requirements.txt     # Python dependencies
│   └── venv/               # Virtual environment (created on first run)
├── frontend/
│   ├── package.json        # npm dependencies
│   ├── vite.config.js      # Vite configuration
│   ├── index.html          # HTML template
│   ├── src/
│   │   ├── main.js        # Face recognition logic
│   │   └── style.css      # Dark theme styles
│   ├── node_modules/      # npm packages (created on first run)
│   └── dist/              # Built frontend (created after build)
├── start.sh               # Build and start script
└── README.md              # This file
```

## API Endpoints

- `GET /api/health` - Health check endpoint
- `GET /*` - Serves the SPA (all routes return index.html)

## Browser Compatibility

- Requires modern browser with:
  - MediaDevices API (camera access)
  - WebGL (face-api.js)
  - localStorage

## Security Notes

- Face descriptors are stored locally in browser localStorage
- No data is sent to any server
- Models are loaded from GitHub CDN
- Works entirely client-side after initial load

## Troubleshooting

**Camera not working?**
- Ensure HTTPS or localhost (required for camera access)
- Check browser permissions

**Models not loading?**
- Check internet connection
- Verify GitHub CDN is accessible

**Build fails?**
- Ensure Node.js 18+ is installed
- Try deleting `node_modules` and `package-lock.json`, then rebuild
