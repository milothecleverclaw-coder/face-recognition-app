from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Face Recognition App")

# Get the directory where this script is located
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend", "dist")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "face-recognition-app"}

# Mount static files from the dist directory
if os.path.exists(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

# Catch-all route to serve index.html for SPA routing
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    index_path = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not built. Run: cd frontend && npm run build"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
