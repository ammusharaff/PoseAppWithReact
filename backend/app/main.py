# backend/app/main.py

import webbrowser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import sys
import time
import json
import traceback
from typing import Dict, Any, Optional, List
import threading
import signal
import zipfile
import io

from src.poseapp.pose_engine import PoseEngine
from src.poseapp.data_models import (
    StartSessionRequest, StopSessionResponse, SetModeRequest, FramePayload, UpdateFPSRequest
)

# --- PATH CONFIG ---
def get_base_paths():
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
        static_dir = os.path.join(base, "static")
        assets_dir = os.path.join(base, "src", "assets")
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        backend_root = os.path.dirname(app_dir)
        static_dir = os.path.join(backend_root, "static")
        assets_dir = os.path.join(backend_root, "src", "assets")
    return static_dir, assets_dir

STATIC_DIR, ASSETS_DIR = get_base_paths()

def get_writable_dir():
    if "APPIMAGE" in os.environ:
        return os.path.dirname(os.environ["APPIMAGE"])
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.getcwd()

writable_base = get_writable_dir()
if not os.access(writable_base, os.W_OK):
    writable_base = os.path.join(os.path.expanduser("~"), "PoseApp")

SESSION_ROOT = os.path.join(writable_base, "sessions")
os.makedirs(SESSION_ROOT, exist_ok=True)

# --- APP SETUP ---
app = FastAPI(title="PoseApp")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(ASSETS_DIR):
    app.mount("/media", StaticFiles(directory=ASSETS_DIR), name="media")

if os.path.exists(STATIC_DIR):
    static_assets = os.path.join(STATIC_DIR, "assets")
    if os.path.exists(static_assets):
        app.mount("/assets", StaticFiles(directory=static_assets), name="static_web")
    
    @app.get("/")
    async def read_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# --- GLOBAL STATE ---
engine: PoseEngine = PoseEngine(session_root_override=SESSION_ROOT)
executor = ThreadPoolExecutor(max_workers=2)
is_streaming: bool = False
active_websockets = []
target_fps_delay: float = 0.0 
streaming_task: Optional[asyncio.Task] = None # NEW: Track the loop task

# --- HELPER FUNCTIONS ---
async def run_in_executor_async(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

async def streaming_loop():
    global is_streaming, target_fps_delay
    print("[Streaming Loop] Starting...")
    while is_streaming and engine._cap and engine._cap.isOpened():
        start_t = time.time()
        try:
            payload = await run_in_executor_async(engine.process_frame)
            if payload is None:
                # Engine signaled stop (e.g. camera disconnect)
                break
            
            data = payload.model_dump_json()
            for ws in active_websockets:
                await ws.send_text(data)
            
            elapsed = time.time() - start_t
            if target_fps_delay > 0:
                sleep_time = max(0, target_fps_delay - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.001) 
            else:
                 await asyncio.sleep(0.001)
            
        except Exception as e:
            print(f"[Streaming Loop] Error: {e}")
            break
    
    print("[Streaming Loop] Exiting...")
    is_streaming = False # Ensure flag is down

async def stop_streaming():
    global is_streaming, active_websockets, streaming_task
    
    # 1. Signal the loop to stop
    is_streaming = False
    
    # 2. Wait for the loop to actually finish (Critical for OpenCV safety)
    # We check if we are running INSIDE the task to avoid deadlock
    try:
        current_task = asyncio.current_task()
        if streaming_task and streaming_task != current_task and not streaming_task.done():
            print("[System] Waiting for stream loop to finish...")
            await asyncio.wait_for(streaming_task, timeout=2.0)
    except Exception as e:
        print(f"[System] Warning during stop wait: {e}")

    # 3. Now it is safe to release camera
    engine.stop_camera()
    
    # 4. Close websockets
    for ws in active_websockets:
        try: await ws.close(code=1000)
        except: pass
    active_websockets = []

# --- ENDPOINTS ---
@app.post("/camera/start", response_model=StopSessionResponse)
async def start_camera(req: StartSessionRequest):
    global is_streaming, target_fps_delay, streaming_task
    
    # Ensure clean stop before start
    await stop_streaming()
    
    if req.target_fps >= 60:
        target_fps_delay = 0.0
    else:
        target_fps_delay = 1.0 / float(req.target_fps)
    
    backend = req.model_backend
    if backend == "MoveNet": backend = "MoveNet_Lightning"
    
    success = await run_in_executor_async(engine.start_camera, req)
    if success:
        is_streaming = True
        streaming_task = asyncio.create_task(streaming_loop())
        return StopSessionResponse(status="success", message=f"Started {backend}")
    return StopSessionResponse(status="error", message="Failed")

@app.post("/camera/update_fps")
async def update_fps(req: UpdateFPSRequest):
    global target_fps_delay
    if req.target_fps >= 60:
        target_fps_delay = 0.0
    else:
        target_fps_delay = 1.0 / float(req.target_fps)
    return {"status": "success", "message": f"FPS updated to {req.target_fps}"}

@app.post("/camera/stop", response_model=StopSessionResponse)
async def stop_camera():
    await stop_streaming()
    return StopSessionResponse(status="success", message="Stopped")

@app.post("/session/mode", response_model=StopSessionResponse)
async def set_mode(req: SetModeRequest):
    engine.set_mode(req)
    return StopSessionResponse(status="success", message="Mode updated")

@app.post("/data/export")
async def export_data():
    # Calling save_session_data flushes current buffers to disk
    summary = await run_in_executor_async(engine.save_session_data)
    if summary: return {"status": "success", "summary": summary}
    return {"status": "error", "message": "No data"}

@app.get("/data/download")
async def download_session():
    path = engine._session_path
    if not path or not os.path.exists(path):
        return Response(status_code=404, content="No session data found")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, path)
                zip_file.write(file_path, arcname)
                
    timestamp_name = os.path.basename(path)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=PoseApp_Session_{timestamp_name}.zip"}
    )

@app.websocket("/pose/stream")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    active_websockets.append(ws)
    try:
        while is_streaming: await asyncio.sleep(0.5)
    except: pass
    finally:
        if ws in active_websockets: active_websockets.remove(ws)

@app.on_event("shutdown")
async def shutdown():
    await stop_streaming()
    executor.shutdown(wait=False)

@app.post("/app/quit")
async def quit_app():
    loop = asyncio.get_event_loop()
    loop.call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return {"status": "success", "message": "Server shutting down..."}

if __name__ == "__main__":
    import uvicorn
    if getattr(sys, 'frozen', False):
        import threading
        def open_browser():
            time.sleep(1.5)
            webbrowser.open("http://127.0.0.1:8000")
        threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)