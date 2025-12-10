# ws_test.py - FINALIZED FOR ROBUSTNESS

import asyncio
import websockets
import json
import base64
import sys
from typing import Optional, Any

# The URL of your running FastAPI WebSocket endpoint
WS_URL = "ws://127.0.0.1:8000/pose/stream"
FRAME_COUNT = 5
# ADDED DELAY: Give the camera hardware a safe margin to open
STARTUP_DELAY_S = 3.0

# --- Helper function for robust timestamp formatting ---
def format_timestamp(ts: Optional[Any]) -> str:
    """Safely formats the timestamp, handling None or non-numeric types."""
    if ts is None or not isinstance(ts, (int, float)):
        return "N/A"
    try:
        # Use a string format method to avoid f-string parsing issues on type edges
        return "%.3f" % ts
    except ValueError:
        return "N/A"

async def receive_frames():
    """Connects to the WebSocket after a delay and prints the structure of incoming frames."""
    
    # --- STEP 1: WAIT FOR ENGINE TO STABILIZE ---
    print(f"[Stabilization] Waiting {STARTUP_DELAY_S} seconds for camera/model initialization...")
    await asyncio.sleep(STARTUP_DELAY_S) 
    print("[Stabilization] Attempting WebSocket connection...")
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print(f"[{'='*10} CONNECTED TO {WS_URL} {'='*10}]")
            
            # Receive the first message (could be a FramePayload or an error JSON)
            initial_message = await websocket.recv()
            
            try:
                data = json.loads(initial_message)
                if isinstance(data, dict) and 'error' in data:
                    print(f"\n[SERVER ERROR] {data['error']}")
                    sys.exit(1)
            except json.JSONDecodeError:
                # If json.loads fails, assume the raw message is the first successful frame string.
                pass 
                
            print(f"\nReceiving first {FRAME_COUNT} frames...")
            print("-" * 50)
            
            # Prepare the list of payload strings to process
            payload_messages = [initial_message] + [await websocket.recv() for _ in range(FRAME_COUNT - 1)]

            # --- STEP 2: PROCESS RECEIVED FRAMES ---
            for i, payload_str in enumerate(payload_messages):
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    print(f"Frame {i+1}: Failed to decode JSON payload.")
                    continue
                
                # Extract data for verification checks
                ts = payload.get('timestamp')
                keypoints_list = payload.get('keypoints_list', [])
                computed_angles = payload.get('computed_angles', {})
                frame_data_len = len(payload.get('frame_base64', ''))
                
                # Print results using the robust helper function
                angles_count = len(computed_angles)
                
                print(f"Frame {i+1}:")
                print(f"  - Timestamp: {format_timestamp(ts)} s")
                print(f"  - Keypoints: {len(keypoints_list)} detected")
                print(f"  - Angles: {angles_count} computed ({'elbow_left_flex' in computed_angles: True if Present})")
                print(f"  - Frame Data Length: {frame_data_len // 1024} KB")
                
                if i == 0:
                    print("-" * 50)
                
    except ConnectionRefusedError:
        print(f"\n[CONNECTION REFUSED] Ensure FastAPI server is running.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(receive_frames())