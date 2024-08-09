import requests

requests.post("http://localhost:8000/set_camera", json={
    "camera_id": 1,
    "width": 1280,
    "height": 720,
    "fps": 30
})

import asyncio
import websockets
import json

async def connect_to_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        config = {
            "text_prompt": "pig",
            "box_threshold": 0.3,
            "text_threshold": 0.3,
            "polygon": [[0, 0], [1280, 0], [1280, 720], [0, 720]]
        }
        await websocket.send(json.dumps(config))

        while True:
            result = await websocket.recv()
            result = json.loads(result)
            # Process the result (e.g., display the image, use the bounding box)

asyncio.get_event_loop().run_until_complete(connect_to_websocket())