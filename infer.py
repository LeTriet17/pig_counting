import requests

requests.post("http://localhost:8000/set_camera", json={
    "camera_source": "https://rr5---sn-i3b7knlk.googlevideo.com/videoplayback?expire=1723318392&ei=GGy3ZsiXGLCDvcAP1M-lkAw&ip=112.185.54.203&id=o-AOmxFjky2S8PPWbe8_icQ1LWmCM_EVmi2OUtrQRhD_TR&itag=18&source=youtube&requiressl=yes&xpc=EgVo2aDSNQ%3D%3D&bui=AQmm2ex3LLEol6ykdwSbAO7i7lWJMi0uIhDdPpIns-09XuxRfvTRX7G8LkfEDimRxmG2UFpoe9Lp9kZ0&vprv=1&mime=video%2Fmp4&rqh=1&gir=yes&clen=78376478&ratebypass=yes&dur=979.998&lmt=1707320135240753&c=ANDROID_CREATOR&txp=5538434&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cxpc%2Cbui%2Cvprv%2Cmime%2Crqh%2Cgir%2Cclen%2Cratebypass%2Cdur%2Clmt&sig=AJfQdSswRgIhANOJ6aiL8p4zFyZiaQJyHgLzxb9XnRgZqbzilWZ7zBrUAiEA5yBqud5c-gblY7aa3wj3n0_RwzU4HXO7Wo0zo1OKsOM%3D&title=A%20Week%20Full%20of%20Pig%20Farming&rm=sn-3u-20nr7z,sn-3u-bh2zd76,sn-oguelz7e&rrc=79,79,104&fexp=24350516,24350517,24350557,24350560&req_id=b0c19c78c276a3ee&redirect_counter=3&cms_redirect=yes&cmsv=e&ipbypass=yes&mh=Qb&mip=2405:4803:b4bc:c990:c45a:d350:45e7:c5a4&mm=30&mn=sn-i3b7knlk&ms=nxu&mt=1723296521&mv=m&mvi=5&pl=46&lsparams=ipbypass,mh,mip,mm,mn,ms,mv,mvi,pl&lsig=AGtxev0wRgIhAI7dBP-ZUKe7viqKbWZx9dWAQSbCAub4zQ-NOgqutMoNAiEA27zVubHhKcrsQf_g5NYF76prQjYSwe02D3JndwWbFO8%3D",
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