import requests

requests.post("http://localhost:8000/set_camera", json={
    "camera_source": "https://rr3---sn-42u-nbosr.googlevideo.com/videoplayback?expire=1723241304&ei=-D62ZvXeI8b1j-8PzauS6Qk&ip=186.179.39.16&id=o-AF9L_m8N2lX3Q2KpzdK5mBM0N3JAUyyIEltf-ZudhhHH&itag=18&source=youtube&requiressl=yes&xpc=EgVo2aDSNQ%3D%3D&bui=AXc671K2c4fkfZiurqkHSaXNhZ3GtO4lKcsYpNE0kGOmNqWdDYbjbYQh9zy4PYKQ7P4HQI_Kbo4guNKi&vprv=1&mime=video%2Fmp4&rqh=1&gir=yes&clen=78376478&ratebypass=yes&dur=979.998&lmt=1707320135240753&c=ANDROID_CREATOR&txp=5538434&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cxpc%2Cbui%2Cvprv%2Cmime%2Crqh%2Cgir%2Cclen%2Cratebypass%2Cdur%2Clmt&sig=AJfQdSswRQIhAOXV5ukR4DppsVe-OaRz1R6LEztCnTBuiuDwHDpOA0taAiAmuJ_l7lp8RHlhs3jx9kS4NIvnEirhk0oL6JxoJQ8q3g%3D%3D&title=A%20Week%20Full%20of%20Pig%20Farming&redirect_counter=1&rm=sn-ab5eel7l&rrc=104&fexp=24350516,24350518,24350557,24350560&req_id=8a53e24d6c8a3ee&cms_redirect=yes&cmsv=e&ipbypass=yes&mh=Qb&mip=2405:4803:b4bc:c990:c45a:d350:45e7:c5a4&mm=31&mn=sn-42u-nbosr&ms=au&mt=1723218688&mv=u&mvi=3&pl=46&lsparams=ipbypass,mh,mip,mm,mn,ms,mv,mvi,pl&lsig=AGtxev0wRQIgQ_e_hEYCcORBxWyAlyISRk3zzzd5-MI03N5ibBIvPn0CIQCIw963RAnSufPb9KhM9DCUMG_ogpoY5yXpcfzCc3mwDQ%3D%3D",
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