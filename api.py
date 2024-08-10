import os
import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import asyncio
import base64
import json
import threading
import aioredis
from aioredis.exceptions import ConnectionError as RedisConnectionError
import time
from PIL import Image, ImageDraw
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from fastapi.responses import StreamingResponse
import time
from mechanisms.segmentation_pipe import load_model, ground_image, sam_seg_rects
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

app = FastAPI()

# Configuration
class Config:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    FRAME_QUEUE = "frame_queue"
    PROCESSING_QUEUE = "processing_queue"
    RESULT_QUEUE = "result_queue"
    MAX_QUEUE_SIZE = 1000

redis_config = Config()

# Global variables
sam2_model = None
grounding_dino_model = None
sam2_predictor = None
camera_stream = None
sam2_mask_generator = None
redis_client = None

class SegmentationConfig(BaseModel):
    text_prompt: str
    box_threshold: float = 0.3
    text_threshold: float = 0.3
    polygon: List[Tuple[int, int]]

class CameraConfig(BaseModel):
    camera_source: str
    width: int = 640
    height: int = 480
    fps: int = 30

@app.on_event("startup")
async def startup_event():
    global sam2_model, grounding_dino_model, sam2_predictor, sam2_mask_generator, redis_client, redis_config
    
    # Initialize Redis client
    try:
        redis_client = await aioredis.from_url(f"redis://{redis_config.REDIS_HOST}:{redis_config.REDIS_PORT}/{redis_config.REDIS_DB}")
    except RedisConnectionError:
        print("Failed to connect to Redis server during startup")
        redis_client = None
    
    # Load models
    sam2_model = load_sam2_model()
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    grounding_dino_model = load_grounding_dino_model()

    # Enable TF32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Start background tasks
    asyncio.create_task(camera_reader())
    asyncio.create_task(frame_processor())

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client is not None:
        redis_client.close()

def load_sam2_model():
    sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    return build_sam2(model_cfg, sam2_checkpoint, device="cuda")    

def load_grounding_dino_model():
    config_file = "./gd_configs/grounding_dino_config.py"
    checkpoint_path = "./checkpoints/groundingdino_swint_ogc.pth"
    return load_model(config_file, checkpoint_path, use_fp16=False).eval().to("cuda")

@app.post("/set_camera")
async def set_camera(config: CameraConfig):
    global camera_stream
    try:
        if camera_stream is not None:
            camera_stream.release()
        camera_stream = cv2.VideoCapture(config.camera_source)
        camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        camera_stream.set(cv2.CAP_PROP_FPS, config.fps)
        return {"message": "Camera configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set camera: {str(e)}")

async def camera_reader():
    global camera_stream, redis_client, redis_config
    while True:
        if camera_stream is not None and camera_stream.isOpened():
            ret, frame = camera_stream.read()
            if ret:
                frame_data = encode_frame(frame)
                if redis_client is not None:
                    await push_to_redis(redis_config.FRAME_QUEUE, frame_data)
        await asyncio.sleep(0.1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global redis_config
    await websocket.accept()
    try:
        config_data = await websocket.receive_json()
        config_data = json.dumps(config_data)

        while True:
            if redis_client is not None:
                frame_data = await redis_client.rpop(redis_config.FRAME_QUEUE)
                if frame_data:
                    await push_to_redis(redis_config.PROCESSING_QUEUE, json.dumps((frame_data.decode(), config_data)))

                result_data = await redis_client.rpop(redis_config.RESULT_QUEUE)
                if result_data:
                    result = json.loads(result_data)
                    await websocket.send_json(result)
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

async def push_to_redis(queue, data):
    global redis_client
    if redis_client is None:
        print("Redis client is not available. Skipping push to Redis.")
        return
    try:
        await redis_client.lpush(queue, data)
        await redis_client.ltrim(queue, 0, redis_config.MAX_QUEUE_SIZE - 1)
    except RedisConnectionError:
        print(f"Failed to connect to Redis server while pushing to {queue}")

async def frame_processor():
    global redis_config, redis_client
    while True:
        if redis_client is None:
            print("Redis client is not initialized. Attempting to reconnect...")
            try:
                redis_client = await aioredis.from_url(f"redis://{redis_config.REDIS_HOST}:{redis_config.REDIS_PORT}/{redis_config.REDIS_DB}")
            except RedisConnectionError:
                print("Failed to reconnect to Redis server")
                await asyncio.sleep(5)  # Wait before trying to reconnect
                continue

        try:
            data = await redis_client.brpop(redis_config.PROCESSING_QUEUE, timeout=1)
            if data:
                now = time.time()
                _, item = data
                frame_data, config_data = json.loads(item.decode())
                frame = decode_frame(frame_data)
                segment_config = SegmentationConfig(**json.loads(config_data))
                result = process_frame(frame, segment_config)
                
                await push_to_redis(redis_config.RESULT_QUEUE, json.dumps(result))
                print(f"Frame processed in {time.time() - now:.2f} seconds, FPS: {1 / (time.time() - now):.2f}")
        except RedisConnectionError:
            print("Lost connection to Redis server in frame processor")
            redis_client = None  # Reset the client so we attempt to reconnect on the next iteration
        except Exception as e:
            print(f"Error in frame processor: {str(e)}")
        await asyncio.sleep(0.1)  # Small delay to prevent tight looping

def decode_frame(frame_data):
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(frame_data), np.uint8),
        cv2.IMREAD_COLOR,
    )

def process_frame(frame, config):
    # Convert frame to RGB for processing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    now = time.time()
    # Perform box segmentation
    image_with_box, pred_dict = box_segment(image, config.text_prompt, config.box_threshold, config.text_threshold)
    print(f"Box segmentation completed in {time.time() - now:.2f} seconds")
    # Generate masks for pigs
    all_masks = sam_seg_rects(sam2_predictor, None, None, image, pred_dict["boxes"])
    
    # Crop masks to the specified polygon
    cropped_masks = [crop_mask_to_polygon(mask, config.polygon) for mask in all_masks]
    
    # Crop the image to the specified polygon
    cropped = crop_image_to_polygon(image, config.polygon)
    
    # Generate object masks
    object_masks = sam2_mask_generator.generate(np.array(cropped.convert("RGB")))
    print(f"Object masks generated in {time.time() - now:.2f} seconds")
    # Process multiple masks to get non-overlapping masks
    non_overlapping_masks = process_multiple_masks(cropped_masks, object_masks)
    
    # Create a visualization of the results
    visualization = visualize_results(frame, cropped_masks, non_overlapping_masks, config.polygon)
    
    # Prepare the output
    output = {
        "num_pigs": len(all_masks),
        "num_objects": len(non_overlapping_masks),
        "original_frame": encode_frame(frame),
        # "pig_masks": cropped_masks,
        "visualization": encode_frame(visualization)
    }
    
    return output

def visualize_results(frame, pig_masks, object_masks, polygon):
    # Create a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Draw the polygon
    cv2.polylines(vis_frame, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Visualize pig masks
    for mask in pig_masks:
        pig_overlay = vis_frame.copy()
        pig_mask_rgb = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        pig_overlay[mask > 0] = (0, 0, 255)  # Red color for pigs
        cv2.addWeighted(pig_overlay, 0.5, vis_frame, 0.5, 0, vis_frame)
    
    # Visualize object masks
    for mask in object_masks:
        obj_overlay = vis_frame.copy()
        obj_mask = mask['segmentation'].astype(np.uint8) * 255
        obj_mask_rgb = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR)
        obj_overlay[obj_mask > 0] = (255, 0, 0)  # Blue color for objects
        cv2.addWeighted(obj_overlay, 0.3, vis_frame, 0.7, 0, vis_frame)
    
    # Add text for counts
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_frame, f"Pigs: {len(pig_masks)}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis_frame, f"Objects: {len(object_masks)}", (10, 70), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return vis_frame

async def generate_visualization(websocket: WebSocket):
    global redis_client, redis_config
    try:
        while True:
            result_data = await redis_client.rpop(redis_config.RESULT_QUEUE)
            if result_data:
                result = json.loads(result_data)
                visualized_frame = decode_frame(result["visualization"])
                cv2.imwrite("visualized_frame.jpg", visualized_frame)
                print(f'Sending frame to client')
                _, buffer = cv2.imencode('.jpg', visualized_frame)
                await websocket.send_bytes(buffer.tobytes())
            else:
                await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Visualization WebSocket error: {str(e)}")
    finally:
        await websocket.close()
        
@app.websocket("/ws/visualization")
async def websocket_visualization(websocket: WebSocket):
    await websocket.accept()
    await generate_visualization(websocket)

@app.get("/stream/visualization")
async def stream_visualization():
    return StreamingResponse(generate_visualization_stream(), media_type="multipart/x-mixed-replace;boundary=frame")

async def generate_visualization_stream():
    global redis_client, redis_config
    while True:
        result_data = await redis_client.rpop(redis_config.RESULT_QUEUE)
        if result_data:
            result = json.loads(result_data)
            visualized_frame = decode_frame(result["visualization"])
            cv2.imwrite("visualized_frame.jpg", visualized_frame)
            _, buffer = cv2.imencode('.jpg', visualized_frame)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            await asyncio.sleep(0.01)
            
def box_segment(image, text_prompt, box_threshold, text_threshold):
    with torch.no_grad():
        image_with_box, size, boxes_filt, pred_phrases, pred_dict = ground_image(
            grounding_dino_model, text_prompt, image, box_threshold, text_threshold
        )
    return image_with_box, pred_dict

def crop_mask_to_polygon(input_mask, poly):
    
    # Remove the singleton dimension
    mask = input_mask[0]
    
    # Create a new mask for the polygon
    polygon_mask = Image.new('L', (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(polygon_mask).polygon(poly, outline=1, fill=255)
    polygon_mask = np.array(polygon_mask)
    
    # Apply the polygon mask
    masked = mask * (polygon_mask > 0)
    
    # Find the bounding box of the polygon
    y_coords, x_coords = np.where(polygon_mask > 0)
    top = y_coords.min()
    bottom = y_coords.max()
    left = x_coords.min()
    right = x_coords.max()
    
    # Crop the masked image to the bounding box
    cropped = masked[top:bottom+1, left:right+1]
    
    return cropped

def crop_image_to_polygon(image, polygon):
    """
    Crops the given PIL image to the specified polygon area.

    :param image: A PIL Image object.
    :param polygon: List of (x, y) tuples representing the vertices of the polygon.
    :return: A new PIL Image object cropped to the polygon area.
    """
    # Create a mask the same size as the image, filled with black (0)
    if type(image) == np.ndarray:
        image = np.uint8(np.squeeze(image))
        h, w = image.shape[:2]
        image = Image.fromarray(image)
    mask = Image.new("L", image.size, 0)

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Draw the polygon on the mask with white (255)
    draw.polygon(polygon, outline=255, fill=255)

    # Apply the mask to the image
    result = Image.new("RGB", image.size)
    result.paste(image, mask=mask)

    # Crop the image to the bounding box of the polygon
    bbox = mask.getbbox()
    cropped_image = result.crop(bbox)

    return cropped_image


def process_multiple_masks(all_cropped_masks, object_masks, threshold=0.95):
    non_overlapping_masks = object_masks.copy()
    for pig_mask in all_cropped_masks:
        for idx, obj_mask in enumerate(non_overlapping_masks):
            if check_mask_overlap(pig_mask, obj_mask["segmentation"], threshold):
                del non_overlapping_masks[idx]
                break
    return non_overlapping_masks


def check_mask_overlap(mask1, mask2, threshold=0.95):
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    mask1_area = np.sum(mask1)
    mask2_area = np.sum(mask2)
    overlap_percentage = intersection_area / min(mask1_area, mask2_area)
    return overlap_percentage > threshold


def create_final_mask(non_overlapping_masks):
    if not non_overlapping_masks:
        return None
    final_mask = np.zeros_like(non_overlapping_masks[0]["segmentation"], dtype=bool)
    for mask in non_overlapping_masks:
        final_mask = np.logical_or(final_mask, mask["segmentation"])
    return final_mask


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
