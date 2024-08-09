import os
import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import List, Tuple, Optional
import asyncio
import base64
import json
import threading
import redis
from redis.exceptions import ConnectionError as RedisConnectionError
import time
from PIL import Image, ImageDraw

from mechanisms.segmentation_pipe import load_model, ground_image, sam_seg_rects
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

app = FastAPI()

# Global variables
sam2_model = None
grounding_dino_model = None
sam2_predictor = None
camera_stream = None

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Redis queues
FRAME_QUEUE = "frame_queue"
PROCESSING_QUEUE = "processing_queue"
RESULT_QUEUE = "result_queue"

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


class SegmentationConfig(BaseModel):
    text_prompt: str
    box_threshold: float = 0.3
    text_threshold: float = 0.3
    polygon: List[Tuple[int, int]]


class CameraConfig(BaseModel):
    camera_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@app.on_event("startup")
async def startup_event():
    global sam2_model, grounding_dino_model, sam2_predictor

    # Load SAM2 model
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Load GroundingDINO model
    config_file = "./gd_configs/grounding_dino_config.py"
    checkpoint_path = "./checkpoints/groundingdino_swint_ogc.pth"
    grounding_dino_model = load_model(config_file, checkpoint_path).eval().to("cuda")

    # Enable TF32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@app.post("/set_camera")
async def set_camera(config: CameraConfig):
    global camera_stream
    if camera_stream is not None:
        camera_stream.release()
    camera_stream = cv2.VideoCapture(config.camera_id)
    camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    camera_stream.set(cv2.CAP_PROP_FPS, config.fps)
    return {"message": "Camera configuration updated"}


def camera_reader():
    global camera_stream
    while True:
        if camera_stream is not None and camera_stream.isOpened():
            ret, frame = camera_stream.read()
            if ret:
                # Encode frame to JPEG
                _, buffer = cv2.imencode(".jpg", frame)
                frame_data = base64.b64encode(buffer).decode("utf-8")

                # Push frame to Redis queue
                try:
                    redis_client.lpush(FRAME_QUEUE, frame_data)
                    redis_client.ltrim(
                        FRAME_QUEUE, 0, 9
                    )  # Keep only the 10 most recent frames
                except RedisConnectionError:
                    print("Failed to connect to Redis server")
        else:
            time.sleep(0.1)


def frame_processor():
    while True:
        try:
            # Pop frame and config from processing queue
            data = redis_client.brpop(PROCESSING_QUEUE, timeout=1)
            if data:
                _, item = data
                frame_data, config_data = json.loads(item)

                # Decode frame
                frame = cv2.imdecode(
                    np.frombuffer(base64.b64decode(frame_data), np.uint8),
                    cv2.IMREAD_COLOR,
                )

                # Process frame
                config = SegmentationConfig(**json.loads(config_data))
                result = process_frame(frame, config)

                # Push result to result queue
                redis_client.lpush(RESULT_QUEUE, json.dumps(result))
                redis_client.ltrim(
                    RESULT_QUEUE, 0, 9
                )  # Keep only the 10 most recent results
        except RedisConnectionError:
            print("Failed to connect to Redis server")
            time.sleep(1)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await websocket.receive_json()
        config_data = json.dumps(config)

        while True:
            # Check for new frame
            frame_data = redis_client.rpop(FRAME_QUEUE)
            if frame_data:
                # Push to processing queue
                redis_client.lpush(
                    PROCESSING_QUEUE, json.dumps((frame_data.decode(), config_data))
                )

            # Check for new result
            result_data = redis_client.rpop(RESULT_QUEUE)
            if result_data:
                result = json.loads(result_data)
                await websocket.send_json(result)
            else:
                await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await websocket.close()


def process_frame(frame, config):
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform box segmentation
    image_with_box, pred_dict = box_segment(
        image, config.text_prompt, config.box_threshold, config.text_threshold
    )

    # Perform SAM segmentation
    all_masks = sam_seg_rects(sam2_predictor, None, None, image, pred_dict["boxes"])

    # Crop masks to polygon
    cropped_masks = [crop_to_polygon(mask, config.polygon) for mask in all_masks]

    # Process object masks
    object_masks = generate_object_masks(image)
    non_overlapping_masks = process_multiple_masks(cropped_masks, object_masks)

    # Create final masked image
    final_mask = create_final_mask(non_overlapping_masks)
    masked_image = apply_mask_to_image(image, final_mask)

    # Convert masked image to base64
    _, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)
    )
    masked_image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Get bounding box
    bounding_box = get_bounding_box(final_mask)

    return {"masked_image": masked_image_base64, "bounding_box": bounding_box}


def box_segment(image, text_prompt, box_threshold, text_threshold):
    with torch.no_grad():
        image_with_box, size, boxes_filt, pred_phrases, pred_dict = ground_image(
            grounding_dino_model, text_prompt, image, box_threshold, text_threshold
        )
    return image_with_box, pred_dict


def crop_to_polygon(mask, poly):
    # Create a polygon mask
    poly_mask = Image.new("L", mask.shape[-2:], 0)
    ImageDraw.Draw(poly_mask).polygon(poly, outline=1, fill=1)
    poly_mask = np.array(poly_mask)

    # Apply the polygon mask
    cropped_mask = mask * poly_mask

    # Get the bounding box of the polygon
    non_zero = np.nonzero(poly_mask)
    top, left = np.min(non_zero, axis=1)
    bottom, right = np.max(non_zero, axis=1)

    # Crop the mask
    cropped_mask = cropped_mask[top : bottom + 1, left : right + 1]

    return cropped_mask, (left, top, right, bottom)


def process_multiple_masks(all_cropped_masks, object_masks, threshold=0.95):
    non_overlapping_masks = object_masks.copy()
    for pig_mask, _ in all_cropped_masks:
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


def apply_mask_to_image(image, mask):
    if mask is None:
        return image
    masked_image = np.array(image)
    masked_image[~mask] = [0, 0, 0]  # Set background to black
    return Image.fromarray(masked_image)


def get_bounding_box(mask):
    if mask is None:
        return None
    non_zero = np.nonzero(mask)
    if len(non_zero[0]) == 0:
        return None
    top, left = np.min(non_zero, axis=1)
    bottom, right = np.max(non_zero, axis=1)
    return (int(left), int(top), int(right), int(bottom))


def generate_object_masks(image):
    # This is a placeholder function. In a real-world scenario, you would use
    # an object detection model to generate masks for all objects in the image.
    # For this example, we'll create a dummy mask.
    dummy_mask = np.zeros((image.height, image.width), dtype=bool)
    dummy_mask[50:150, 50:150] = True  # Create a small square mask
    return [{"segmentation": dummy_mask}]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
