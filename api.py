import base64
import os
import sys
import threading
import time
from io import BytesIO
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import modules.advanced_parameters
import modules.async_worker as worker

app = FastAPI()

active_request = None
request_lock = threading.Lock()


class UniversalRequest(BaseModel):
    model: str = "sd_xl_base_1.0_0.9vae.safetensors"
    refiner: str = "sd_xl_refiner_1.0_0.9vae.safetensors"
    refiner_switch: float = 0.667
    prompt: str = ""
    negative_prompt: str = ""
    mode: str = "Speed"
    style: Optional[List[str]] = []
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    seed: Optional[int] = None
    sharpness: float = 2
    sampler: str = "dpmpp_sde_gpu"
    scheduler: str = "karras"
    guidance_scale: float = 7
    enable_prompt_expansion: bool = False
    lora_models: List[dict] = []
    # advanced configuration
    controlnet_softness: float = 0.25  # 0 - 1
    canny_low_threshold: int = 64  # 0 - 255
    canny_high_threshold: int = 128  # 0 - 255
    inpaint_engine: str = "v1"  # v1, v2.5
    refiner_swap_method: str = "joint"  # joint, separate, vae
    enable_free_u: bool = True
    free_u_b1: float = 1.01
    free_u_b2: float = 1.02
    free_u_s1: float = 0.99
    free_u_s2: float = 0.95
    # input_image
    enable_input_image: bool = False
    uov_mode: str = "Disabled"
    uov_image: Optional[str] = None
    inpaint_image: Optional[str] = None
    outpaint_mode: List[str] = []
    mask_image: Optional[str] = None
    # control net
    control_images: List[dict] = []
    # used to control pipeline
    current_tab: Optional[str] = "inpaint"


def load_base64(base64_string: str):
    base64_bytes = base64_string.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)
    image = Image.open(BytesIO(image_bytes))

    return np.array(image)


def handler(req: UniversalRequest):
    modules.advanced_parameters.set_all_advanced_parameters(
        1.5,
        0.8,
        0.3,
        7.0,
        req.sampler,  # "dpmpp_2m_sde_gpu",
        req.scheduler,  # "karras",
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        False,
        True,
        False,
        req.controlnet_softness,  # 0.25,
        req.canny_low_threshold,
        req.canny_high_threshold,
        req.inpaint_engine,  # "v1"
        req.refiner_swap_method,  # "joint",
        req.enable_free_u,  # True,
        req.free_u_s1,  # 1.01,
        req.free_u_s2,  # 1.02,
        req.free_u_b1,  # 0.99,
        req.free_u_b2,  # 0.95,
    )

    # preprocess lora args
    lora_args_list = []
    if len(req.lora_models) < 5:
        # add basic offset lora
        lora_args_list.append("sd_xl_offset_example-lora_1.0.safetensors")
        lora_args_list.append(0.5)
    for item in req.lora_models:
        lora_args_list.append(item["model"])
        lora_args_list.append(item["weight"])
    while len(lora_args_list) < 10:
        lora_args_list.append("None")
        lora_args_list.append(0.5)

    # preprocess control net args
    control_net_args_list = []
    for item in req.control_images:
        control_net_args_list.append(load_base64(item["image"]))
        control_net_args_list.append(item["stop_at"])
        control_net_args_list.append(item["weight"])
        control_net_args_list.append(item["mode"])
    while len(control_net_args_list) < 16:
        control_net_args_list.append(None)
        control_net_args_list.append(0.5)
        control_net_args_list.append(0.6)
        control_net_args_list.append("Image Prompt")

    style = req.style
    if style is None:
        style = []
    if req.enable_prompt_expansion:
        style.append("Fooocus V2")

    worker_args = (
        req.prompt,  # str in 'parameter_8' Textbox component
        req.negative_prompt,  # str in 'Negative Prompt' Textbox component
        style,  # List[str] in 'Image Style' Checkboxgroup component
        req.mode,  # str in 'Performance' Radio component
        f"{req.width}×{req.height}",  # str in 'Aspect Ratios' Radio component
        req.batch_size,  # int | float (numeric value between 1 and 32) in 'Image Number' Slider component
        req.seed,  # int | float in 'Seed' Number component
        req.sharpness,  # int | float (numeric value between 0.0 and 30.0) in 'Sampling Sharpness' Slider component
        req.guidance_scale,  # int | float (numeric value between 1.0 and 30.0) in 'Guidance Scale' Slider component
        req.model,  # str (Option from: ['sd_xl_base_1.0_0.9vae.safetensors', 'sd_xl_refiner_1.0_0.9vae.safetensors']) in 'SDXL Base Model' Dropdown component
        req.refiner,  # str (Option from: ['None', 'sd_xl_base_1.0_0.9vae.safetensors', 'sd_xl_refiner_1.0_0.9vae.safetensors']) in 'SDXL Refiner' Dropdown component
        req.refiner_switch,
        # 5 loras
        *lora_args_list,
        req.enable_input_image,  # bool in 'Input Image' Checkbox component
        req.current_tab,  # str in 'parameter_68' Textbox component
        req.uov_mode,  # str in 'Upscale or Variation:' Radio component
        load_base64(req.uov_image) if req.uov_image is not None else None,
        req.outpaint_mode,
        {
            "image": load_base64(req.inpaint_image),
            "mask": load_base64(req.mask_image)
            if req.mask_image is not None
            else np.zeros_like(load_base64(req.inpaint_image)),
        }
        if req.inpaint_image is not None
        else None,
        *control_net_args_list,
    )
    worker.buffer.append(list(worker_args))
    finished = False
    results = []

    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == "preview":
                # percentage, title, image = product
                # yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                # gr.update(visible=True, value=image) if image is not None else gr.update(), \
                # gr.update(visible=False)
                pass
            if flag == "finish":
                # yield gr.update(visible=False), \
                #     gr.update(visible=False), \
                #     gr.update(visible=True, value=product)
                # print(product)
                finished = True

                if len(product) >= 2:
                    # ignore last one, which is image wall
                    product = product[:-1]

                for item in product:
                    im = Image.fromarray(item)

                    # Save the image to a buffer
                    with BytesIO() as buffer:
                        im.save(buffer, "PNG")
                        img_bytes = buffer.getvalue()

                    # Encode the image to base64
                    base64_bytes = base64.b64encode(img_bytes)
                    base64_string = base64_bytes.decode("utf-8")
                    results.append(base64_string)

    return results


@app.post("/v1/generation")
async def generation(req: UniversalRequest):
    global active_request

    if request_lock.acquire(blocking=False):
        try:
            response = await run_in_threadpool(handler, req)
        finally:
            active_request = None
            request_lock.release()
        return response

    # If we can't acquire the lock, that means another
    # request is being processed. Return 503 error.
    return JSONResponse(
        status_code=503, content={"error": "Server is busy processing another request"}
    )


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root)
    os.chdir(root)
    backend_path = os.path.join(root, "backend", "headless")
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    port = os.environ.get("PORT", "7860")

    uvicorn.run("api:app", host="0.0.0.0", port=int(port), log_level="critical")
