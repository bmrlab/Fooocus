import base64
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from PIL import Image
from pydantic import BaseModel

import modules.advanced_parameters
import modules.async_worker as worker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源列表
    allow_credentials=True,  # 允许跨源Cookie
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

active_request = None
request_lock = threading.Lock()

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SECRET_KEY = os.environ.get("SECRET_KEY", "secret key")
ALLOW_USERNAME = os.environ.get("USERNAME", "admin")
ALLOW_PASSWORD = os.environ.get("PASSWORD", "admin")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


class Token(BaseModel):
    access_token: str
    token_type: str


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
    skipping_cn_preprocessor: bool = False
    inpaint_disable_initial_latent: bool = False
    inpaint_engine: str = "v1"  # v1, v2.5, v2.6
    inpaint_strength: float = 1.0
    inpaint_respective_field: float = 0.618
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
    inpaint_additional_prompt: str = ''
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
        False,
        1.5,
        0.8,
        0.3,
        7.0,
        req.sampler,  # "dpmpp_2m_sde_gpu",
        req.scheduler,  # "karras",
        False,  # grid
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        False,
        True,
        False,
        req.skipping_cn_preprocessor,
        req.controlnet_softness,  # 0.25,
        req.canny_low_threshold,
        req.canny_high_threshold,
        req.refiner_swap_method,  # "joint",
        req.enable_free_u,  # True,
        req.free_u_s1,  # 1.01,
        req.free_u_s2,  # 1.02,
        req.free_u_b1,  # 0.99,
        req.free_u_b2,  # 0.95,
        False,
        req.inpaint_disable_initial_latent,
        req.inpaint_engine,
        req.inpaint_strength,
        req.inpaint_respective_field,
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
        req.model,
        # str (Option from: ['sd_xl_base_1.0_0.9vae.safetensors', 'sd_xl_refiner_1.0_0.9vae.safetensors']) in 'SDXL Base Model' Dropdown component
        req.refiner,
        # str (Option from: ['None', 'sd_xl_base_1.0_0.9vae.safetensors', 'sd_xl_refiner_1.0_0.9vae.safetensors']) in 'SDXL Refiner' Dropdown component
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
        req.inpaint_additional_prompt,
        *control_net_args_list,
    )

    task = worker.AsyncTask(args=list(worker_args))
    finished = False
    worker.async_tasks.append(task)
    results = []

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == "finish":
                finished = True

                if modules.advanced_parameters.generate_image_grid:
                    # if enable image grid, ignore last image
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
async def generation(
    req: UniversalRequest,
    #  current_user=Depends(get_current_user)
):
    global active_request

    if request_lock.acquire(blocking=False):
        try:
            response = await run_in_threadpool(handler, req)
        finally:
            active_request = None
            request_lock.release()
        return response

    # If we can't acquire the lock, that means another
    # request is being processed. Return 429 error.
    return JSONResponse(
        status_code=429, content={"error": "Server is busy processing another request"}
    )


@app.post("/v1/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

    # 这里应该检查用户名和密码，示例中省略
    if username != ALLOW_USERNAME or password != ALLOW_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/v1/progress")
def progress():
    data = worker.global_progress_results
    return {"data": data}
def ini_fcbh_args():
    from args_manager import args
    return args


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root)
    os.chdir(root)
    backend_path = os.path.join(root, "backend", "headless")
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # enable this to receive cli args
    args = ini_fcbh_args()

    port = os.environ.get("PORT", "7860")

    uvicorn.run("api:app", host="0.0.0.0", port=int(port), log_level="critical")
