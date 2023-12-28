import base64
import os
import sys
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Optional

import attrs
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from PIL import Image
from pydantic import BaseModel

import modules.async_worker
import muse_helper.task_queue
from muse_helper.async_task import async_task_to_response
from muse_helper.exception import QueueFullException

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源列表
    allow_credentials=True,  # 允许跨源Cookie
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

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
    refiner_switch: float = 0.8
    prompt: str = ""
    negative_prompt: str = ""
    mode: str = "Speed"
    style: Optional[List[str]] = []
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    seed: Optional[int] = None
    sharpness: float = 2
    sampler: str = "dpmpp_2m_sde_gpu"
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
    inpaint_additional_prompt: str = ""
    # control net
    control_images: List[dict] = []
    # used to control pipeline
    current_tab: Optional[str] = "inpaint"
    # overwrite some default value
    overwrite_vary_strength: Optional[float] = -1.0
    overwrite_upscale_strength: Optional[float] = -1.0


def load_base64(base64_string: str):
    base64_bytes = base64_string.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)
    image = Image.open(BytesIO(image_bytes))

    return np.array(image)


@app.post("/v1/generation")
async def generation(req: UniversalRequest, current_user=Depends(get_current_user)):
    try:
        task = muse_helper.task_queue.task_queue.add_task(dict(req))
    except QueueFullException:
        return JSONResponse(
            status_code=429,
            content={"error": "Server is busy processing other requests"},
        )

    return attrs.asdict(task)


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


@app.get("/v1/result/{task_id}")
def result(task_id: str):
    res = muse_helper.task_queue.task_queue.get_task_result(task_id)

    try:
        if res is not None:
            response_data = async_task_to_response(res.async_task)

            return {"task_id": res.task_id, **response_data}
        else:
            return {"message": "Task not found"}
    except Exception as e:
        e.with_traceback()
        return {"message": f"{e}"}


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
