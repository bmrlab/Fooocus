import base64
import os
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from PIL import Image

import muse_helper.task_queue
from muse_helper.api_model import Token, FooocusTaskInput
from muse_helper.async_task import (
    async_task_to_preview_response,
    async_task_to_result_response,
)
from muse_helper.exception import QueueFullException

import random

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


def logger(msg: str):
    """
    logger function

    TODO should not using print
    """
    print(f"[Muse] {msg}")


def convert_base64_for_logger(base64_str: str):
    return base64_str[:10]


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


def load_base64(base64_string: str):
    base64_bytes = base64_string.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)
    image = Image.open(BytesIO(image_bytes))

    return np.array(image)


@app.post("/v1/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

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


@app.post("/v1/generation")
async def generation(req: FooocusTaskInput, current_user=Depends(get_current_user)):
    try:
        req_dict = dict(req)
        if req.seed is None:
            req_dict["seed"] = random.randint(0, 2**63 - 1)
        task = muse_helper.task_queue.task_queue.add_task(dict(req))

        params_for_logger = dict(req)
        # extract image field which will be base64 string
        # we log it separately
        for key in ["inpaint_image", "mask_image", "uov_image"]:
            value = params_for_logger.pop(key, None)
            if value is not None:
                logger(
                    f"task {str(task.task_id)} {key}: {convert_base64_for_logger(value)}"
                )

        for idx, item in enumerate(params_for_logger.pop("control_images", [])):
            image = item.pop("image")
            logger(
                f"task {str(task.task_id)} control image ({idx}): {convert_base64_for_logger(image)}"
            )
            logger(f"task {str(task.task_id)} control image ({idx}): {item}")

        # logger remaining fields
        logger(f"task {str(task.task_id)} params: {params_for_logger}")

    except QueueFullException:
        return JSONResponse(
            status_code=429,
            content={"error": "Server is busy processing other requests"},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": e})

    return {"task_id": task.task_id}


@app.get("/v1/result/{task_id}")
def result(task_id: str):
    res = muse_helper.task_queue.task_queue.get_task_result(task_id)

    try:
        if res is not None:
            response_data = async_task_to_result_response(res.async_task)
            return {"task_id": res.task_id, **response_data}
        else:
            return {"message": "Task not found"}
    except Exception as e:
        e.with_traceback()
        return {"message": f"{e}"}


@app.get("/v1/preview/{task_id}")
def preview(task_id: str):
    res = muse_helper.task_queue.task_queue.get_task_result(task_id)

    try:
        if res is not None:
            response_data = async_task_to_preview_response(res.async_task)
            return {"task_id": res.task_id, **response_data}
        else:
            return {"message": "Task not found"}
    except Exception as e:
        e.with_traceback()
        return {"message": f"{e}"}
