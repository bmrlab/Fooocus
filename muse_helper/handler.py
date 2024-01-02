import base64
from io import BytesIO

import numpy as np
from PIL import Image

import modules.async_worker as worker
from muse_helper.api_model import UniversalRequest
from muse_helper.task import Task


def load_base64(base64_string: str):
    base64_bytes = base64_string.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)
    image = Image.open(BytesIO(image_bytes))

    return np.array(image)


def handler(input_task: Task):
    req = UniversalRequest(**input_task.params)

    # preprocess lora args
    lora_args_list = []

    has_default_lora = False
    for item in req.lora_models:
        if item["model"] == "sd_xl_offset_example-lora_1.0.safetensors":
            has_default_lora = True
            break

    if len(req.lora_models) < 5 and not has_default_lora:
        # add basic offset lora
        lora_args_list.append("sd_xl_offset_example-lora_1.0.safetensors")
        lora_args_list.append(0.5)
    for item in req.lora_models:
        lora_args_list.append(item["model"])
        lora_args_list.append(item["weight"])
    while len(lora_args_list) < 10:
        lora_args_list.append("None")
        lora_args_list.append(1)

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
    input_task.async_task = task
    worker.async_tasks.append(task)
