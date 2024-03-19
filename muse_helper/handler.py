import base64
from io import BytesIO

import numpy as np
from PIL import Image

import modules.async_worker as worker
import modules.flags as flags
from muse_helper.api_model import FooocusTaskInput
from muse_helper.task import Task


def load_base64(base64_string: str):
    base64_bytes = base64_string.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)
    image = Image.open(BytesIO(image_bytes))

    return np.array(image)
 

def handler(input_task: Task):
    import modules.config

    req = FooocusTaskInput(**input_task.params)

    # preprocess lora args
    lora_args_list = []

    has_default_lora = False
    for item in req.lora_models:
        if item["model"] == "sd_xl_offset_example-lora_1.0.safetensors":
            has_default_lora = True
            break

    if len(req.lora_models) < modules.config.default_max_lora_number and not has_default_lora:
        # add basic offset lora
        lora_args_list.append(True)
        lora_args_list.append("sd_xl_offset_example-lora_1.0.safetensors")
        lora_args_list.append(0.5)
    for item in req.lora_models:
        lora_args_list.append(True)
        lora_args_list.append(item["model"])
        lora_args_list.append(item["weight"])
    while len(lora_args_list) < 3 * modules.config.default_max_lora_number:
        lora_args_list.append(False)
        lora_args_list.append("None")
        lora_args_list.append(0.1)

    # preprocess control net args
    control_net_args_list = []
    for item in req.control_images:
        control_net_args_list.append(load_base64(item["image"]))
        control_net_args_list.append(item["stop_at"])
        control_net_args_list.append(item["weight"])
        control_net_args_list.append(item["mode"])
    while len(control_net_args_list) < flags.controlnet_image_count * 4:
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
        False, # generate_image_grid
        req.prompt, # prompt
        req.negative_prompt, # negative_prompt
        style,  # style_selections
        req.mode, # performance_selection
        f"{req.width}×{req.height}",  # aspect_ratios_selection
        req.batch_size,  # image_number
        "png",  # output_format
        req.seed,  # image_seed
        True, # read_wildcards_in_order
        req.sharpness,  # sharpness
        req.guidance_scale,  # guidance_scale
        req.model, # base_model_name
        req.refiner, # refiner_model_name
        req.refiner_switch, # refiern_switch
        *lora_args_list, # loras
        req.enable_input_image,  # input_image_checkbox
        req.current_tab,  # current_tab
        req.uov_mode,  # uov_method
        load_base64(req.uov_image) if req.uov_image is not None else None, # uov_input_image
        req.outpaint_mode, # outpaint_seletions
        {
            "image": load_base64(req.inpaint_image),
            "mask": load_base64(req.mask_image)
            if req.mask_image is not None
            else np.zeros_like(load_base64(req.inpaint_image)),
        }
        if req.inpaint_image is not None
        else None, # inpaint_input_image
        req.inpaint_additional_prompt, # inpaint_additional_prompt
        req.inpaint_mask_image_upload, # inpaint_mask_image_upload
        False, # disable_preview
        False, # disable_intermediate_results
        False, # disable_seed_increment
        1.5, # adm_scaler_positive
        0.8, # adm_scaler_negative
        0.3, # adm_scaler_end
        7, # adaptive_cfg
        req.sampler, # sampler_name
        req.scheduler, # scheduler_name
        -1, # overwrite_step
        -1, # overwrite_switch
        -1, # overwrite_width
        -1, # overwrite_height
        -1, # overwrite_vary_strength
        -1, # overwrite_upscale_strength
        True, # mixing_image_prompt_and_vary_upscale
        True, # mixing_image_prompt_and_inpaint
        False, # debugging_cn_preprocessor
        False, # skipping_cn_preprocessor
        req.canny_low_threshold, # canny_low_threshold
        req.canny_high_threshold, # canny_high_threshold
        req.refiner_swap_method, # refiner_swap_method
        req.controlnet_softness, # controlnet_softness
        req.enable_free_u, # freeu_enabled
        req.free_u_b1, # freeu_b1
        req.free_u_b2, # freeu_b2
        req.free_u_s1, # freeu_s1
        req.free_u_s2, # freeu_s2
        False, # debugging_inpaint_preprocessor
        req.inpaint_disable_initial_latent, # inpaint_disable_initial_latent
        req.inpaint_engine, # inpaint_engine
        req.inpaint_strength, # inpaint_strength
        req.inpaint_respective_field, # inpaint_respective_field
        False, # inpaint_mask_upload_checkbox
        False, # invert_mask_checkbox
        req.inpaint_erode_or_dilate, # inpaint_erode_or_dilate
        False, # save_metadata_to_images
        "fooocus", # metadata_schema ,
        *control_net_args_list,  # cn_tasks
    )

    task = worker.AsyncTask(args=list(worker_args))
    input_task.async_task = task
    worker.async_tasks.append(task)
