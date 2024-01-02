from typing import List, Optional

from pydantic import BaseModel


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
