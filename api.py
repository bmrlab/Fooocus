import os
import sys

import uvicorn

from modules import config # this must be imported
import modules.async_worker  # this must be imported
from args_manager import args  # this must be imported
from muse_helper.router import app  # this must be imported


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root)
    os.chdir(root)
    backend_path = os.path.join(root, "backend", "headless")
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    port = os.environ.get("PORT", "7860")

    uvicorn.run(
        "api:app", host="0.0.0.0", port=int(port), log_level="critical", workers=1
    )
