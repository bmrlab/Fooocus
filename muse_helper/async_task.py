import base64
import io
from typing import Dict

from PIL import Image

from modules.async_worker import AsyncTask


def np_to_base64(image_np):
    image = Image.fromarray(image_np)
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        b64_string = base64.b64encode(buffer.getvalue())

    b64_string = b64_string.decode("utf-8")
    return b64_string


def async_task_to_response(async_task: AsyncTask) -> Dict:
    yields = async_task.yields
    previews = []
    is_finished = False
    results = []

    preview_parts = [[]]
    for item in yields:
        if item[0] == "preview":
            _, _, image = item[1]
            if image is not None:
                preview_parts[-1].append(item[1])
        elif item[0] == "results":
            preview_parts.append([])
        elif item[0] == "finish":
            is_finished = True
        else:
            pass

    for part in preview_parts:
        if len(part) > 0:
            previews.append(np_to_base64(part[-1][-1]))

    if is_finished:
        results = [np_to_base64(item) for item in async_task.results]

    return {"previews": previews, "finished": is_finished, "results": results}
