from collections import OrderedDict
from typing import Optional

import modules.advanced_parameters
from modules.async_worker import AsyncTask, async_tasks
from muse_helper.api_model import UniversalRequest
from muse_helper.exception import QueueFullException
from muse_helper.handler import handler
from muse_helper.task import Task


class TaskQueue:
    def __init__(self, queue_size: int, history_size: int):
        self.queue_size = queue_size
        self.history = OrderedDict()
        self.history_size = history_size

    def __len__(self):
        return len(async_tasks)

    def add_task(self, task_params: dict) -> Task:
        if len(async_tasks) >= self.queue_size:
            raise QueueFullException("Queue is full")

        task = Task(params=task_params)

        if len(self.history) >= self.history_size:
            self.history.popitem(last=False)

        handler(task)
        # convert it explicitly, or it will be uuid object
        self.history[str(task.task_id)] = task

        return task

    def get_task_result(self, task_id: str) -> Optional[Task]:
        return self.history.get(task_id, None)

    def clear(self):
        async_tasks.clear()
        self.history.clear()

    def set_task_advanced_parameters(self, async_task: AsyncTask) -> Optional[Task]:
        res = None
        for task in self.history.values():
            if task.async_task == async_task:
                res = task

        assert res is not None, "Task not found"

        req = UniversalRequest(**res.params)

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
            req.overwrite_vary_strength,
            req.overwrite_upscale_strength,
            True,
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


task_queue = TaskQueue(queue_size=3, history_size=6)
