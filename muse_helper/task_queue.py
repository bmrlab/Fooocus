from collections import OrderedDict
from typing import Optional

from modules.async_worker import async_tasks
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


task_queue = TaskQueue(queue_size=3, history_size=6)
