import uuid
from typing import Optional

from attrs import define, field

from modules.async_worker import AsyncTask


@define
class Task:
    task_id: str = field(factory=uuid.uuid4)
    params: dict = field(default={})
    async_task: Optional[AsyncTask] = field(default=None)
