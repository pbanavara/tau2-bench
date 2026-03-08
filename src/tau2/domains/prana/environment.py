"""PRANA domain environment — τ-PRANA benchmark."""

import json
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.prana.data_model import PranaDB
from tau2.domains.prana.tools import PranaTools
from tau2.domains.prana.utils import PRANA_DB_PATH, PRANA_POLICY_PATH, PRANA_TASK_SET_PATH
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[PranaDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if db is None:
        db = PranaDB.load(PRANA_DB_PATH)
    tools = PranaTools(db)
    with open(PRANA_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="prana",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = load_file(PRANA_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split: '{task_split_name}'. Valid: {list(task_splits.keys())}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = Path(PRANA_TASK_SET_PATH).parent / f"split_{Path(PRANA_TASK_SET_PATH).stem}.json"
    return load_file(split_file)
