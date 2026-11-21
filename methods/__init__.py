# 文件作用：定义多Agent流程共用的角色数据结构与默认角色顺序。
# 主要职责：提供规划、批评、改进和最终裁决四个Agent的名称及角色配置。

from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    name: str
    role: str


def default_agents() -> List[Agent]:
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Critic", role="critic"),
        Agent(name="Refiner", role="refiner"),
        Agent(name="Judger", role="judger"),
    ]


__all__ = ["Agent", "default_agents"]
