from abc import ABC
from typing import Any, Optional


class Model(ABC):
    def simulate(
        self, t_max: int, parameters: dict[str, Any], seed: Optional[int], **kwargs
    ):
        pass
