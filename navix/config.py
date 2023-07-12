from typing import Any


class Config:
    def __init__(self):
        self.ARRAY_CHECKS_ENABLED = False

    def update(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def reset(self) -> None:
        self.__init__()


config = Config()
