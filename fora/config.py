from dataclasses import dataclass


@dataclass(frozen=True)
class FORAConfig:
    """Refresh cached features every `interval` steps, starting at step zero."""

    interval: int = 3

    def __post_init__(self) -> None:
        if type(self.interval) is not int or self.interval < 1:
            raise ValueError("interval must be a positive integer")
