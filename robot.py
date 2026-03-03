from typing import Tuple


class Robot:
    """Represents a robot on the Ricochet Robots board."""

    def __init__(self, x: int, y: int, color: Tuple[int, int, int], name: str):
        self.x = x
        self.y = y
        self.color = color
        self.name = name

    def __repr__(self) -> str:
        return f"Robot(name={self.name}, pos=({self.x}, {self.y}), color={self.color})"
