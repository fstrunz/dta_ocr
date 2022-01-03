from dataclasses import dataclass
from typing import List
from page.elements import Point, Coordinates

@dataclass(frozen=True, eq=True)
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_points(self) -> List[Point]:
        return [
            Point(self.xmin, self.ymin),
            Point(self.xmax, self.ymin),
            Point(self.xmax, self.ymax),
            Point(self.xmin, self.ymax)
        ]

    def to_coords(self) -> Coordinates:
        return Coordinates(self.to_points())
