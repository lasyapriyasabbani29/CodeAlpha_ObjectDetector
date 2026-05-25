from __future__ import annotations


PALETTE = [
    (25, 211, 255),
    (56, 226, 138),
    (255, 153, 51),
    (255, 82, 119),
    (173, 124, 255),
    (70, 183, 255),
    (232, 224, 96),
    (68, 240, 210),
    (255, 112, 67),
    (151, 203, 83),
    (236, 92, 189),
    (118, 154, 255),
]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]

