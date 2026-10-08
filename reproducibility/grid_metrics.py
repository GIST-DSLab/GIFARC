"""Actual exact, GC and FG definitions from the solving evaluation."""
from __future__ import annotations
from typing import Any

def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def exact_match(predicted: list[list[int]], gold: list[list[int]]) -> int:
    return int(predicted == gold)


def grid_cell_accuracy(predicted: list[list[int]], gold: list[list[int]]) -> float:
    if grid_shape(predicted) != grid_shape(gold):
        return 0.0
    total = 0
    correct = 0
    for pred_row, gold_row in zip(predicted, gold):
        for pred_value, gold_value in zip(pred_row, gold_row):
            total += 1
            if pred_value == gold_value:
                correct += 1
    return correct / total if total else 0.0


def foreground_accuracy(
    predicted: list[list[int]],
    gold: list[list[int]],
    input_grid: list[list[int]],
) -> tuple[float, dict[str, Any]]:
    if grid_shape(predicted) != grid_shape(gold):
        return 0.0, {"foreground_mode": "shape_mismatch", "foreground_empty": False}

    if grid_shape(gold) == grid_shape(input_grid):
        foreground_cells = [
            (row_idx, col_idx)
            for row_idx, (gold_row, input_row) in enumerate(zip(gold, input_grid))
            for col_idx, (gold_value, input_value) in enumerate(zip(gold_row, input_row))
            if gold_value != input_value
        ]
        foreground_mode = "input_aligned_difference"
    else:
        foreground_cells = [
            (row_idx, col_idx)
            for row_idx, gold_row in enumerate(gold)
            for col_idx, _ in enumerate(gold_row)
        ]
        foreground_mode = "all_gold_cells_due_to_shape_change"

    if not foreground_cells:
        return 1.0, {"foreground_mode": foreground_mode, "foreground_empty": True}

    correct = 0
    for row_idx, col_idx in foreground_cells:
        if predicted[row_idx][col_idx] == gold[row_idx][col_idx]:
            correct += 1
    return correct / len(foreground_cells), {
        "foreground_mode": foreground_mode,
        "foreground_empty": False,
    }
