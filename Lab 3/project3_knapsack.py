"""
Project 3 - Dynamic Programming solution for the unbounded knapsack problem.

This script provides:
1. A recursive definition helper for P(C) documented in the report.
2. A bottom-up dynamic programming implementation that computes P(C) for a given
   capacity and item list where unlimited copies of each item are allowed.
3. Demonstration runs for the two datasets required by the assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Item:
    """Represents a single item type with weight and profit."""

    weight: int
    profit: int


def unbounded_knapsack(
    items: Sequence[Item], capacity: int
) -> Tuple[int, List[int], List[int]]:
    """
    Compute the maximum profit for the unbounded knapsack problem.

    Returns a tuple of:
    - The optimal profit P(capacity).
    - The DP table containing optimal profits for every capacity 0..capacity.
    - The predecessor array with the index of the item chosen for each capacity.
    """
    dp: List[int] = [0] * (capacity + 1)
    choice: List[int] = [-1] * (capacity + 1)

    for c in range(1, capacity + 1):
        best_profit = dp[c]
        best_item = choice[c]
        for idx, item in enumerate(items):
            if item.weight <= c:
                candidate = item.profit + dp[c - item.weight]
                if candidate > best_profit:
                    best_profit = candidate
                    best_item = idx
        dp[c] = best_profit
        choice[c] = best_item

    return dp[-1], dp, choice


def reconstruct_selection(items: Sequence[Item], capacity: int, choice: Sequence[int]) -> List[int]:
    """
    Reconstruct the multiplicity of each item in an optimal solution using the choice table.
    """
    counts = [0] * len(items)
    c = capacity

    while c > 0 and choice[c] != -1:
        idx = choice[c]
        counts[idx] += 1
        c -= items[idx].weight

    return counts


def demonstrate(dataset_name: str, items: Sequence[Item], capacity: int) -> None:
    """Run the solver for a dataset and print informative output."""
    optimal_profit, dp, choice = unbounded_knapsack(items, capacity)
    counts = reconstruct_selection(items, capacity, choice)

    print(f"Dataset: {dataset_name}")
    print(f"Capacity C = {capacity}")
    print("Items (index, weight, profit):")
    for idx, item in enumerate(items):
        print(f"  {idx}: weight={item.weight}, profit={item.profit}")
    print("\nDP table P(c) for c = 0..C:")
    print(" ".join(f"{p:3d}" for p in dp))
    print("\nItem choices leading to an optimal solution:")
    for idx, count in enumerate(counts):
        print(f"  Item {idx}: {count} time(s)")
    print(f"\nOptimal profit P({capacity}) = {optimal_profit}")
    print("-" * 60)


def main() -> None:
    capacity = 14
    dataset_a = [
        Item(weight=4, profit=7),
        Item(weight=6, profit=6),
        Item(weight=8, profit=9),
    ]
    dataset_b = [
        Item(weight=5, profit=7),
        Item(weight=6, profit=6),
        Item(weight=8, profit=9),
    ]

    demonstrate("Dataset (2): weights=[4, 6, 8], profits=[7, 6, 9]", dataset_a, capacity)
    demonstrate("Dataset (4b): weights=[5, 6, 8], profits=[7, 6, 9]", dataset_b, capacity)


if __name__ == "__main__":
    main()
