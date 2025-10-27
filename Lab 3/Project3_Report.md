# Project 3 - Dynamic Programming (Unbounded Knapsack)

## Problem Statement
We are given a knapsack with capacity `C` and `n` types of objects. An unlimited number of objects of each type is available. The `i`-th object type has weight `w_i` and profit `p_i` (all positive integers). The goal is to select a multiset of objects whose total weight does not exceed `C` while maximising the total profit. Let `P(C)` denote the optimal profit obtainable with capacity `C`.

The datasets required by the assignment are:

- Dataset (2): `C = 14`, weights `[4, 6, 8]`, profits `[7, 6, 9]`.
- Dataset (4b): `C = 14`, weights `[5, 6, 8]`, profits `[7, 6, 9]`.

## 1. Recursive Definition of `P(C)`

Let `W = {w_0, ..., w_{n-1}}` and `P = {p_0, ..., p_{n-1}}`. The recurrence for an unbounded knapsack is

```
P(0) = 0
P(C) = max_{i : w_i <= C} (p_i + P(C - w_i))      for C > 0
```

If no weight fits into `C`, the set of feasible indices is empty and `P(C) = 0`. This recurrence captures the optimal substructure: to achieve the best profit for capacity `C`, select the item whose inclusion leaves the subproblem with capacity `C - w_i`.

## 2. Subproblem Graph for `P(14)` - Dataset (2)

The subproblem graph is a directed acyclic graph whose vertices represent capacities and whose edges point from `C` to `C - w_i` for every feasible `i`. Shared subproblems appear once in the graph even if they arise from multiple paths. For `C = 14`, weights `[4, 6, 8]`, profits `[7, 6, 9]`, the graph is:

```
P(14)
|- P(10)
|  |- P(6)
|  |  |- P(2)
|  |  \- P(0)
|  |- P(4)
|  |  \- P(0)
|  \- P(2)
|- P(8)
|  |- P(4)
|  |  \- P(0)
|  |- P(2)
|  \- P(0)
\- P(6)
   |- P(2)
   \- P(0)
```

- Each edge `P(C) -> P(C - w_i)` corresponds to choosing item `i`.
- Subproblems `P(6)` and `P(4)` are reused multiple times; memoisation or bottom-up DP stores their solutions once.
- Nodes `P(2)` and `P(0)` are terminal. `P(2)` has no outgoing edges because no item fits in capacity 2, so its optimal profit is 0.

## 3. Bottom-Up Dynamic Programming Algorithm

We process capacities from 0 up to `C`, computing the optimal profit once per capacity. For each capacity we iterate over all items, relax the choice `p_i + P(C - w_i)`, and store the best profit. A `choice` array remembers which item achieved the optimum for reconstruction.

```python
def unbounded_knapsack(items, C):
    dp = [0] * (C + 1)
    choice = [-1] * (C + 1)
    for c in range(1, C + 1):
        for i, item in enumerate(items):
            if item.weight <= c:
                candidate = item.profit + dp[c - item.weight]
                if candidate > dp[c]:
                    dp[c] = candidate
                    choice[c] = i
    return dp, choice
```

- Time complexity: `O(n * C)` because each capacity considers all `n` item types once.
- Space complexity: `O(C)` for the DP and choice arrays. The multiplicities of each item can be reconstructed by walking backwards from capacity `C` using `choice`.

## 4. Implementation and Results

The accompanying script `Lab 3/project3_knapsack.py` implements the algorithm above. Running it (with Python 3.11+) prints full DP tables, chosen items, and optimal profits for both datasets:

```
python project3_knapsack.py
```

### Dataset (2): weights = [4, 6, 8], profits = [7, 6, 9]
- DP table `P(c)` for `c = 0..14`: `0 0 0 0 7 7 7 7 14 14 14 14 21 21 21`
- Optimal profit `P(14) = 21`
- Item usage: three copies of weight 4 / profit 7 item (total weight 12) plus the remaining capacity 2 unused because no item fits.

### Dataset (4b): weights = [5, 6, 8], profits = [7, 6, 9]
- DP table `P(c)` for `c = 0..14`: `0 0 0 0 0 7 7 7 9 9 14 14 14 16 16`
- Optimal profit `P(14) = 16`
- Item usage: one copy each of `(weight=5, profit=7)` and `(weight=8, profit=9)` for total weight 13 and profit 16; the remaining capacity cannot be filled profitably.

## 5. How to Reproduce
1. Ensure Python 3.11+ is available.
2. Navigate to `Lab 3`.
3. Run `python project3_knapsack.py` to reproduce the outputs reported above.
4. Modify the `items` lists or `capacity` inside `main()` to experiment with other datasets.

The report, recurrence, graph explanation, and code together satisfy all requirements stated in Project 3.
