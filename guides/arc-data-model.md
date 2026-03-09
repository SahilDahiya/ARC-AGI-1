# ARC Data Model Guide

Last updated: 2026-03-08

## Purpose

This guide explains the core ARC data types used in the codebase and how they map to the JSON task files.

## Type Aliases

The aliases are defined in `src/arc_agi_1/data.py`:

```python
Grid: TypeAlias = list[list[int]]
Pair: TypeAlias = dict[str, Grid]
Task: TypeAlias = dict[str, list[Pair]]
TaskMap: TypeAlias = dict[str, Task]
```

These do not create new runtime objects. They make the code easier to read and give static type checkers useful structure.

## What Each Type Means

`Grid`
- A 2D matrix of integers.
- Each integer is an ARC cell color in the range `0..9`.
- Example:

```python
[[0, 7, 7],
 [7, 7, 7],
 [0, 7, 7]]
```

`Pair`
- One input/output example.
- Contains two keys:
  - `"input"`
  - `"output"`
- Both values are `Grid`s.
- Example:

```python
{
    "input": [[0, 7], [7, 7]],
    "output": [[7, 0], [0, 7]],
}
```

`Task`
- One full ARC problem.
- Usually loaded from one JSON file.
- Contains:
  - `"train"`: list of demonstration pairs
  - `"test"`: list of query pairs
- Example:

```python
{
    "train": [pair1, pair2, pair3],
    "test": [pair4],
}
```

`TaskMap`
- A dictionary of many tasks keyed by task id.
- Example:

```python
{
    "007bbfb7": task_a,
    "00d62c1b": task_b,
}
```

## Relationship Between Types

The nesting is:

`TaskMap` -> many `Task`s  
`Task` -> lists of `Pair`s  
`Pair` -> `"input"` and `"output"` `Grid`s  
`Grid` -> 2D integer cells

## How This Maps To Files

In the dataset:
- one JSON file = one `Task`
- `data/training/*.json` and `data/evaluation/*.json` are collections of tasks

When we load a split in `load_split(...)`, we get a `TaskMap`:

```python
tasks = {
    "007bbfb7": {...},
    "00d62c1b": {...},
}
```

Each value in that dictionary is a `Task`.

## Why This Matters

These names show what level of structure a function expects:

- if a function takes `Grid`, it works on one grid
- if it takes `Pair`, it works on one input/output example
- if it takes `Task`, it works on one ARC problem
- if it takes `TaskMap`, it works on a whole split or set of tasks

That distinction matters for:
- scoring
- dataset construction
- task-conditioned modeling
- test-time training loops
