You are a puzzle maker designing geometric, physical, and topological puzzles for curious middle-schoolers.

Each puzzle consists of uncovering a deterministic rule, pattern, procedure, algorithm, or transformation law that maps inputs to outputs.
Both the inputs and outputs are 2D grids of colored pixels. There are 10 colors, but the order of the colors is never relevant to the puzzle.

The middle schoolers are trying to discover this deterministic transformation, which can be implemented as a Python function called `main`.
Designing a puzzle involves also creating example inputs, which can be implemented as a Python function called `generate_input`. Unlike `main`, the `generate_input` function should be stochastic, so that every time you run it, you get another good example of what the transformation can be applied to.

Here is a overview of the puzzle you are designing:

{description}

Please implement the puzzle by writing code containing the `generate_input` and `main` functions. Use the following standard library (`common.py`):

```python
{common_lib}
```

Here are some examples from puzzles with similar descriptions to show you how to use functions in `common.py`:

{examples}

Your task is to implement the puzzle, following these steps:

1. Inspect the example puzzle implementations, making note of the functions used and the physical/geometric/topological/logical details
2. Inspect the new puzzle's description
3. Brainstorm a possible implementation for the new puzzle
4. Generate a code block formatted like the earlier examples with a comment starting `# concepts:` listing the concepts and `# description:` describing the inputs and transformation from the given description.
5. When writing code, please use variable names that are meaningfully related to the core concepts of the problem. For example, if the problem involves snow falling phenomena, use variable names like 'snowflake', 'precipitation', 'accumulation', 'gravity', 'obstacle', etc.
6. Specifically, when implementing the generate_input function and main function, make sure each variable name is directly associated with the concepts in the problem. For instance, use 'gravity_strength' for a variable representing the intensity of gravity, and 'obstacle_positions' for storing the locations of obstacles - choose names that clearly reveal the role and meaning of each variable.
7. Additionally, in the generate_input function, please restrict the grid size to be between 1x1 and 30x30. Do not create grids larger than 30x30. Implement the generate_input function that creates inputs appropriate for the problem and the main function that utilizes them while following these constraints.

Be sure to make the transformation `main` deterministic. Follow the description closely.
