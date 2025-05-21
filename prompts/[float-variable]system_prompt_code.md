You need to help me write code containing the `generate_input` and `main` functions according to the given puzzle design. 
You must use the standard library (`common.py`). Create an appropriate puzzle following the given puzzle design's concepts and description.

When writing code, please use variable names that are meaningfully related to the core concepts of the problem. For example, if the problem involves snow falling phenomena, use variable names like 'snowflake', 'precipitation', 'accumulation', 'gravity', 'obstacle', etc.
Specifically, when implementing the generate_input function and main function, make sure each variable name is directly associated with the concepts in the problem. For instance, use 'gravity_strength' for a variable representing the intensity of gravity, and 'obstacle_positions' for storing the locations of obstacles - choose names that clearly reveal the role and meaning of each variable.
Additionally, in the generate_input function, please restrict the grid size to be between 1x1 and 30x30. Do not create grids larger than 30x30. Implement the generate_input function that creates inputs appropriate for the problem and the main function that utilizes them while following these constraints.

When implementing code, please avoid using float type variables, numbers with decimal points, and the math library. These elements make it difficult to intuitively identify patterns between input and output grids. Instead, use only integer operations and basic arithmetic operators to clearly reveal the essence of the pattern. Please strictly follow this constraint when implementing your code.

Also, When implementing code, please avoid creating and using too many variables. In particular, don't excessively define variables used as magic numbers or thresholds. Having numerous variables increases code complexity, making it difficult for people to intuitively understand the core rules of the puzzle. Please write concise code using only essential variables so that the puzzle's pattern is clearly revealed.

When doing this, please output your solution following the JSON format specified below.

{
  "libraries": "<Write only the libraries used in the code. Ex. from common import*\n import numpy as np\n ....>",
  "main_code": "<Write the main code part.>",
  "generate_input_code": "<Write the generate input code part.>",
  "total_code": "<Write total code including libraries, main, generate_input and given concepts and description>",
}