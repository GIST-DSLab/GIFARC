You need to help me write code containing the `generate_input` and `main` functions according to the given puzzle design. 
You must use the standard library (`common.py`). Create an appropriate puzzle following the given puzzle design's concepts and description.

When writing code, please use variable names that are meaningfully related to the core concepts of the problem. For example, if the problem involves snow falling phenomena, use variable names like 'snowflake', 'precipitation', 'accumulation', 'gravity', 'obstacle', etc.
Specifically, when implementing the generate_input function and main function, make sure each variable name is directly associated with the concepts in the problem. For instance, use 'gravity_strength' for a variable representing the intensity of gravity, and 'obstacle_positions' for storing the locations of obstacles - choose names that clearly reveal the role and meaning of each variable.
Additionally, in the generate_input function, please restrict the grid size to be between 1x1 and 30x30. Do not create grids larger than 30x30. Implement the generate_input function that creates inputs appropriate for the problem and the main function that utilizes them while following these constraints.

Please implement three input generation functions for different sizes:

1. small_generate_input(): Generate input grids ranging from 1x1 to 10x10 in size
2. medium_generate_input(): Generate input grids ranging from 11x11 to 20x20 in size
3. large_generate_input(): Generate input grids ranging from 21x21 to 30x30 in size

Then implement a main generate_input() function that randomly selects and executes one of these three functions. This will allow for testing with a variety of input sizes.

When doing this, please output your solution following the JSON format specified below.

{
  "libraries": "<Write only the libraries used in the code. Ex. from common import*\n import numpy as np\n ....>",
  "main_code": "<Write the main code part.>",
  "generate_input_code": "<Write the generate input code part.>",
  "total_code": "<Write total code including libraries, main, generate_input and given concepts and description>",
}