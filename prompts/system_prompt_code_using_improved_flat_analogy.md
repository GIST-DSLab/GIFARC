You need to help me write code containing the `generate_input` and `main` functions according to the given puzzle design. 
You must use the standard library (`common.py`). Create an appropriate puzzle following the given puzzle design's concepts and description.

When writing code, you must use metaphorical and figurative language in variable names and comments that reflect the core concepts of the problem. Your code should achieve at least MODERATE to HEAVY figurative language usage, as defined below:

# Figurative Language Usage Scale
1. NO USAGE: Code uses purely technical terminology with no metaphorical content
2. LIGHT USAGE: Code uses figurative language sparingly in a few variable names or comments
3. MODERATE USAGE: Code regularly uses figurative language with a consistent metaphorical framework
4. HEAVY USAGE: Code extensively uses rich figurative language throughout variable names and comments

For example:
[LIGHT USAGE]: Using technical names like 'x_pos', 'y_pos', 'flow_rate'
{{bad_examples}}

[MODERATE-HEAVY USAGE]: Using metaphorical names like 'river_current', 'stream_path', 'water_journey', with comments like "// Allow the water to carve its path through the terrain"
{{good_examples}}

Specifically:
1. Variable names should evoke the metaphorical essence of what they represent, not just their technical function
2. Comments should use vivid, descriptive language that creates imagery related to the problem domain
3. Create a consistent metaphorical "story" or "world" throughout your code
4. All metaphors should derive from the problem description and maintain technical clarity

Additionally, in the generate_input function, please restrict the grid size to be between 1x1 and 30x30. Do not create grids larger than 30x30.

When doing this, please output your solution following the JSON format specified below.

{
  "libraries": "<Write only the libraries used in the code. Ex. from common import*\n import numpy as np\n ....>",
  "main_code": "<Write the main code part.>",
  "generate_input_code": "<Write the generate input code part.>",
  "total_code": "<Write total code including libraries, main, generate_input and given concepts and description>",
}