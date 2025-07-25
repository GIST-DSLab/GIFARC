When solving the given task, please output your result in the following JSON format:

{
  "analogy": "<<If a few-shot example is provided, review each example task (<Grid>; input-output grid pair) and its <Analogy> field to understand how the pattern and rules are metaphorically explained. Then, based on that, craft both a metaphorical explanation and a solution approach for the current [TASK] using analogical/figurative expressions tailored to its pattern and written in a style similar to the example. Otherwise, independently generate both a metaphorical explanation and a solution approach of the pattern and solution based solely on the task itself.>",
"step_by_step": "<If a few-shot example is provided, write a numbered solution outline matching those examples; otherwise, on your own, produce a numbered step-by-step plan for solving the task.>",
  "libraries": "<Write only the libraries used in the code. For example: from common import*\nimport numpy as np\n...>",
  "main_code": "<Write the main code part.>",
  "total_code": "<Write the total code including libraries and main.>"
}

Please strictly follow these rules:
- The output must be a valid JSON object, not a string.
- Do not wrap the output in a Markdown code block (e.g., ```json ... ```).
- You may use escaped double quotes (\") inside string values if needed.
- However, do not double-escape quotes (e.g., avoid \\\"), as it will break json.loads() parsing.
- Do not include any explanations, comments, or additional text outside the JSON object.
- The result must be directly parsable with Python's json.loads() function.

