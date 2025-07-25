When solving the given task, please output your result in the following JSON format:

{
  "analogy": "<Let's think step by step>",
  "step_by_step": "<Let's solve step by step>",
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

