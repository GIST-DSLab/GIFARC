# GIF Analysis Request

[Visual Analysis]
- Detailed scenario description: Describe the overall narrative of the GIF
- List all explicit objects (physical entities with clear boundaries)
- List all implicit objects (formed by groups of smaller objects creating recognizable patterns)

[Static patterns]
- Identify elements and relationships that remain consistent throughout the GIF
- Describe spatial arrangements, backgrounds, or design features that don't change
- List all constant patterns or objects

[Dynamic patterns:]
- Describe how elements interact and change over time
- Specify whether changes occur gradually, abruptly, or through multi-stage transformations
- Format interactions as: {"Objects involved": [], "Type of interaction": "", "Interaction parameters": []}

[Core Principles]
- Identify the general reasoning principles that explain how and why the dynamic elements change
- Clear analogy (explicit objects with defined parameters)
- Ambiguous analogy (implicit objects with indirect parameters)
- Constraint analogy (limiting behaviors or interactions)

*Only return the analysis in the JSON format. Do not include natural language descriptions or bullet lists.*.
{
  "scenario": "<detailed description of the overall narrative>",
  "objects": [
    {
      "name": "<object name>",
      "type": "explicit/implicit",
    }
  ],
  "static_patterns": [
    "<consistent elements throughout the GIF>"
  ],
  "dynamic_patterns": [
    "<changes and transformations over time>"
  ],
  "interactions": [
    {
      "objects_involved": ["<object1>", "<object2>"],
      "interaction_type": "clear/ambiguous/constraint/poetic",
      "interaction_parameters": ["<parameter1>", "<parameter2>"]
    }
  ],
  "core_principles": [
    "<general reasoning principles explaining transformations>"
  ],
}