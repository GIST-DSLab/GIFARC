Please analyze the given GIF by extracting core reasoning elements.  
For each section below, provide your response as a list of clearly separated items (not long-form paragraphs).  

[scenario]
- Describe the complete narrative of the GIF in 1-3 concise sentences.
- Include initial state, key changes, and final state.
- Include only observable facts and avoid speculation.

[visual_elements]  
- Identify the key visual elements, objects, and actors in the GIF.  
- List the major visual components in the scene (e.g., objects, colors, spatial arrangement, shapes).

[objects]
- List all visual objects appearing in the GIF using this format:
    {
    "name": "object name",
    "type": "explicit" or "implicit"
    }
- explicit = physical objects with clear boundaries
- implicit = patterns or structures formed by multiple elements
- Identify 2-8 key objects.

[static_patterns]  
- Identify the elements or relationships that remain consistent throughout the GIF.  
- Describe repeating spatial arrangements, consistent backgrounds, or fixed design features.
- List all patterns or objects that remain constant throughout the GIF.

[dynamic_patterns]  
- Describe how the elements interact and change over time.  
- Consider whether the changes occur gradually (step-by-step), abruptly, or through multi-stage transformations (e.g., directional shifts, scaling, rotation).
- List the distinct changes or interactions that occur over time.

[core_principles]  
- Identify all general reasoning principles or mechanisms that explain **how and why** the dynamic elements change over time.  
- Each principle should be **generalizable beyond the current GIF** and help abstract the reasoning structure behind the transformation.  
- List one or more such principles that account for the observed dynamics.  
- After listing them, **summarize the single most fundamental principle in one concise sentence.**  
- Examples include: physical forces (e.g., gravity causes downward movement), goal‑oriented behaviors (movement toward a target), causal chains (one event triggers another), symmetry‑based transformations (reflection or alignment), and repetitive or cyclic patterns.

[interactions]
- List object interactions using this format:
    {
    "objects_involved": ["object1", "object2"],
    "interaction_type": "clear", "ambiguous", or "constraint",
    "interaction_parameters": ["parameter1", "parameter2"]
    }
- interaction_type definitions:
    -- clear: distinct physical interactions (collisions, contact, etc.)
    -- ambiguous: indirect or unclear interactions
    -- constraint: interactions that establish limitations or boundaries
- Record 2-6 key interactions, including only those directly observed in the GIF.