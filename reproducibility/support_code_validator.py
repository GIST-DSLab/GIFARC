"""Actual structure and finite seeded-behavior validator definitions.

No model-backed neutralization function is included. This executes only the
three inspected support code pairs supplied with the supplement.
"""
from __future__ import annotations
import ast
import random
import sys
import types
from pathlib import Path
from typing import Any
import numpy as np

def validate_code_structure(original_code: str, neutralized_code: str) -> tuple[bool, str | None]:
    try:
        original_tree = ast.parse(original_code)
        candidate_tree = ast.parse(neutralized_code)
    except SyntaxError as exc:
        return False, f"neutralized_code_syntax_error: {exc}"

    if _top_level_imports(original_tree) != _top_level_imports(candidate_tree):
        return False, "neutralized_code_changed_imports"
    if _top_level_assignments(original_tree) != _top_level_assignments(candidate_tree):
        return False, "neutralized_code_changed_top_level_assignments"
    if _top_level_function_signatures(original_tree) != _top_level_function_signatures(candidate_tree):
        return False, "neutralized_code_changed_top_level_function_names_or_signatures"
    return True, None


def validate_behavior(
    *,
    original_code: str,
    neutralized_code: str,
    common_module_dir: str | Path,
    task_id: int,
) -> tuple[bool, str | None]:
    try:
        original_module = _load_module_from_source(
            original_code, f"original_task_{task_id}", common_module_dir
        )
        neutral_module = _load_module_from_source(
            neutralized_code, f"neutral_task_{task_id}", common_module_dir
        )
    except Exception as exc:
        return False, f"neutralized_code_import_error: {exc}"

    if not hasattr(original_module, "generate_input") or not hasattr(original_module, "main"):
        return False, "original_code_missing_required_entrypoints"
    if not hasattr(neutral_module, "generate_input") or not hasattr(neutral_module, "main"):
        return False, "neutralized_code_missing_required_entrypoints"

    for seed in [0, 1, 2]:
        try:
            random.seed(seed)
            np.random.seed(seed)
            original_input = np.array(original_module.generate_input())
        except Exception as exc:
            return False, f"original_code_generate_input_error_seed_{seed}: {exc}"

        try:
            random.seed(seed)
            np.random.seed(seed)
            neutral_input = np.array(neutral_module.generate_input())
        except Exception as exc:
            return False, f"neutralized_code_generate_input_error_seed_{seed}: {exc}"

        if not np.array_equal(original_input, neutral_input):
            return False, f"neutralized_code_generate_input_mismatch_seed_{seed}"

        try:
            original_output = np.array(original_module.main(original_input.copy()))
        except Exception as exc:
            return False, f"original_code_main_error_seed_{seed}: {exc}"
        try:
            neutral_output = np.array(neutral_module.main(neutral_input.copy()))
        except Exception as exc:
            return False, f"neutralized_code_main_error_seed_{seed}: {exc}"
        if not np.array_equal(original_output, neutral_output):
            return False, f"neutralized_code_output_mismatch_seed_{seed}"

    return True, None


def _top_level_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.dump(node, include_attributes=False))
    return imports


def _top_level_assignments(tree: ast.Module) -> list[str]:
    targets: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
    return targets


def _top_level_function_signatures(tree: ast.Module) -> list[tuple[str, int]]:
    signatures: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            signatures.append((node.name, len(node.args.args)))
    return signatures


def _load_module_from_source(source_code: str, module_name: str, common_module_dir: str | Path) -> Any:
    common_path = str(Path(common_module_dir))
    if common_path not in sys.path:
        sys.path.insert(0, common_path)
    module = types.ModuleType(module_name)
    module.__file__ = f"<{module_name}>"
    exec(compile(source_code, module.__file__, "exec"), module.__dict__)
    return module
