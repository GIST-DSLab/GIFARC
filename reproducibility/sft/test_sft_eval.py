"""CPU-only regressions for evaluation coverage, scoring, and safe resumption.

Run with: python -B -m unittest discover -s reproducibility/sft -p test_sft_eval.py
No model, CUDA, network, or generated solution code is executed.
"""

import ast
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
RESPONSE = "# RESULT TEST OUTPUT GRID\nThe output grid for the test input grid is:\n```\nRed\n```"


def pure_definitions(path, function_names, constant_names=()):
    """Read the real pure helpers without importing training/model libraries."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constant_names for target in node.targets
        ):
            nodes.append(node)
    module = types.ModuleType(path.stem)
    module.__dict__["re"] = re
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), module.__dict__)
    return module


class TextTensor(str):
    def to(self, device):
        if device != "cpu":
            raise AssertionError("The regression suite must stay on CPU")
        return self


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, prompt, **kwargs):
        return {"input_ids": TextTensor(prompt)}

    def decode(self, output, **kwargs):
        return str(output)


class Model:
    def __init__(self, fail_on=None, response=RESPONSE):
        self.calls = []
        self.fail_on = fail_on
        self.response = response

    def eval(self):
        return self

    def parameters(self):
        return iter([types.SimpleNamespace(device="cpu")])

    def generate(self, **kwargs):
        self.calls.append(str(kwargs["input_ids"]))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("synthetic generation failure")
        return [self.calls[-1] + self.response]


class EvaluationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gifarc-sft-eval-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input_path = self.directory / "input.jsonl"
        self.target_path = self.directory / "target.jsonl"
        self.output_path = self.directory / "output.jsonl"
        self.scoring_calls = []
        train = pure_definitions(HERE / "sft_train.py", {"create_input_prompt", "create_input_only_format"})
        arc = pure_definitions(
            HERE / "arc_evaluation.py",
            {"parse_grid_from_text", "parse_bare_grid", "extract_test_output_grid", "compare_grids", "compare_grids_foreground", "evaluate_transduction", "evaluate_arc_output"},
            {"COLOR_MAPPING", "REVERSED_MAPPING", "COLOR_REPLACEMENTS"},
        )
        real_evaluate = arc.evaluate_arc_output

        def score(**kwargs):
            self.scoring_calls.append(kwargs)
            if kwargs.get("evaluation_mode") != "transduction":
                raise AssertionError("These tests must never execute generated programs")
            # Use the actual parser and exact grid comparison; avoid optional NLP/model work.
            kwargs.update(model=None, tokenizer=None, device=None)
            return real_evaluate(**kwargs)

        arc.evaluate_arc_output = score
        torch = types.ModuleType("torch")
        torch.no_grad = nullcontext
        torch.manual_seed = lambda seed: None
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        transformers = types.ModuleType("transformers")
        for name in ("AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig"):
            setattr(transformers, name, object())
        peft = types.ModuleType("peft")
        peft.PeftModel = object()
        progress = types.ModuleType("tqdm")
        progress.tqdm = lambda iterable, **kwargs: iterable
        progress.tqdm.write = lambda text: None
        numpy = types.ModuleType("numpy")
        numpy.random = types.SimpleNamespace(seed=lambda seed: None)
        spec = importlib.util.spec_from_file_location("_sft_eval_under_test", HERE / "sft_eval.py")
        self.evaluator = importlib.util.module_from_spec(spec)
        module_patch = patch.dict(sys.modules, {
            "torch": torch, "transformers": transformers, "peft": peft,
            "tqdm": progress, "sft_train": train, "arc_evaluation": arc,
            "numpy": numpy, "_sft_eval_under_test": self.evaluator,
        })
        module_patch.start()
        self.addCleanup(module_patch.stop)
        spec.loader.exec_module(self.evaluator)
        self.model = Model()
        self.tokenizer = Tokenizer()
        self.console = io.StringIO()

    def messages_row(self, problem_id="chat", test_index=0):
        return {
            "problem_id": problem_id,
            "test_index": test_index,
            "messages": [
                {"role": "system", "content": "Solve the puzzle."},
                {"role": "user", "content": "Here is the input grid for the test example:\nInput:\nBlue\n\n"},
            ],
        }

    def raw_row(self, problem_id="raw", test_index=0):
        return {
            "problem_id": problem_id,
            "test_index": test_index,
            "source": "A puzzle transformation",
            "examples": [[[[1]], [[2]]]],
            "ground_truth_output": [[2]],
        }

    def target_row(self, problem_id="chat", test_index=0):
        return {
            "problem_id": problem_id,
            "test_index": test_index,
            "ground_truth_output": [[2]],
            "answer_messages": [{"role": "assistant", "content": RESPONSE}],
        }

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def prepare(self, inputs, targets=None):
        self.write_rows(self.input_path, inputs)
        if targets is None:
            targets = [self.target_row(row.get("problem_id"), row.get("test_index")) for row in inputs]
        self.write_rows(self.target_path, targets)

    def run_evaluation(self, temperature=0):
        with redirect_stdout(self.console):
            return self.evaluator.generate_evaluation_outputs(
                model_name="test-model", model=self.model, tokenizer=self.tokenizer,
                input_data_path=str(self.input_path), ground_truth_data_path=str(self.target_path),
                output_path=str(self.output_path), max_length=1024, max_new_tokens=128,
                evaluation_mode="transduction", model_path="test-adapter", temperature=temperature,
            )

    def result_rows(self):
        if not self.output_path.exists():
            return []
        return [json.loads(line) for line in self.output_path.read_text(encoding="utf-8").splitlines()]

    def test_mixed_messages_and_raw_rows_generate_and_score_exactly_once(self):
        self.prepare([self.messages_row(), self.raw_row()])
        self.run_evaluation()
        rows = self.result_rows()
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(len(self.scoring_calls), 2)
        self.assertCountEqual([(row["problem_id"], row["test_index"]) for row in rows], [("chat", 0), ("raw", 0)])
        self.assertEqual([row.get("transduction_result") for row in rows], [1, 1])

    def test_transduction_mode_scores_correct_grid_without_solution_code_or_direct_phrase(self):
        self.prepare([self.messages_row()])
        self.run_evaluation()
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(len(self.scoring_calls), 1)
        self.assertEqual(self.result_rows()[0].get("transduction_result"), 1)
        self.assertNotIn("evaluation_error", self.result_rows()[0])


    def invalid_datasets(self):
        valid = self.messages_row()
        target = self.target_row()
        missing_input_id = dict(valid)
        del missing_input_id["problem_id"]
        missing_input_index = dict(valid)
        del missing_input_index["test_index"]
        missing_target_id = dict(target)
        del missing_target_id["problem_id"]
        missing_target_index = dict(target)
        del missing_target_index["test_index"]
        return [
            ("input missing problem_id", [missing_input_id], [target]),
            ("input missing test_index", [missing_input_index], [target]),
            ("duplicate input key", [valid, valid], [target]),
            ("target missing problem_id", [valid], [missing_target_id]),
            ("target missing test_index", [valid], [missing_target_index]),
            ("duplicate target key", [valid], [target, target]),
            ("missing target row", [valid], []),
            ("mismatched target key", [valid], [self.target_row("other")]),
        ]

    def test_invalid_keys_fail_before_generation(self):
        for description, inputs, targets in self.invalid_datasets():
            with self.subTest(description=description):
                self.prepare(inputs, targets)
                self.model.calls.clear()
                self.scoring_calls.clear()
                self.output_path.unlink(missing_ok=True)
                with self.assertRaises(ValueError):
                    self.run_evaluation()
                self.assertEqual(self.model.calls, [])
                self.assertEqual(self.scoring_calls, [])
                self.assertFalse(self.output_path.exists())

    def test_cli_rejects_invalid_keys_before_loading_a_model(self):
        for description, inputs, targets in self.invalid_datasets():
            with self.subTest(description=description):
                self.prepare(inputs, targets)
                self.output_path.unlink(missing_ok=True)
                argv = [
                    "sft_eval.py", "--model_path", "test-adapter",
                    "--input_data_path", str(self.input_path),
                    "--ground_truth_data_path", str(self.target_path),
                    "--output_path", str(self.output_path),
                ]
                with patch.object(sys, "argv", argv), patch.object(
                    self.evaluator, "load_model_and_tokenizer", return_value=(self.model, self.tokenizer)
                ) as load_model, redirect_stdout(self.console):
                    with self.assertRaises((ValueError, SystemExit)) as rejected:
                        self.evaluator.main()
                    if isinstance(rejected.exception, SystemExit):
                        self.assertNotEqual(rejected.exception.code, 0)
                    load_model.assert_not_called()
                self.assertFalse(self.output_path.exists())

    def test_malformed_json_fails_before_generation(self):
        for damaged_path in (self.input_path, self.target_path):
            with self.subTest(path=damaged_path.name):
                self.prepare([self.messages_row()])
                damaged_path.write_text('{"problem_id":\n', encoding="utf-8")
                self.model.calls.clear()
                self.output_path.unlink(missing_ok=True)
                with self.assertRaises(ValueError):
                    self.run_evaluation()
                self.assertEqual(self.model.calls, [])
                self.assertFalse(self.output_path.exists())

    def test_transduction_does_not_require_parsing_a_test_input_grid(self):
        row = self.messages_row()
        row["messages"][-1]["content"] = "Predict this puzzle's output from its supplied description."
        self.prepare([row])
        self.run_evaluation()
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.result_rows()[0].get("transduction_result"), 1)
        self.assertNotIn("evaluation_error", self.result_rows()[0])

    def test_expected_grid_can_come_from_target_answer_messages(self):
        target = self.target_row()
        del target["ground_truth_output"]
        self.prepare([self.messages_row()], [target])
        self.run_evaluation()
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.result_rows()[0].get("transduction_result"), 1)
        self.assertEqual(self.result_rows()[0].get("ground_truth_output"), [[2]])

    def complete_row(self, problem_id="chat", test_index=0):
        return {
            "idx": 0,
            "problem_id": problem_id,
            "test_index": test_index,
            "generated_output": RESPONSE,
            "ground_truth_response": RESPONSE,
            "ground_truth_output": [[2]],
            "generation_time": 0.125,
            "transduction_output": [[2]],
            "transduction_result": 1,
            "evaluation_details": {"transduction": {"result": 1, "predicted_grid": [[2]], "accuracy": 1.0}},
            "audit_marker": "preserve the completed row exactly",
        }

    def test_partial_resume_preserves_completed_rows_and_generates_only_missing_keys(self):
        self.prepare([self.messages_row("shared", 0), self.raw_row("shared", 1)])
        previous = self.complete_row("shared", 0)
        self.write_rows(self.output_path, [previous])
        previous_bytes = self.output_path.read_bytes()
        self.run_evaluation()
        rows = self.result_rows()
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(len(self.scoring_calls), 1)
        self.assertTrue(self.output_path.read_bytes().startswith(previous_bytes))
        self.assertEqual(rows[0], previous)
        self.assertCountEqual([(row["problem_id"], row["test_index"]) for row in rows], [("shared", 0), ("shared", 1)])
        self.assertEqual(rows[1].get("transduction_result"), 1)
        self.assertRegex(self.console.getvalue(), r"Transduction accuracy:\s*2/2")

    def test_complete_resume_uses_full_coverage_for_accuracy_without_generating(self):
        self.prepare([self.messages_row("shared", 0), self.raw_row("shared", 1)])
        previous = [self.complete_row("shared", 0), self.complete_row("shared", 1)]
        self.write_rows(self.output_path, previous)
        previous_bytes = self.output_path.read_bytes()
        self.run_evaluation()
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.scoring_calls, [])
        self.assertEqual(self.output_path.read_bytes(), previous_bytes)
        self.assertRegex(self.console.getvalue(), r"Transduction accuracy:\s*2/2")

    def test_duplicate_or_unexpected_prior_result_keys_are_rejected_before_generation(self):
        previous = self.complete_row()
        for description, saved in [
            ("duplicate", [previous, previous]),
            ("unexpected", [self.complete_row("unexpected")]),
        ]:
            with self.subTest(description=description):
                self.prepare([self.messages_row()])
                self.write_rows(self.output_path, saved)
                saved_bytes = self.output_path.read_bytes()
                self.model.calls.clear()
                with self.assertRaises(ValueError):
                    self.run_evaluation()
                self.assertEqual(self.model.calls, [])
                self.assertEqual(self.output_path.read_bytes(), saved_bytes)

    def test_incomplete_or_invalid_prior_results_cannot_count_as_completed(self):
        missing_score = self.complete_row()
        del missing_score["transduction_result"]
        missing_generation = self.complete_row()
        del missing_generation["generated_output"]
        invalid_score = dict(self.complete_row(), transduction_result="1")
        for description, saved in [
            ("key only", {"problem_id": "chat", "test_index": 0}),
            ("missing score", missing_score),
            ("missing generation", missing_generation),
            ("invalid score", invalid_score),
        ]:
            with self.subTest(description=description):
                self.prepare([self.messages_row()])
                self.write_rows(self.output_path, [saved])
                saved_bytes = self.output_path.read_bytes()
                self.model.calls.clear()
                with self.assertRaises(ValueError):
                    self.run_evaluation()
                self.assertEqual(self.model.calls, [])
                self.assertEqual(self.output_path.read_bytes(), saved_bytes)

    def test_generation_failure_is_visible_and_retains_prior_completed_rows(self):
        self.prepare([self.messages_row("prior"), self.messages_row("new"), self.messages_row("failure")])
        previous = self.complete_row("prior")
        self.write_rows(self.output_path, [previous])
        self.model.fail_on = 2
        with self.assertRaisesRegex(RuntimeError, "synthetic generation failure"):
            self.run_evaluation()
        rows = self.result_rows()
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(rows[0], previous)
        self.assertEqual([row["problem_id"] for row in rows], ["prior", "new"])
        self.assertEqual(rows[1].get("transduction_result"), 1)
        self.assertNotRegex(self.console.getvalue(), r"(?i)Evaluation completed!?")

    def test_scoring_runtime_failure_is_visible_and_retains_prior_completed_rows(self):
        self.prepare([self.messages_row("prior"), self.messages_row("failure")])
        previous = self.complete_row("prior")
        self.write_rows(self.output_path, [previous])
        with patch.object(self.evaluator, "evaluate_arc_output", side_effect=RuntimeError("synthetic scoring failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic scoring failure"):
                self.run_evaluation()
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.result_rows(), [previous])
        self.assertNotRegex(self.console.getvalue(), r"(?i)Evaluation completed!?")

    def test_damaged_unterminated_tail_is_removed_before_appending_missing_result(self):
        self.prepare([self.messages_row("prior"), self.messages_row("new")])
        previous = self.complete_row("prior")
        self.write_rows(self.output_path, [previous])
        previous_bytes = self.output_path.read_bytes()
        with self.output_path.open("ab") as stream:
            stream.write(b'{"problem_id": "new", "generated_output": "interrupted')
        self.run_evaluation()
        rows = self.result_rows()
        self.assertEqual(len(self.model.calls), 1)
        self.assertTrue(self.output_path.read_bytes().startswith(previous_bytes))
        self.assertEqual(rows[0], previous)
        self.assertEqual([row["problem_id"] for row in rows], ["prior", "new"])
        self.assertEqual(rows[1].get("transduction_result"), 1)

    def test_corrupt_terminated_result_line_fails_without_changing_file(self):
        self.prepare([self.messages_row("prior"), self.messages_row("new")])
        self.write_rows(self.output_path, [self.complete_row("prior")])
        with self.output_path.open("ab") as stream:
            stream.write(b'{"problem_id":\n')
        saved_bytes = self.output_path.read_bytes()
        with self.assertRaises(ValueError):
            self.run_evaluation()
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.output_path.read_bytes(), saved_bytes)

    def test_unique_extra_target_rows_allow_evaluating_a_subset(self):
        self.prepare([self.messages_row()], [self.target_row("unused"), self.target_row()])
        self.run_evaluation()
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual([row["problem_id"] for row in self.result_rows()], ["chat"])
        self.assertEqual(self.result_rows()[0]["transduction_result"], 1)

    def test_wrong_grid_is_scored_incorrect_against_the_matched_target(self):
        target = self.target_row()
        target["ground_truth_output"] = [[3]]
        self.prepare([self.messages_row()], [target])
        self.run_evaluation()
        row = self.result_rows()[0]
        self.assertEqual(len(self.scoring_calls), 1)
        self.assertEqual(row["ground_truth_output"], [[3]])
        self.assertEqual(row["transduction_output"], [[2]])
        self.assertEqual(row["transduction_result"], 0)

    def test_bare_and_fenced_grid_are_scored_but_narrative_color_mentions_are_not(self):
        for response, expected in [
            ("Blue", 1),
            ("```\nBlue\n```", 1),
            ("The answer might be Blue because that color appears in the puzzle.", 0),
        ]:
            with self.subTest(response=response):
                target = self.target_row()
                target["ground_truth_output"] = [[1]]
                self.prepare([self.messages_row()], [target])
                self.output_path.unlink(missing_ok=True)
                self.model.response = response
                self.model.calls.clear()
                self.run_evaluation()
                self.assertEqual(len(self.model.calls), 1)
                self.assertEqual(self.result_rows()[0]["transduction_result"], expected)

    def test_prior_result_with_incompatible_prompt_is_rejected(self):
        self.prepare([self.messages_row()])
        previous = self.complete_row()
        previous["input_prompt"] = "A prompt for a different evaluation request"
        self.write_rows(self.output_path, [previous])
        saved_bytes = self.output_path.read_bytes()
        with self.assertRaises(ValueError):
            self.run_evaluation()
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.output_path.read_bytes(), saved_bytes)

    def test_prior_result_with_different_requested_mode_is_rejected(self):
        self.prepare([self.messages_row()])
        previous = self.complete_row()
        previous["evaluation_mode"] = "induction"
        self.write_rows(self.output_path, [previous])
        saved_bytes = self.output_path.read_bytes()
        with self.assertRaises(ValueError):
            self.run_evaluation()
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.output_path.read_bytes(), saved_bytes)

    def test_complete_unterminated_final_row_is_preserved_before_appending(self):
        self.prepare([self.messages_row("prior"), self.messages_row("new")])
        previous = self.complete_row("prior")
        original = json.dumps(previous).encode("utf-8")
        self.output_path.write_bytes(original)
        self.run_evaluation()
        rows = self.result_rows()
        self.assertEqual(len(self.model.calls), 1)
        self.assertTrue(self.output_path.read_bytes().startswith(original + b"\n"))
        self.assertEqual(rows[0], previous)
        self.assertEqual([row["problem_id"] for row in rows], ["prior", "new"])
        self.assertEqual(rows[1]["transduction_result"], 1)


    def test_resume_rejects_changed_generation_settings(self):
        self.prepare([self.messages_row()])
        self.run_evaluation()
        saved = self.output_path.read_bytes()
        self.model.calls.clear()
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.run_evaluation(temperature=0.4)
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.output_path.read_bytes(), saved)

    def test_locked_output_rejects_a_second_writer(self):
        self.prepare([self.messages_row()])
        with self.evaluator.locked_output(self.output_path):
            with self.assertRaisesRegex(ValueError, "could not lock"):
                self.run_evaluation()
        self.assertEqual(self.model.calls, [])
        self.run_evaluation()
        self.assertEqual(len(self.result_rows()), 1)


    def test_target_colors_outside_arc_palette_fail_before_generation(self):
        target = self.target_row()
        target["ground_truth_output"] = [[10]]
        self.prepare([self.messages_row()], [target])
        with self.assertRaisesRegex(ValueError, "invalid ground truth"):
            self.run_evaluation()
        self.assertEqual(self.model.calls, [])

    def test_legacy_raw_result_without_an_index_requires_a_new_output(self):
        raw = self.raw_row()
        del raw["test_index"]
        self.prepare([raw], [self.target_row("raw")])
        previous = self.complete_row("raw")
        previous.update(test_index=None, original_data=raw)
        self.write_rows(self.output_path, [previous])
        saved = self.output_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "new output path"):
            self.run_evaluation()
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.output_path.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
