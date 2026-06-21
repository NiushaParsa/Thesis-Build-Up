from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import chunking_utils
import metrics
import prepare_dataset
from fixed_sized_granularity_separate import METHOD_NAME, _make_eval_id


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class WordTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def encode(self, text, add_special_tokens=False):
        result = []
        for token in text.split():
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            result.append(self.token_to_id[token])
        return result

    def decode(self, token_ids):
        return " ".join(self.id_to_token[token_id] for token_id in token_ids)


class DeterministicIdTests(unittest.TestCase):
    def test_ingestion_uuid_is_stable_and_seed_sensitive(self):
        seed = "paper_g3_c7"
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))
        self.assertEqual(prepare_dataset._make_uuid(seed), expected)
        self.assertEqual(prepare_dataset._make_uuid(seed), prepare_dataset._make_uuid(seed))
        self.assertNotEqual(prepare_dataset._make_uuid(seed), prepare_dataset._make_uuid(seed + "x"))

    def test_evaluation_uuid_is_stable_per_question_and_granularity(self):
        question_id = "question-1"
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{METHOD_NAME}|{question_id}|3"))
        self.assertEqual(_make_eval_id(question_id, 3), expected)
        self.assertNotEqual(_make_eval_id(question_id, 3), _make_eval_id(question_id, 4))


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.original_path = prepare_dataset.CHECKPOINT_PATH
        self.original_checkpoint = prepare_dataset._checkpoint
        self.temp_dir = tempfile.TemporaryDirectory()
        prepare_dataset.CHECKPOINT_PATH = Path(self.temp_dir.name) / "checkpoint.json"
        prepare_dataset._checkpoint = {}

    def tearDown(self):
        prepare_dataset.CHECKPOINT_PATH = self.original_path
        prepare_dataset._checkpoint = self.original_checkpoint
        self.temp_dir.cleanup()

    def test_saved_stages_are_loaded_and_skipped_on_resume(self):
        prepare_dataset.save_checkpoint("doc-1", 10, split="train")
        prepare_dataset.save_checkpoint("doc-1", "questions", split="train")
        prepare_dataset.save_checkpoint("doc-1", 10, split="train")

        prepare_dataset._checkpoint = {}
        loaded = prepare_dataset.load_checkpoint()

        self.assertEqual(loaded["doc-1"]["split"], "train")
        self.assertEqual(loaded["doc-1"]["done"], [10, "questions"])
        self.assertTrue(prepare_dataset.is_done("doc-1", 10))
        self.assertTrue(prepare_dataset.is_done("doc-1", "questions"))
        self.assertFalse(prepare_dataset.is_done("doc-1", "evidence"))


class ChunkingTests(unittest.TestCase):
    def test_fixed_chunks_are_non_overlapping_and_reconstruct_text(self):
        text = "abcdefghij"
        with patch.object(chunking_utils, "get_tokenizer", return_value=CharacterTokenizer()):
            chunks = chunking_utils.chunk_text(text, 4)

        self.assertEqual([chunk["content"] for chunk in chunks], ["abcd", "efgh", "ij"])
        self.assertEqual([chunk["token_count"] for chunk in chunks], [4, 4, 2])
        self.assertEqual(
            [(chunk["span_start"], chunk["span_end"]) for chunk in chunks],
            [(0, 4), (4, 8), (8, 10)],
        )
        self.assertEqual("".join(chunk["content"] for chunk in chunks), text)


class EvidenceFilteringTests(unittest.TestCase):
    def test_only_questions_with_non_empty_highlighted_evidence_are_answerable(self):
        paper = {
            "qas": {
                "question_id": ["answerable", "empty", "missing"],
                "answers": [
                    {"answer": [{"highlighted_evidence": ["  ", "useful evidence"]}]},
                    {"answer": [{"highlighted_evidence": ["", "   "]}]},
                    {"answer": [{}]},
                ],
            }
        }
        self.assertEqual(prepare_dataset._get_answerable_question_ids(paper), {"answerable"})


class TokenF1Tests(unittest.TestCase):
    def test_normalization_and_multiset_overlap(self):
        tokenizer = WordTokenizer()
        with patch.object(metrics, "get_tokenizer", return_value=tokenizer):
            self.assertEqual(metrics.token_f1("Hello, WORLD!", "hello world"), 1.0)
            self.assertAlmostEqual(metrics.token_f1("alpha alpha beta", "alpha beta beta"), 2 / 3)
            self.assertEqual(metrics.token_f1("alpha", "gamma"), 0.0)
            self.assertEqual(metrics.token_f1("", "gamma"), 0.0)


class JsonlWriterTests(unittest.TestCase):
    def test_parallel_record_groups_produce_complete_json_lines(self):
        output = io.StringIO()
        worker_count = 20
        records_per_worker = 10

        def write_group(worker):
            prepare_dataset._write_jsonl_records(
                output,
                ({"worker": worker, "record": record} for record in range(records_per_worker)),
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_group, range(worker_count)))

        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), worker_count * records_per_worker)
        by_worker = {}
        for index, row in enumerate(rows):
            by_worker.setdefault(row["worker"], []).append((index, row["record"]))
        for records in by_worker.values():
            positions = [position for position, _ in records]
            values = [value for _, value in records]
            self.assertEqual(positions, list(range(positions[0], positions[0] + records_per_worker)))
            self.assertEqual(values, list(range(records_per_worker)))


if __name__ == "__main__":
    unittest.main()
