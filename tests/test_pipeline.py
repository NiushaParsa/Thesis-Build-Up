from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import chunking_utils
import metrics
import prepare_dataset
from fixed_sized_granularity_separate import (
    METHOD_NAME,
    _make_eval_id,
    cosine_similarity,
    evaluate_question,
)


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

    def test_precision_recall_f1_are_zero_for_zero_length_normalized_text(self):
        tokenizer = WordTokenizer()
        with patch.object(metrics, "get_tokenizer", return_value=tokenizer):
            self.assertEqual(
                metrics.token_precision_recall_f1("!!!", "..."),
                (0.0, 0.0, 0.0),
            )


class FakeEvaluationClient:
    def __init__(self, evidence_points, chunk_points):
        self.evidence_points = evidence_points
        self.chunk_points = chunk_points
        self.query_calls = []
        self.scroll_calls = []

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return self.evidence_points, None

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=self.chunk_points)


class FixedSeparateEvaluationTests(unittest.TestCase):
    @staticmethod
    def evidence(point_id, text, vector):
        return SimpleNamespace(
            id=point_id,
            payload={"evidence_text": text},
            vector=vector,
        )

    @staticmethod
    def chunk(point_id, index, text, vector, score, token_count):
        return SimpleNamespace(
            id=point_id,
            payload={"chunk_idx": index, "content": text, "chunk_size": token_count},
            vector=vector,
            score=score,
        )

    def test_multiple_unique_evidence_and_fewer_than_k_results(self):
        diagonal = 2 ** -0.5
        client = FakeEvaluationClient(
            evidence_points=[
                self.evidence("e3", "alpha beta", [0.6, 0.8]),
                self.evidence("e2", "gamma", [0.0, 1.0]),
                self.evidence("e1", " alpha beta ", [1.0, 0.0]),
            ],
            chunk_points=[
                self.chunk("c1", 4, "alpha", [1.0, 0.0], 0.9, 1),
                self.chunk("c2", 7, "gamma delta", [diagonal, diagonal], 0.5, 2),
            ],
        )
        tokenizer = WordTokenizer()
        with patch.object(metrics, "get_tokenizer", return_value=tokenizer):
            records = list(
                evaluate_question(
                    client=client,
                    question_point_id="q1",
                    question_vector=[1.0, 0.0],
                    document_id="doc1",
                    question_text="question",
                    split="test",
                    top_k=5,
                    granularity_levels=[1],
                )
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["unique_evidence_ids"], ["e1", "e2"])
        self.assertEqual(record["unique_evidence_count"], 2)
        self.assertEqual(record["k_requested"], 5)
        self.assertEqual(record["retrieved_k"], 2)
        self.assertEqual(record["returned_k"], 2)
        self.assertEqual(record["joined_retrieved_text_token_count"], 3)
        self.assertEqual(record["joined_unique_evidence_token_count"], 3)
        self.assertEqual(record["set_level_precision"], 0.666667)
        self.assertEqual(record["set_level_recall"], 0.666667)
        self.assertEqual(record["set_level_f1"], 0.666667)
        self.assertEqual(record["best_query_similarity_topk"], 0.9)
        self.assertEqual(record["mean_query_similarity_topk"], 0.7)
        self.assertEqual(record["best_evidence_similarity_topk"], 1.0)
        self.assertEqual(record["mean_evidence_similarity_topk"], 0.603553)

        first, second = record["retrieved_chunks"]
        self.assertEqual(
            (first["chunk_id"], first["chunk_idx"], first["rank"]),
            ("c1", 4, 1),
        )
        self.assertEqual(first["granularity_level"], 1)
        self.assertEqual(first["granularity_tokens"], 10)
        self.assertEqual(first["chunk_token_count"], 1)
        self.assertEqual(first["query_similarity"], 0.9)
        self.assertEqual(
            [item["cosine_similarity"] for item in first["evidence_cosine_similarities"]],
            [1.0, 0.0],
        )
        self.assertEqual(first["max_evidence_similarity"], 1.0)
        self.assertEqual(first["mean_evidence_similarity"], 0.5)
        self.assertEqual(
            [item["token_f1"] for item in first["evidence_token_f1_scores"]],
            [0.666667, 0.0],
        )
        self.assertEqual(first["max_chunk_f1"], 0.666667)
        self.assertEqual(second["mean_evidence_similarity"], 0.707107)
        self.assertEqual(second["max_chunk_f1"], 0.666667)
        self.assertTrue(client.scroll_calls[0]["with_vectors"])
        self.assertTrue(client.query_calls[0]["with_vectors"])

    def test_empty_evidence_skips_question_without_searching(self):
        client = FakeEvaluationClient(
            evidence_points=[self.evidence("e1", "   ", [1.0, 0.0])],
            chunk_points=[self.chunk("c1", 0, "alpha", [1.0, 0.0], 0.9, 1)],
        )
        records = list(
            evaluate_question(
                client,
                "q1",
                [1.0, 0.0],
                "doc1",
                "question",
                "test",
                granularity_levels=[1],
            )
        )
        self.assertEqual(records, [])
        self.assertEqual(client.query_calls, [])

    def test_no_retrieval_results_yields_zero_set_metrics(self):
        client = FakeEvaluationClient(
            evidence_points=[self.evidence("e1", "alpha", [1.0, 0.0])],
            chunk_points=[],
        )
        tokenizer = WordTokenizer()
        with patch.object(metrics, "get_tokenizer", return_value=tokenizer):
            record = list(
                evaluate_question(
                    client,
                    "q1",
                    [1.0, 0.0],
                    "doc1",
                    "question",
                    "test",
                    top_k=5,
                    granularity_levels=[1],
                )
            )[0]
        self.assertEqual(record["returned_k"], 0)
        self.assertEqual(record["retrieved_chunks"], [])
        self.assertEqual(record["set_level_precision"], 0.0)
        self.assertEqual(record["set_level_recall"], 0.0)
        self.assertEqual(record["set_level_f1"], 0.0)
        self.assertEqual(record["best_query_similarity_topk"], 0.0)
        self.assertEqual(record["mean_evidence_similarity_topk"], 0.0)

    def test_cosine_similarity_has_manually_verifiable_and_zero_vector_behavior(self):
        self.assertEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 1.0], [1.0, 0.0]), 2 ** -0.5)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])


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
