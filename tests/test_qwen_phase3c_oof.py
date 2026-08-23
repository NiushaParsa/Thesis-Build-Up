import json
import unittest
from pathlib import Path

import numpy as np

import qwen_phase3c_oof as phase3c_oof


class CleanPhase3COOFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = phase3c_oof.source_rows("train")

    def test_folds_are_deterministic_paper_grouped_and_complete(self) -> None:
        folds = phase3c_oof.training_folds(self.train)
        self.assertEqual(
            phase3c_oof.stable_hash(folds.tolist()),
            "892a44f57a11c6fa9be7ec708df3355f4247555bde319c61325c0d201492ee62",
        )
        self.assertEqual(np.bincount(folds).tolist(), [449, 449, 449, 450, 448])
        by_document = {}
        for row, fold in zip(self.train, folds):
            by_document.setdefault(str(row["document_id"]), set()).add(int(fold))
        self.assertEqual(len(by_document), 845)
        self.assertTrue(all(len(values) == 1 for values in by_document.values()))

    def test_original_primary_candidate_is_frozen_exactly(self) -> None:
        summary = json.loads(
            (
                phase3c_oof.ORIGINAL_PHASE3C_ROOT / "final_summary.json"
            ).read_text(encoding="utf-8")
        )
        observed = summary["models"]["qwen_logits_tree"]["selected_candidate"][
            "parameters"
        ]
        self.assertEqual(observed, phase3c_oof.FIXED_CANDIDATE)

    def test_fusion_matrix_has_five_logits_plus_173_tree_features(self) -> None:
        rows = self.train[:3]
        arrays = {
            "question_ids": np.asarray([str(row["question_id"]) for row in rows]),
            "document_ids": np.asarray([str(row["document_id"]) for row in rows]),
            "oracle_labels": np.asarray(
                [int(row["oracle_label"]) for row in rows], dtype=np.int64
            ),
            "logits": np.zeros((3, 5), dtype=np.float32),
            "token_counts": np.ones(3, dtype=np.int64),
        }
        matrix, names = phase3c_oof.fusion_matrix(rows, arrays)
        self.assertEqual(matrix.shape, (3, 178))
        self.assertEqual(len(names), 178)
        self.assertEqual(names[:5], [f"qwen_logit_{x}" for x in (10, 20, 40, 80, 160)])
        self.assertTrue(all(name.startswith("tree__") for name in names[5:]))

    def test_logit_alignment_rejects_question_reordering(self) -> None:
        rows = self.train[:2]
        arrays = {
            "question_ids": np.asarray(
                [str(rows[1]["question_id"]), str(rows[0]["question_id"])]
            ),
            "document_ids": np.asarray([str(row["document_id"]) for row in rows]),
            "oracle_labels": np.asarray(
                [int(row["oracle_label"]) for row in rows], dtype=np.int64
            ),
            "logits": np.zeros((2, 5), dtype=np.float32),
            "token_counts": np.ones(2, dtype=np.int64),
        }
        with self.assertRaisesRegex(RuntimeError, "question ID"):
            phase3c_oof.validate_logit_arrays(rows, arrays)


if __name__ == "__main__":
    unittest.main()
