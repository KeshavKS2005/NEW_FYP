"""
Tests for Module 11: Incremental YOLO Challenger Trainer & Dual-Gated Metric Evaluator
Validates Gate 1 (Zero Catastrophic Forgetting <= 1.5% drop) and Gate 2 (New-Class Competence mAP50 >= 0.60, Recall >= 0.65).
"""

import pytest
from module_11.config import Module11Config
from module_11.trainer_evaluator import ChallengerTrainerEvaluator


class TestModule11:
    """Test suite for Dual Metric Evaluation Gates."""

    def test_gate1_old_class_retention_pass(self):
        """Challenger drop in old class mAP <= 1.5% passes Gate 1."""
        cfg = Module11Config(max_old_map_drop=0.015)
        evaluator = ChallengerTrainerEvaluator(cfg)

        metrics = evaluator.evaluate_gates(
            champ_old_map=0.520,
            chal_old_map=0.510,     # Drop of 0.010 <= 0.015
            new_cls_map50=0.720,
            new_cls_recall=0.750
        )
        assert metrics["passed_old_retention"] is True
        assert metrics["passed_new_competence"] is True
        assert metrics["promote_challenger"] is True

    def test_gate1_catastrophic_forgetting_reject(self):
        """Challenger drop in old class mAP > 1.5% fails Gate 1 and rejects promotion."""
        cfg = Module11Config(max_old_map_drop=0.015)
        evaluator = ChallengerTrainerEvaluator(cfg)

        metrics = evaluator.evaluate_gates(
            champ_old_map=0.520,
            chal_old_map=0.490,     # Drop of 0.030 > 0.015 (catastrophic forgetting!)
            new_cls_map50=0.850,
            new_cls_recall=0.900
        )
        assert metrics["passed_old_retention"] is False
        assert metrics["passed_new_competence"] is True
        assert metrics["promote_challenger"] is False

    def test_gate2_new_class_incompetence_reject(self):
        """New class mAP50 < 0.60 or Recall < 0.65 fails Gate 2 and rejects promotion."""
        cfg = Module11Config(min_new_class_map50=0.60, min_new_class_recall=0.65)
        evaluator = ChallengerTrainerEvaluator(cfg)

        # Subtest A: Low mAP50
        metrics_a = evaluator.evaluate_gates(
            champ_old_map=0.500,
            chal_old_map=0.505,     # Old classes improved
            new_cls_map50=0.550,     # < 0.60
            new_cls_recall=0.800
        )
        assert metrics_a["passed_old_retention"] is True
        assert metrics_a["passed_new_competence"] is False
        assert metrics_a["promote_challenger"] is False

        # Subtest B: Low Recall
        metrics_b = evaluator.evaluate_gates(
            champ_old_map=0.500,
            chal_old_map=0.505,
            new_cls_map50=0.750,
            new_cls_recall=0.580     # < 0.65
        )
        assert metrics_b["passed_old_retention"] is True
        assert metrics_b["passed_new_competence"] is False
        assert metrics_b["promote_challenger"] is False
