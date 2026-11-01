import math
import random
import unittest

from chamberbench.experiment import (
    Config, Model, Sample, N_CALIBRATION, N_FAULTS, N_PAIRS, N_TRAIN, fit,
    healthy_samples, rates, run, score,
)


class ExperimentTests(unittest.TestCase):
    def test_reproducible_run_and_normalized_settings(self):
        first = run(Config())
        second = run(Config.parse({"fault_strength": 3.50, "seed": 42, "target_fpr": 0.05}))
        self.assertEqual(first, second)
        changed = run(Config(seed=43))
        self.assertNotEqual(first["id"], changed["id"])
        self.assertNotEqual(first["models"], changed["models"])

    def test_test_labels_do_not_influence_model_or_threshold(self):
        easy, hard = run(Config(fault_strength=8.0)), run(Config(fault_strength=0.5))
        self.assertEqual(easy["models"], hard["models"])
        self.assertGreater(easy["metrics"]["recipe_aware"]["familiar"]["recall"],
                           hard["metrics"]["recipe_aware"]["familiar"]["recall"])

    def test_calibration_uses_separate_healthy_cases_and_correct_rank(self):
        config = Config()
        training = healthy_samples(random.Random("42/train"), N_TRAIN)
        calibration = healthy_samples(random.Random("42/calibrate"), N_CALIBRATION)
        model = fit("recipe_aware", training, calibration, config.target_fpr)
        values = sorted(model.evaluate(s.features("recipe_aware"))["score"] for s in calibration)
        index = math.ceil((len(values) + 1) * (1 - config.target_fpr)) - 1
        self.assertEqual(model.threshold, values[index])
        self.assertLessEqual(sum(x > model.threshold for x in values) / len(values),
                             config.target_fpr)

    def test_paired_design_and_confusion_counts(self):
        result = run(Config())
        for cohorts in result["metrics"].values():
            for counts in cohorts.values():
                self.assertEqual(counts["tp"] + counts["fn"], N_FAULTS)
                self.assertEqual(counts["tn"] + counts["fp"], N_PAIRS - N_FAULTS)
        aware = result["metrics"]["recipe_aware"]
        self.assertEqual(aware["familiar"], aware["new_recipe"])
        self.assertLess(aware["new_recipe"]["fp"], result["metrics"]["raw"]["new_recipe"]["fp"])

    def test_setpoint_shift_preserves_residual_score(self):
        result = run(Config())
        a = Sample((30, 500, 65), (31, 498, 73))
        b = Sample((55, 785, 94), (56, 783, 102))
        a_scores, b_scores = score(result, a), score(result, b)
        self.assertEqual(a_scores["recipe_aware"], b_scores["recipe_aware"])
        self.assertNotEqual(a_scores["raw"]["score"], b_scores["raw"]["score"])
        self.assertEqual(a_scores["recipe_aware"]["top_sensor"], "temperature_c")

    def test_score_equal_to_threshold_is_not_flagged(self):
        model = Model((0, 0, 0), (1, 1, 1), 9)
        self.assertFalse(model.evaluate((3, 0, 0))["flagged"])

    def test_metrics_when_no_predictions_are_positive(self):
        self.assertEqual(rates(0, 0, 5, 1), {
            "tp": 0, "fp": 0, "tn": 5, "fn": 1, "recall": 0,
            "precision": None, "false_alarm_rate": 0, "f1": 0,
        })

    def test_threshold_tradeoff_on_identical_test_pairs(self):
        strict, loose = run(Config(target_fpr=0.01)), run(Config(target_fpr=0.25))
        for detector in strict["models"]:
            self.assertGreaterEqual(strict["models"][detector]["threshold"],
                                    loose["models"][detector]["threshold"])
            for cohort in ("familiar", "new_recipe"):
                a = strict["metrics"][detector][cohort]
                b = loose["metrics"][detector][cohort]
                self.assertLessEqual(a["tp"], b["tp"])
                self.assertLessEqual(a["fp"], b["fp"])
