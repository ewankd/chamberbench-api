"""Data generation, detector fitting, calibration, and held-out evaluation."""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from statistics import mean, pstdev


VERSION = "1"
SENSORS = ("pressure_mtorr", "rf_power_w", "temperature_c")
NOISE_SD = (0.8, 10.0, 1.4)
FAMILIAR_RECIPES = ((30.0, 500.0, 65.0), (42.0, 640.0, 79.0))
NEW_RECIPE = (55.0, 785.0, 94.0)
DETECTORS = ("raw", "recipe_aware")
N_TRAIN, N_CALIBRATION, N_PAIRS, N_FAULTS = 600, 400, 400, 80


def number(value, name, lower, upper):
    if type(value) not in (int, float) or not lower <= value <= upper:
        raise ValueError(f"{name} must be a number between {lower} and {upper}.")
    return float(value)


def require_fields(obj, required, optional=()):
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object.")
    missing = set(required) - obj.keys()
    unknown = obj.keys() - set(required) - set(optional)
    if missing:
        raise ValueError("Missing fields: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError("Unknown fields: " + ", ".join(sorted(unknown)))


@dataclass(frozen=True)
class Config:
    seed: int = 42
    target_fpr: float = 0.05
    fault_strength: float = 3.5

    @classmethod
    def parse(cls, obj):
        require_fields(obj, (), ("seed", "target_fpr", "fault_strength"))
        defaults = cls()
        seed = obj.get("seed", defaults.seed)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer from 0 to 4294967295.")
        return cls(
            seed,
            number(obj.get("target_fpr", defaults.target_fpr), "target_fpr", 0.01, 0.25),
            number(obj.get("fault_strength", defaults.fault_strength), "fault_strength", 0.5, 8),
        )

    def experiment_id(self):
        settings = json.dumps([VERSION, asdict(self)], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(settings.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Sample:
    setpoint: tuple
    reading: tuple

    def features(self, detector):
        if detector == "raw":
            return self.reading
        return tuple(x - s for x, s in zip(self.reading, self.setpoint))

    @classmethod
    def parse(cls, obj):
        require_fields(obj, ("setpoint", "reading"))
        limits = ((0, 2000), (0, 10000), (-100, 1000))
        vectors = []
        for key in ("setpoint", "reading"):
            require_fields(obj[key], SENSORS)
            vectors.append(tuple(
                number(obj[key][name], f"{key}.{name}", low, high)
                for name, (low, high) in zip(SENSORS, limits)
            ))
        return cls(*vectors)


def observe(recipe, noise):
    return Sample(recipe, tuple(s + e for s, e in zip(recipe, noise)))


def noise_vector(rng):
    return [rng.gauss(0, sd) for sd in NOISE_SD]


def healthy_samples(rng, count):
    return [
        observe(rng.choice(FAMILIAR_RECIPES), noise_vector(rng))
        for _ in range(count)
    ]


def held_out_pairs(rng, strength):
    faulty_indices = set(rng.sample(range(N_PAIRS), N_FAULTS))
    for i in range(N_PAIRS):
        noise = noise_vector(rng)
        faulty = i in faulty_indices
        if faulty:
            sensor = rng.randrange(len(SENSORS))
            noise[sensor] += rng.choice((-1, 1)) * strength * NOISE_SD[sensor]
        # The pair shares one noise vector and label. Only the recipe changes.
        yield faulty, observe(rng.choice(FAMILIAR_RECIPES), noise), observe(NEW_RECIPE, noise)


@dataclass(frozen=True)
class Model:
    center: tuple
    scale: tuple
    threshold: float

    def contributions(self, features):
        return tuple(((x - m) / s) ** 2 for x, m, s in zip(features, self.center, self.scale))

    def evaluate(self, features):
        terms = self.contributions(features)
        score = sum(terms)
        return {
            "score": score,
            "threshold": self.threshold,
            "flagged": score > self.threshold,
            "top_sensor": SENSORS[max(range(len(terms)), key=terms.__getitem__)],
        }


def fit(detector, training, calibration, target_fpr):
    columns = list(zip(*(s.features(detector) for s in training)))
    center = tuple(mean(column) for column in columns)
    scale = tuple(max(pstdev(column), 1e-12) for column in columns)
    uncalibrated = Model(center, scale, 0.0)
    scores = sorted(sum(uncalibrated.contributions(s.features(detector))) for s in calibration)
    # One-based rank. The configured FPR range guarantees the rank is in bounds.
    rank = math.ceil((len(scores) + 1) * (1 - target_fpr))
    return Model(center, scale, scores[rank - 1])


def rates(tp, fp, tn, fn):
    def ratio(a, b):
        return a / b if b else None

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": ratio(tp, tp + fn),
        "false_alarm_rate": ratio(fp, fp + tn),
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def run(config):
    # Independent streams keep the test cases fixed when calibration settings change.
    training = healthy_samples(random.Random(f"{config.seed}/train"), N_TRAIN)
    calibration = healthy_samples(random.Random(f"{config.seed}/calibrate"), N_CALIBRATION)
    pairs = list(held_out_pairs(random.Random(f"{config.seed}/test"), config.fault_strength))
    models = {}
    metrics = {}
    for detector in DETECTORS:
        model = fit(detector, training, calibration, config.target_fpr)
        models[detector] = asdict(model)
        metrics[detector] = {}
        for cohort, samples in (
            ("familiar", ((fault, familiar) for fault, familiar, new in pairs)),
            ("new_recipe", ((fault, new) for fault, familiar, new in pairs)),
        ):
            tp = fp = tn = fn = 0
            for fault, sample in samples:
                flagged = sum(model.contributions(sample.features(detector))) > model.threshold
                if fault and flagged:
                    tp += 1
                elif flagged:
                    fp += 1
                elif fault:
                    fn += 1
                else:
                    tn += 1
            metrics[detector][cohort] = rates(tp, fp, tn, fn)
    return {
        "id": config.experiment_id(), "version": VERSION, "config": asdict(config),
        "sample_counts": {
            "train_healthy": N_TRAIN, "calibration_healthy": N_CALIBRATION,
            "test_per_cohort": N_PAIRS, "faults_per_cohort": N_FAULTS,
        },
        "sensor_order": SENSORS, "models": models, "metrics": metrics,
    }


def score(result, sample):
    return {
        detector: Model(**params).evaluate(sample.features(detector))
        for detector, params in result["models"].items()
    }
