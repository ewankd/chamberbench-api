"""A small client that runs an experiment and scores a sample over HTTP."""

import argparse
import json
from urllib import error, request


def call(base, path, data=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = request.Request(base.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=15) as response:
        return response.status, json.load(response)


def percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--fault-strength", type=float, default=3.5)
    args = parser.parse_args(argv)
    settings = {"seed": args.seed, "target_fpr": args.target_fpr, "fault_strength": args.fault_strength}
    try:
        status, result = call(args.base_url, "/experiments", settings)
        item_path = "/experiments/" + result["id"]
        print(f"HTTP {status} | Experiment {result['id']}")
        print(f"Full result: {args.base_url.rstrip('/')}{item_path}\n")
        print(f"{'Detector':<15} {'Test recipe':<13} {'Recall':>8} {'False alarms':>13} {'F1':>8}")
        for name, cohorts in result["metrics"].items():
            for cohort, stats in cohorts.items():
                print(f"{name:<15} {cohort:<13} {percent(stats['recall']):>8} "
                      f"{percent(stats['false_alarm_rate']):>13} {percent(stats['f1']):>8}")
        sample = {
            "setpoint": {"pressure_mtorr": 55, "rf_power_w": 785, "temperature_c": 94},
            "reading": {"pressure_mtorr": 55.2, "rf_power_w": 786, "temperature_c": 94.4},
        }
        _, scored = call(args.base_url, item_path + "/score", sample)
        print("\nOne near-target reading on the new recipe:")
        for name, values in scored["detectors"].items():
            decision = "FLAG" if values["flagged"] else "PASS"
            print(f"{name:<15} {decision} (score {values['score']:.2f}, "
                  f"threshold {values['threshold']:.2f})")
    except error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()}") from None
    except (error.URLError, TimeoutError):
        raise SystemExit("Could not reach the API. Start python -m chamberbench in another terminal.") from None


if __name__ == "__main__":
    main()
