# ChamberBench

**Can a fault detector tell a machine fault from a planned change in operating settings?** ChamberBench is a local Python REST API that runs this experiment on a simulated semiconductor process chamber. It compares two anomaly detectors, saves each experiment in SQLite, and uses the saved models to score new readings.

Imagine a chamber whose temperature target changes from 65 °C to 94 °C. A reading of 94.4 °C could be perfectly ordinary. A detector trained on raw temperatures may still raise an alarm. ChamberBench tests whether giving the detector the intended settings helps it make that distinction.

## Run your first experiment

You need Python 3.11 or newer. All runtime and test dependencies are part of Python. Extract the ZIP and open a terminal in the folder that contains this README. On Windows, `py` also works in place of `python`.

1. Start the API in the first terminal.

   ```shell
   python -m chamberbench
   ```

2. Open a second terminal in the same folder and run the client.

   ```shell
   python -m chamberbench.demo
   ```

3. Look at the two `new_recipe` rows. **Recall** is the share of actual faults caught. **False alarms** is the share of healthy cases flagged. The last two lines show how the same models judge one new reading. Open the printed result URL in a browser to see the full JSON. Press Ctrl+C in the first terminal to stop the service.

To change one variable, run the client again with a weaker fault. Keep the same seed so the underlying cases stay matched.

```shell
python -m chamberbench.demo --fault-strength 1.5
```

## What happens to an incoming request?

The demo is a client, a separate program that sends requests to the server. It sends JSON to `POST /experiments`. The server checks the inputs, runs both detectors on the same data, saves the fitted parameters and test results, and returns an experiment ID. Sending the same settings again retrieves the saved result.

The client then sends a setpoint and an actual reading to `POST /experiments/{id}/score`. The server loads the two fitted models and returns a score, alarm threshold, decision, and the sensor contributing most to each score. This request uses the saved models directly.

## An example result

These counts come from the default simulation on Python 3.12 with seed 42. Each test group contains 80 injected faults and 320 healthy cases.

| Inputs given to the detector | Test recipe | Faults caught out of 80 | Healthy cases flagged out of 320 |
| --- | --- | ---: | ---: |
| Raw readings | Familiar | 18 | 18 |
| Difference from setpoint | Familiar | 69 | 13 |
| Raw readings | New | 80 | 320 |
| Difference from setpoint | New | 69 | 13 |

On the new recipe, the raw detector caught every fault because it flagged every reading. Accounting for setpoints reduced false alarms to 13, with 11 missed faults. The simulator deliberately keeps sensor noise the same across recipes. These measurements describe that assumption and would need to be checked against real equipment data before drawing conclusions about factory performance.

## How the experiment works

1. **Fit.** Generate 600 healthy runs from two recipes. Both detectors learn a mean and standard deviation for each input. The raw detector sees pressure, RF power, and temperature. The recipe-aware detector sees the measured value minus its target for each sensor.
2. **Calibrate.** Score 400 separate healthy runs. For each detector, select a threshold from its own calibration scores using the requested false alarm target. The default target is 5%. The observed test rate can differ.
3. **Test.** Generate 400 held-out cases and inject a positive or negative offset into one sensor in exactly 80 of them. Replay every case twice, once with a familiar recipe and once with an unseen recipe. Each pair has identical noise, fault location, and fault size so the comparison isolates the effect of the setpoints.
4. **Score.** For each sensor, subtract its learned mean and divide by its learned standard deviation. Square those three values and sum them. A score strictly above the calibrated threshold triggers an alarm. The largest term identifies which sensor contributed the most to the decision.

The score uses a diagonal Gaussian model of healthy behavior. It learns without labeled faults. The test labels are used only for evaluation. The simulator uses independent Gaussian sensor noise, fixed setpoints within each recipe, and one-sensor faults. The two test groups are paired, so they should not be treated as independent samples.

## API reference

The default base URL is `http://127.0.0.1:8000`. POST requests use `Content-Type: application/json`.

| Method | Path | Response |
| --- | --- | --- |
| GET | `/` | Available routes |
| GET | `/health` | Service status |
| POST | `/experiments` | Settings, fitted models, confusion counts, and metrics. Returns 201 for a new run or 200 for an existing run. Both include the result URL in the Location header. |
| GET | `/experiments/{id}` | The stored experiment |
| POST | `/experiments/{id}/score` | One decision per detector for the submitted reading |

**Experiment settings.** Send `{}` for the defaults. The seed must be an integer from 0 through 4294967295. The target false alarm rate ranges from 0.01 to 0.25, and fault strength ranges from 0.5 to 8 in units of a sensor's normal noise standard deviation.

```json
{"seed": 42, "target_fpr": 0.05, "fault_strength": 3.5}
```

**Score a reading.** Send the following object to the score endpoint. Both objects require all three sensors. Pressure is in millitorr, RF power in watts, and temperature in degrees Celsius.

```json
{
  "setpoint": {"pressure_mtorr": 55, "rf_power_w": 785, "temperature_c": 94},
  "reading":  {"pressure_mtorr": 55.2, "rf_power_w": 786, "temperature_c": 94.4}
}
```

For the default experiment the raw detector flags this sample, while the recipe-aware detector accepts it. Change **only** the measured temperature to 105 °C and the recipe-aware detector flags it too. Its largest score contribution comes from temperature.

You can try that change in PowerShell while the server is running:

```powershell
$base = 'http://127.0.0.1:8000'
$run = Invoke-RestMethod "$base/experiments" -Method Post -ContentType 'application/json' -Body '{}'
$sample = @{
    setpoint = @{ pressure_mtorr = 55; rf_power_w = 785; temperature_c = 94 }
    reading  = @{ pressure_mtorr = 55.2; rf_power_w = 786; temperature_c = 105 }
}
$result = Invoke-RestMethod "$base/experiments/$($run.id)/score" -Method Post -ContentType 'application/json' -Body ($sample | ConvertTo-Json)
$result.detectors.recipe_aware
```

The server returns JSON errors for malformed JSON (400), unknown experiments (404), the wrong HTTP method (405), a missing content length (411), bodies larger than 8 KiB (413), the wrong content type (415), and invalid fields or ranges (422).

## Code and tests

| File | Responsibility |
| --- | --- |
| `chamberbench/api.py` | Routes, JSON validation, and HTTP responses |
| `chamberbench/experiment.py` | Simulation, model fitting, calibration, metrics, and scoring |
| `chamberbench/store.py` | SQLite persistence and duplicate-request handling |
| `chamberbench/demo.py` | A client that calls the running API |
| `tests/` | Experiment design, API behavior, persistence, concurrent duplicate runs, and a live HTTP demo |

Experiments are identified by their normalized settings and engine version. Repeated submissions use the same database row, including concurrent retries. SQLite stores the full result and both models in one transaction. By default the development server listens on the loopback interface and uses `chamberbench.sqlite3` in the current folder. You can override the port and database path with `--port` and `--db`.

```shell
python -m unittest discover -s tests -v
```

The included GitHub Actions workflow runs these tests on Python 3.11, 3.12, and 3.13.

## Upload to GitHub

Create an empty GitHub repository named `chamberbench-api`. In the extracted folder containing this README, run the commands below. Replace `YOUR-USERNAME` with your GitHub username and use a Git client signed in to your account.

```shell
git init -b main
git add .
git commit -m "Add ChamberBench experiment API"
git remote add origin https://github.com/YOUR-USERNAME/chamberbench-api.git
git push -u origin main
```

After uploading, open the repository's Actions tab to see the test run.
