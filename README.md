# Hybrid Physics-Informed Forecasting

### [Presentation deck and demo video (Google Drive)](https://drive.google.com/drive/folders/1ua1AiKySnGl7Wt33RzyJdwDGcEIiU0Zl?usp=sharing)

**Problem SH-DST-03.** Given seven days of a GNSS satellite's position and clock
error, predict day 8 at arbitrary timestamps.

We do not learn the orbit from the data. We **derive** it from orbital mechanics,
which fixes the rhythms the error is allowed to contain before a single row is
read, and then fit only the amplitudes. The result is a model that works on 10% of
a week of data, and that refuses to predict when the data cannot support it.

![How the model works](figures/f22_how_it_works.png)

---

## Team

|                       |                                                                                   |
| --------------------- | --------------------------------------------------------------------------------- |
| **Team**              | Honoured Ones &nbsp;·&nbsp; `SHIH-TID-264`                                        |
| **Team Lead**         | Vedant Ghule                                                                      |
| **Members**           | Vedant Ghule, Jayvardhan Gaikwad, Neil Lunavat, Danish Sayyad, Anushka Puri       |
| **College**           | PCCOE, Pune                                                                       |
| **Problem Statement** | `SH-DST-03`                                                                       |
| **Event**             | Smart Horizon 2026, 48-Hour International Hackathon (Grand Finale)                |
| **Host**              | New Horizon College of Engineering, Bengaluru &nbsp;·&nbsp; 3 to 5 September 2026 |

---

## Results

| what                                     | result                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Their MEO files**                      | **57%** and **19%** better than doing nothing, and better than a predictor shown the answer key's own average                                                                                                                                                                                                                                                         |
| **120 physics-generated windows**        | **78%** better on GEO, **51%** on MEO, against known day-8 truth the model never saw                                                                                                                                                                                                                                                                                  |
| **16 methods benchmarked on their data** | first, by **28%** over the runner-up                                                                                                                                                                                                                                                                                                                                  |
| **Six standard ML models, 27,000 rows**  | all plateau at the trivial baseline. The inputs do not determine the output                                                                                                                                                                                                                                                                                           |
| **Their GEO file**                       | **nothing works, and that is the finding.** A predictor handed the answer key's own average improves on doing nothing by **0.3%**. The file's 24 largest jumps alternate sign 24 times out of 24, across all four channels at once, which no satellite can do. We reproduce the failure on demand by injecting that artefact into clean data: 0.017 m becomes 0.743 m |

We also rebuilt the competition's scorer. Their reference file does not match the
test they name: it is **Shapiro-Francia**, not Shapiro-Wilk. The metric is
scale-invariant, so it grades the _shape_ of the errors and not their size, and at
six test points a flawless model still averages 0.92.

---

## Run it

```bash
# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh                # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"     # Windows

# 2. reproduce the exact environment (uv.lock is committed)
uv sync

# 3. seven days in, day 8 out
uv run python predict.py data/DATA_MEO_Train2.csv --orbit MEO
```

Output is a CSV covering the 24 hours after the last input row, on a 15-minute
grid, with a 1-sigma uncertainty on every column. Use `--at times.csv` to predict
at supplied timestamps instead, and `--orbit GEO` for geosynchronous data.
Duplicates, gaps, unsorted rows and irregular sampling are cleaned automatically.

```bash
uv run python clean.py                 # the loader, and exactly what it cleans
uv run python sw_score.py              # our scorer, checked against their reference file
uv run python experiments.py           # all eight experiments, printed as tables
uv run python figures/make_figures.py  # all 22 figures
```

---

## What to read, in order

|     |                                        |                                                               |
| --- | -------------------------------------- | ------------------------------------------------------------- |
| 1   | [`STORYLINE.md`](STORYLINE.md)         | The whole argument in ten minutes. **Start here.**            |
| 2   | [`01_FULL_STORY.md`](01_FULL_STORY.md) | The same story with every figure, table and number behind it  |
| 3   | [`PHYSICS.md`](PHYSICS.md)             | The derivations, each equation followed by plain English      |
| 4   | [`STAGE_04.ipynb`](STAGE_04.ipynb)     | All of it as a runnable notebook, executed, with inline plots |
| 5   | [`02_PITCH.md`](02_PITCH.md)           | Slide-by-slide, one image each                                |

## Try it yourself

Open **[`interactive.html`](interactive.html)** in any browser. No install, no server.

Four presets, each a claim about the _data_ rather than the model:

- **A MEO satellite.** Two rhythms in x and y, and neither is the orbital period
- **Almost no data.** 90% of the week missing, still solved
- **Not a satellite.** Physically impossible data, and the model declines to fit it
- **Nothing to find.** No rhythm exists, so averaging is the correct answer

## Files

|                           |                                                       |
| ------------------------- | ----------------------------------------------------- |
| `model.py`                | the model                                             |
| `predict.py`              | command line: seven days in, day 8 out                |
| `clean.py`                | loading and cleaning any 7-day file, delivered or not |
| `synth.py`                | the physics-based data generator                      |
| `sw_score.py`             | the competition's metric, reverse-engineered          |
| `experiments.py`          | all eight experiments                                 |
| `figures/make_figures.py` | all 22 figures                                        |
| `data/`                   | the delivered files, unmodified                       |

## Notes

All input is derived from the given dataset or a signal generated from orbital mechanics.

The model holds no trained weights. It fits itself inside whatever seven-day
window it is handed, so an unseen satellite is not a distribution shift.
