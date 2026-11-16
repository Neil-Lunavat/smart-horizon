# Smart Horizon: the whole road

Everything from the GNSS project, start to end: every stage, every dead end, and what each one taught us. The finished
submission is the [`master`](https://github.com/Neil-Lunavat/smart-horizon) branch; it is `Stage-04` here.

## Read in this order

|     |                                                |                                                                    |
| --- | ---------------------------------------------- | ------------------------------------------------------------------ |
| 1   | [`JOURNEY.md`](JOURNEY.md)                     | The exploration log: what we tried, what broke, what survived, why |
| 2   | [`Stage-01/`](Stage-01)                        | Why the delivered data fails                                       |
| 3   | [`Stage-02/`](Stage-02)                        | Task A on real broadcast and precise orbits, and why we left it    |
| 4   | [`STAGE-02-FINDINGS.md`](STAGE-02-FINDINGS.md) | What Stage 2 found                                                 |
| 5   | [`TASK_B_FINAL.md`](TASK_B_FINAL.md)           | Task B, its numbers, and the scripts in this folder                |
| 6   | [`Stage-03/`](Stage-03)                        | Back to the delivered files                                        |
| 7   | [`Stage-04/`](Stage-04)                        | The submission                                                     |
| 8   | [`SOLUTION/`](SOLUTION)                        | The model as a standalone deliverable                              |

`SHIH-TID-264.pdf` and `.pptx` are the deck, and `script shih.md` is what was said over it.

## What is not here

The heavy files, about 930 MB. [`REBUILD.md`](REBUILD.md) says how to make each one again.

| left out                                                 | size   |
| -------------------------------------------------------- | ------ |
| `raw/`: 488 daily navigation and orbit files, downloaded | 529 MB |
| `full2.parquet`                                          | 195 MB |
| `Stage-02/taskA.npz`                                     | 132 MB |
| `Stage-02/handover.parquet`                              | 32 MB  |
| `taskB_data.npz`                                         | 29 MB  |
| `Stage-02/v3_pred.npz`                                   | 11 MB  |
