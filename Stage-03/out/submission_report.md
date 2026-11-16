# PS-08 submission report — SMART-HORIZON

Scored with `sw_score.swtest`, calibrated against the organisers'
`SW_ReferenceData.xlsx` (their stated p = 0.5840; ours = 0.5838).
Residual is **predicted − test**, per Note.pdf.

`which` records which test MATLAB `swtest.m` would take on that sample:
Shapiro-Francia when kurtosis > 3, Shapiro-Wilk otherwise.

## GEO  (orbit GEO, train n=142, test n=69)

| parameter | n | test | W | p | H | mean (m) | sd (m) | RMS (m) |
|---|---|---|---|---|---|---|---|---|
| x | 69 | SF | 0.8689 | 0.0000 | 1 | +0.0535 | 13.9433 | 13.8420 |
| y | 69 | SF | 0.8076 | 0.0000 | 1 | -1.2299 | 19.7716 | 19.6663 |
| z | 69 | SF | 0.8736 | 0.0000 | 1 | +1.5676 | 10.4946 | 10.5356 |
| clock | 69 | SF | 0.5359 | 0.0000 | 1 | -0.8421 | 15.8959 | 15.8027 |
| **average** | | | **0.7715** | **0.0000** | **1.00** | **-0.1127** | **15.0263** | **14.9616** |

- **Priority 1** — W = **0.7715**, p = 0.0000, H = 4/4 parameters reject normality at α = 0.05.
- **Priority 2** — residual mean -0.1127 m, sd 15.0263 m.
- **Priority 3** — `qq_GEO.png`.
- **Context** — a residual that is *exactly* Gaussian at n = 69 scores a median W of 0.983 (5–95%: 0.963–0.992). That is the ceiling, not 1.0.

95% bootstrap CI on W per parameter: x [0.765, 0.920], y [0.672, 0.885], z [0.768, 0.943], clock [0.355, 0.659]

## MEO-1  (orbit MEO, train n=46, test n=6)

| parameter | n | test | W | p | H | mean (m) | sd (m) | RMS (m) |
|---|---|---|---|---|---|---|---|---|
| x | 6 | SW | 0.8412 | 0.1334 | 0 | -0.0666 | 0.2836 | 0.2673 |
| y | 6 | SW | 0.8881 | 0.3086 | 0 | +0.2055 | 0.1666 | 0.2556 |
| z | 6 | SW | 0.9585 | 0.8081 | 0 | -0.0547 | 0.0899 | 0.0986 |
| clock | 6 | SW | 0.9387 | 0.6491 | 0 | +0.1024 | 0.1036 | 0.1394 |
| **average** | | | **0.9067** | **0.4748** | **0.00** | **+0.0467** | **0.1609** | **0.1902** |

- **Priority 1** — W = **0.9067**, p = 0.4748, H = 0/4 parameters reject normality at α = 0.05.
- **Priority 2** — residual mean +0.0467 m, sd 0.1609 m.
- **Priority 3** — `qq_MEO-1.png`.
- **Context** — a residual that is *exactly* Gaussian at n = 6 scores a median W of 0.920 (5–95%: 0.791–0.980). That is the ceiling, not 1.0.

95% bootstrap CI on W per parameter: x [0.580, 0.911], y [0.565, 0.943], z [0.594, 0.951], clock [0.633, 0.958]

## MEO-2  (orbit MEO, train n=143, test n=18)

| parameter | n | test | W | p | H | mean (m) | sd (m) | RMS (m) |
|---|---|---|---|---|---|---|---|---|
| x | 18 | SF | 0.9193 | 0.1131 | 0 | -0.0062 | 0.2117 | 0.2058 |
| y | 18 | SW | 0.8689 | 0.0171 | 1 | +0.0669 | 0.0683 | 0.0943 |
| z | 18 | SF | 0.7218 | 0.0003 | 1 | +0.0109 | 0.1865 | 0.1816 |
| clock | 18 | SF | 0.6806 | 0.0001 | 1 | +0.0101 | 0.0526 | 0.0521 |
| **average** | | | **0.7977** | **0.0327** | **0.75** | **+0.0205** | **0.1298** | **0.1334** |

- **Priority 1** — W = **0.7977**, p = 0.0327, H = 3/4 parameters reject normality at α = 0.05.
- **Priority 2** — residual mean +0.0205 m, sd 0.1298 m.
- **Priority 3** — `qq_MEO-2.png`.
- **Context** — a residual that is *exactly* Gaussian at n = 18 scores a median W of 0.952 (5–95%: 0.891–0.981). That is the ceiling, not 1.0.

95% bootstrap CI on W per parameter: x [0.754, 0.958], y [0.706, 0.922], z [0.610, 0.936], clock [0.544, 0.941]
