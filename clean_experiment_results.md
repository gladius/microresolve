# ASV Clean Pipeline — Experiment Results

Scenarios: 30 (138 turns) + 10 new (43 turns)
Boundary patterns: 395

---

## BASELINE: Phase 2 (Paraphrase + Routing + Learning)


## Baseline (Phase 2)
**Pipeline:** Paraphrase + Routing + Learning

**Second Pass Results:**

- Exact match: 73.2%
- Top-5 recall: 80.4%
- Avg FP/turn: 0.12

**By word count:**

| Words | Total | Exact% | Recall% |
|---|---|---|---|
| 1-5 | 15 | 80.0% | 93.3% |
| 6-10 | 11 | 90.9% | 100.0% |
| 11-20 | 25 | 72.0% | 92.0% |
| 21-40 | 70 | 74.3% | 75.7% |
| 41+ | 17 | 52.9% | 58.8% |

**New scenarios:** 2.3% exact, 34.9% recall

**Corrections applied:** 372

## Experiment A: Phase 2 + Correction
**Pipeline:** Correction -> Paraphrase + Routing + Learning

**Second Pass Results:**

- Exact match: 73.2%
- Top-5 recall: 81.9%
- Avg FP/turn: 0.14

**By word count:**

| Words | Total | Exact% | Recall% |
|---|---|---|---|
| 1-5 | 15 | 73.3% | 86.7% |
| 6-10 | 11 | 90.9% | 100.0% |
| 11-20 | 25 | 76.0% | 96.0% |
| 21-40 | 70 | 77.1% | 81.4% |
| 41+ | 17 | 41.2% | 47.1% |

**New scenarios:** 2.3% exact, 32.6% recall

**Corrections applied:** 366

**Delta vs baseline:** 0.0% (neutral)

## Experiment B: Phase 2 + Boundary Segmentation
**Pipeline:** Boundary Split -> [per segment: Paraphrase + Routing] -> Merge + Learning

**Second Pass Results:**

- Exact match: 70.3%
- Top-5 recall: 87.0%
- Avg FP/turn: 0.25

**By word count:**

| Words | Total | Exact% | Recall% |
|---|---|---|---|
| 1-5 | 15 | 80.0% | 93.3% |
| 6-10 | 11 | 90.9% | 100.0% |
| 11-20 | 25 | 72.0% | 92.0% |
| 21-40 | 70 | 67.1% | 87.1% |
| 41+ | 17 | 58.8% | 64.7% |

**New scenarios:** 2.3% exact, 32.6% recall

**Corrections applied:** 383

**Delta vs baseline:** -2.9%

**Segmentation impact by word count (Experiment B vs Baseline):**

| Words | Baseline Exact% | Exp B Exact% | Delta |
|---|---|---|---|
| 1-5 | 80.0% | 80.0% | +0.0% |
| 6-10 | 90.9% | 90.9% | +0.0% |
| 11-20 | 72.0% | 72.0% | +0.0% |
| 21-40 ** | 74.3% | 67.1% | -7.1% |
| 41+ ** | 52.9% | 58.8% | +5.9% |

**Segmentation stats:** 93/138 turns segmented, avg 2.6 segments/turn

## Experiment C: Full Clean Pipeline
**Pipeline:** Correction -> Boundary Split -> [per segment: Paraphrase + Routing] -> Merge + Learning

**Second Pass Results:**

- Exact match: 63.0%
- Top-5 recall: 79.7%
- Avg FP/turn: 0.26

**By word count:**

| Words | Total | Exact% | Recall% |
|---|---|---|---|
| 1-5 | 15 | 73.3% | 86.7% |
| 6-10 | 11 | 90.9% | 100.0% |
| 11-20 | 25 | 68.0% | 92.0% |
| 21-40 | 70 | 61.4% | 77.1% |
| 41+ | 17 | 35.3% | 52.9% |

**New scenarios:** 0.0% exact, 25.6% recall

**Corrections applied:** 389

**Delta vs baseline:** -10.1%
**Delta vs Exp A (correction only):** -10.1%
**Delta vs Exp B (boundary only):** -7.2%

---

## Experiment D: Dual-Source Confidence Signal
**Analysis of existing Phase 2 data from mesh_experiment_turns.json**

**Detection source analysis (Phase 2, second pass):**

| Source | Total Detections | Correct | TPR |
|---|---|---|---|
| Both (dual-source) | 96 | 96 | 100.0% |
| Routing only | 72 | 53 | 73.6% |
| Paraphrase only | 39 | 38 | 97.4% |

**Dual-source confidence lift:** 1.4x over routing-only

**FINDING:** Dual-source detections are 95%+ accurate. Can be used as high-confidence auto-route signal. Single-source detections should be escalated for human review.

---

## Experiment E: Cumulative Learning (3 passes)
**Phase 2 architecture, 30 scenarios run 3 times with continuous learning**

| Pass | Exact% | Top-5 Recall% | Avg FP/turn | Corrections (cumulative) |
|---|---|---|---|---|
| Pass 1 | 8.0% | 64.5% | 2.21 | 358 |
| Pass 2 | 71.0% | 83.3% | 0.17 | 402 |
| Pass 3 | 67.4% | 74.6% | 0.09 | 449 |
| Generalization | 67.4% | 72.5% | 0.07 | 449 |

**After 3 passes new scenarios:** 4.7% exact, 37.2% recall
**Total corrections applied:** 449

**Learning trajectory:** Does the system keep improving or plateau/degrade across passes?

---

## Experiment F: Score Ratio Analysis
**Analysis of score ratios from existing Phase 2 data**


### Phase 1 Score Ratios

**Score ratio distribution (correct/FP) for mixed turns:**

| Ratio Range | Count | Cumulative% |
|---|---|---|
| 0.0-1.0x | 1 | 10.0% |
| 1.0-1.5x | 0 | 10.0% |
| 1.5-2.0x | 0 | 10.0% |
| 2.0-3.0x | 2 | 30.0% |
| 3.0-5.0x | 1 | 40.0% |
| 5.0-10.0x | 2 | 60.0% |
| 10.0x+ | 4 | 100.0% |

**Clean turns (correct only):** 124
**Mixed turns (correct + FP):** 10
**Ratio stats:** median=8.65x, mean=7.64x, P25=2.73x, P75=10.71x

**Threshold finding:** 90.0% of mixed turns have ratio >= 2.0x
A 2.0x ratio threshold could separate clean from noisy results.

**FP score distribution:** median=1.35, P90=3.29 (N=18)

### Phase 2 Score Ratios

**Score ratio distribution (correct/FP) for mixed turns:**

| Ratio Range | Count | Cumulative% |
|---|---|---|
| 0.0-1.0x | 1 | 7.7% |
| 1.0-1.5x | 0 | 7.7% |
| 1.5-2.0x | 0 | 7.7% |
| 2.0-3.0x | 0 | 7.7% |
| 3.0-5.0x | 4 | 38.5% |
| 5.0-10.0x | 3 | 61.5% |
| 10.0x+ | 5 | 100.0% |

**Clean turns (correct only):** 121
**Mixed turns (correct + FP):** 13
**Ratio stats:** median=7.25x, mean=8.32x, P25=3.63x, P75=11.40x

**Threshold finding:** 92.3% of mixed turns have ratio >= 2.0x
A 2.0x ratio threshold could separate clean from noisy results.

**FP score distribution:** median=2.04, P90=3.29 (N=20)

---

## COMPARISON SUMMARY

### Second Pass Results (30 scenarios, generalization)

| Experiment | Exact% | Top-5 Recall% | Avg FP/turn | Delta vs Baseline |
|---|---|---|---|---|
| Baseline (Phase 2) | 73.2% | 80.4% | 0.12 | +0.0% |
| A: +Correction | 73.2% | 81.9% | 0.14 | +0.0% |
| B: +Boundary | 70.3% | 87.0% | 0.25 | -2.9% |
| C: +Correction+Boundary | 63.0% | 79.7% | 0.26 | -10.1% |

### By Word Count — Segmentation Focus

| Words | Baseline | Exp B (Boundary) | Exp C (Full) | B Delta | C Delta |
|---|---|---|---|---|---|
| 1-5 | 80.0% | 80.0% | 73.3% | +0.0% | -6.7% |
| 6-10 | 90.9% | 90.9% | 90.9% | +0.0% | +0.0% |
| 11-20 | 72.0% | 72.0% | 68.0% | +0.0% | -4.0% |
| 21-40 | 74.3% | 67.1% | 61.4% | -7.1% | -12.9% |
| 41+ | 52.9% | 58.8% | 35.3% | +5.9% | -17.6% |

### New Scenarios (10 unseen)

| Experiment | Exact% | Top-5 Recall% |
|---|---|---|
| Baseline | 2.3% | 34.9% |
| A: +Correction | 2.3% | 32.6% |
| B: +Boundary | 2.3% | 32.6% |
| C: Full Clean | 0.0% | 25.6% |
