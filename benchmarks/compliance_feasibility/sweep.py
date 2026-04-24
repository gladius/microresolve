"""Threshold sweep on existing baseline + benign CSVs.

For each namespace and threshold, compute:
  - precision (of in-domain queries classified above threshold, how many correct)
  - recall (of in-domain GT, how many caught above threshold and correct class)
  - FP rate on benign (how many benign queries fire any class above threshold)

Critical class recall (mental_health_crisis, prohibited classes) tracked separately.
"""
import csv, collections, math

baseline_rows = list(csv.DictReader(open("/tmp/compliance_v2/baseline.csv")))
benign_rows   = list(csv.DictReader(open("/tmp/compliance_v2/benign.csv")))

CRITICAL = {
    "hipaa-triage": {"mental_health_crisis", "clinical_urgent"},
    "eu-ai-act-prohibited": {
        "biometric_categorization","social_scoring","emotion_recognition_workplace",
        "predictive_policing","subliminal_manipulation","exploitation_vulnerability"},
    "colorado-consequential": {"healthcare_decision"},
}

THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def wilson(k, n, z=1.96):
    if n == 0: return (0,0)
    p = k/n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (centre-half, centre+half)

def sweep(ns):
    in_dom = [r for r in baseline_rows if r["namespace"]==ns]
    ben    = [r for r in benign_rows if r["namespace"]==ns]

    print(f"\n=== {ns} ===")
    print(f"{'thr':>5} {'acc':>6} {'crit_rec':>10} {'crit_CI':>14} {'fp_benign':>10}")
    for T in THRESHOLDS:
        correct = sum(1 for r in in_dom if float(r["score"])>=T and r["pred"]==r["gt"])
        total   = len(in_dom)
        acc     = correct/total if total else 0

        crit = [r for r in in_dom if r["gt"] in CRITICAL[ns]]
        crit_caught = sum(1 for r in crit if float(r["score"])>=T and r["pred"]==r["gt"])
        crit_rec = crit_caught/len(crit) if crit else 0
        lo,hi = wilson(crit_caught, len(crit))

        fp = sum(1 for r in ben if float(r["score"])>=T and r["pred"])
        fp_rate = fp/len(ben) if ben else 0

        print(f"{T:>5.1f} {acc:>6.3f} {crit_rec:>10.3f} [{lo:.2f},{hi:.2f}] {fp_rate:>10.3f}")

for ns in ["hipaa-triage","eu-ai-act-prohibited","colorado-consequential"]:
    sweep(ns)
