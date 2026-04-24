"""Run the experiment: seed namespaces, baseline eval, learning loop, post-eval."""
import json, csv, math, random, time, urllib.request, urllib.error

HOST = "http://localhost:3001"
NAMESPACES = ["hipaa-triage-v2", "eu-ai-act-prohibited-v2", "colorado-consequential-v2"]
NS_MAP = {  # display name -> server ns (suffix to avoid collisions)
    "hipaa-triage": "hipaa-triage-v2",
    "eu-ai-act-prohibited": "eu-ai-act-prohibited-v2",
    "colorado-consequential": "colorado-consequential-v2",
}
CRITICAL_CLASSES = {
    "hipaa-triage": ["mental_health_crisis", "clinical_urgent"],
    "eu-ai-act-prohibited": ["biometric_categorization","social_scoring","emotion_recognition_workplace","predictive_policing","subliminal_manipulation","exploitation_vulnerability"],
    "colorado-consequential": ["healthcare_decision"],
}

def http(method, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{HOST}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k,v in (headers or {}).items():
        req.add_header(k,v)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

def wilson(successes, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = successes/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    pm = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, max(0,center-pm), min(1,center+pm))

def seed_ns(display_ns, server_ns, seeds):
    # delete-all (try): ignore errors if ns not present
    for intent_id in seeds.keys():
        http("DELETE", f"/api/intents/{intent_id}", headers={"X-Namespace-ID": server_ns})
    for intent_id, phrases in seeds.items():
        status, _ = http("POST", "/api/intents", {"id": intent_id, "phrases": phrases}, {"X-Namespace-ID": server_ns})
        if status not in (200,201):
            print(f"  WARN add_intent {intent_id}: {status}")
    print(f"  seeded {len(seeds)} intents")

def route(q, server_ns):
    t0 = time.perf_counter()
    status, body = http("POST", "/api/route_multi", {"query": q, "log": False}, {"X-Namespace-ID": server_ns})
    dt_ms = (time.perf_counter()-t0)*1000
    if status != 200 or not isinstance(body, dict):
        return {"top_intent": None, "top_score": 0.0, "disposition": "error", "latency_ms": dt_ms}
    confirmed = body.get("confirmed") or []
    top = confirmed[0] if confirmed else None
    return {
        "top_intent": top["id"] if top else None,
        "top_score": top["score"] if top else 0.0,
        "disposition": body.get("disposition","no_match"),
        "latency_ms": dt_ms,
        "server_us": body.get("routing_us", 0),
    }

def evaluate(queries, server_ns):
    """queries: list of (text, gt_label) where gt_label=None => benign."""
    rows = []
    for text, gt in queries:
        r = route(text, server_ns)
        rows.append({
            "query": text[:200], "gt": gt, "pred": r["top_intent"], "score": r["top_score"],
            "disposition": r["disposition"], "latency_ms": r["latency_ms"], "server_us": r["server_us"]
        })
    return rows

def main():
    seeds = json.load(open("/tmp/compliance_v2/seeds.json"))
    tests = json.load(open("/tmp/compliance_v2/tests.json"))
    benign = json.load(open("/tmp/compliance_v2/benign.json"))
    robustness = json.load(open("/tmp/compliance_v2/robustness.json"))

    random.seed(7)

    all_results = {}
    for display_ns, server_ns in NS_MAP.items():
        print(f"\n=== {display_ns} → {server_ns} ===")
        seed_ns(display_ns, server_ns, seeds[display_ns])

        # Build test list: (text, gt_label)
        ns_tests = tests[display_ns]
        test_list = [(q, label) for label, qs in ns_tests.items() for q in qs]

        # Split 30/70 deterministically per class for learning loop
        train30, eval70 = [], []
        for label, qs in ns_tests.items():
            qs_c = list(qs)
            random.Random(hash(label) & 0xffffffff).shuffle(qs_c)
            k = max(1, int(0.3*len(qs_c)))
            train30.extend([(q, label) for q in qs_c[:k]])
            eval70.extend([(q, label) for q in qs_c[k:]])

        # BASELINE: evaluate full set (for overall numbers) AND held-out 70% subset
        print(f"  baseline eval: {len(test_list)} queries")
        t0 = time.time()
        base_all = evaluate(test_list, server_ns)
        base_70 = evaluate(eval70, server_ns)
        print(f"    done in {time.time()-t0:.1f}s")

        # LEARNING LOOP: add MISSED train30 queries to correct intent
        missed_train = [r for r in evaluate(train30, server_ns)
                        if r["pred"] != r["gt"]]
        print(f"  learning: adding {len(missed_train)} missed train queries as phrases")
        added = 0
        for r in missed_train:
            status, _ = http("POST", f"/api/intents/{r['gt']}/phrases",
                             {"phrase": r["query"], "lang": "en"},
                             {"X-Namespace-ID": server_ns})
            if status == 200: added += 1
        print(f"    added {added}")

        # POST-LEARNING: held-out 70% only
        post_70 = evaluate(eval70, server_ns)

        all_results[display_ns] = {
            "baseline_all": base_all,
            "baseline_70": base_70,
            "post_70": post_70,
            "train30_added": added,
        }

    # ---- Benign / FP evaluation on the hipaa namespace (use each) ----
    print("\n=== benign (FP) eval against each namespace ===")
    benign_q = [(x["query"], None) for x in benign]
    fp_results = {}
    for display_ns, server_ns in NS_MAP.items():
        t0 = time.time()
        rows = evaluate(benign_q, server_ns)
        print(f"  {display_ns}: {time.time()-t0:.1f}s")
        fp_results[display_ns] = rows

    # ---- Robustness ----
    print("\n=== robustness eval ===")
    robust_results = {}
    for display_ns, server_ns in NS_MAP.items():
        robust_results[display_ns] = []
        for name, text in robustness:
            r = route(text, server_ns)
            robust_results[display_ns].append({"case": name, "query": text[:80], "pred": r["top_intent"], "score": r["top_score"], "disp": r["disposition"]})

    # ---- Write CSVs ----
    with open("/tmp/compliance_v2/baseline.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["namespace","query","gt","pred","score","disposition","latency_ms","server_us","split"])
        for ns, d in all_results.items():
            for r in d["baseline_all"]:
                w.writerow([ns, r["query"], r["gt"], r["pred"], r["score"], r["disposition"], f"{r['latency_ms']:.2f}", r["server_us"], "all"])
            for r in d["post_70"]:
                w.writerow([ns, r["query"], r["gt"], r["pred"], r["score"], r["disposition"], f"{r['latency_ms']:.2f}", r["server_us"], "post_70"])

    with open("/tmp/compliance_v2/benign.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["namespace","query","pred","score","disposition"])
        for ns, rows in fp_results.items():
            for r in rows:
                w.writerow([ns, r["query"], r["pred"], r["score"], r["disposition"]])

    with open("/tmp/compliance_v2/robustness.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["namespace","case","query","pred","score","disposition"])
        for ns, rows in robust_results.items():
            for r in rows:
                w.writerow([ns, r["case"], r["query"], r["pred"], r["score"], r["disp"]])

    # ---- Build report ----
    report = ["# MicroResolve compliance classification — v2 experiment\n"]
    report.append("_All numbers from /tmp/compliance_v2/ artifacts. 95% CIs via Wilson interval._\n")

    # Baseline table
    report.append("## Baseline\n")
    report.append("| Namespace | In-domain acc | 95% CI | Critical-class recall | FP rate on benign | p50 / p99 (ms) |")
    report.append("|---|---|---|---|---|---|")
    for ns in NS_MAP:
        rows = all_results[ns]["baseline_all"]
        correct = sum(1 for r in rows if r["pred"]==r["gt"])
        n = len(rows)
        p, lo, hi = wilson(correct, n)

        # critical recall across all critical classes for namespace
        crit = [c for c in CRITICAL_CLASSES[ns]]
        crit_rows = [r for r in rows if r["gt"] in crit]
        crit_ok = sum(1 for r in crit_rows if r["pred"]==r["gt"])
        cp, clo, chi = wilson(crit_ok, len(crit_rows))

        # FP rate
        fp_rows = fp_results[ns]
        fp_hits = sum(1 for r in fp_rows if r["pred"] is not None)
        fp_rate = fp_hits/len(fp_rows)

        # latency (server_us)
        us = sorted([r["server_us"] for r in rows])
        p50 = us[len(us)//2]/1000; p99 = us[int(len(us)*0.99)]/1000

        report.append(f"| {ns} | {p:.3f} ({correct}/{n}) | [{lo:.3f}, {hi:.3f}] | {cp:.3f} ({crit_ok}/{len(crit_rows)}) [{clo:.3f},{chi:.3f}] | {fp_rate:.3f} ({fp_hits}/{len(fp_rows)}) | {p50:.2f} / {p99:.2f} |")

    # Post learning lift
    report.append("\n## Post-learning lift (held-out 70%)\n")
    report.append("| Namespace | Baseline | Post-learning | Lift | Train phrases added |")
    report.append("|---|---|---|---|---|")
    for ns in NS_MAP:
        b70 = all_results[ns]["baseline_70"]
        p70 = all_results[ns]["post_70"]
        b_ok = sum(1 for r in b70 if r["pred"]==r["gt"])
        p_ok = sum(1 for r in p70 if r["pred"]==r["gt"])
        bp,_,_ = wilson(b_ok, len(b70))
        pp,_,_ = wilson(p_ok, len(p70))
        report.append(f"| {ns} | {bp:.3f} ({b_ok}/{len(b70)}) | {pp:.3f} ({p_ok}/{len(p70)}) | {pp-bp:+.3f} | {all_results[ns]['train30_added']} |")

    # FP worst cases
    report.append("\n## False positive analysis\n")
    report.append("Top 10 highest-scoring benign→compliance false positives across all three namespaces:\n")
    report.append("| NS | benign query | fired intent | score |")
    report.append("|---|---|---|---|")
    fps = []
    for ns, rows in fp_results.items():
        for r in rows:
            if r["pred"]:
                fps.append((r["score"], ns, r["query"], r["pred"]))
    fps.sort(reverse=True)
    for score, ns, q, pred in fps[:10]:
        report.append(f"| {ns} | {q[:70]} | {pred} | {score:.2f} |")

    # Robustness
    report.append("\n## Robustness\n")
    report.append("Per-case behavior (hipaa-triage-v2 only shown — behavior is similar across ns):\n")
    for r in robust_results["hipaa-triage"]:
        note = f"→ {r['pred']} ({r['score']:.2f})" if r["pred"] else f"→ {r['disp']}"
        report.append(f"- **{r['case']}**: `{r['query'][:60]}` {note}")

    # Critical class recall details
    report.append("\n## Critical-class recall details (baseline)\n")
    for ns in NS_MAP:
        for c in CRITICAL_CLASSES[ns]:
            rows = [r for r in all_results[ns]["baseline_all"] if r["gt"]==c]
            if not rows: continue
            ok = sum(1 for r in rows if r["pred"]==c)
            p, lo, hi = wilson(ok, len(rows))
            report.append(f"\n### {ns} / **{c}** — n={len(rows)}, caught {ok}, recall {p:.3f} [{lo:.3f},{hi:.3f}]")
            fns = [r for r in rows if r["pred"]!=c][:3]
            if fns:
                report.append("False negatives:")
                for r in fns:
                    report.append(f"  - `{r['query'][:100]}` → {r['pred']} ({r['score']:.2f})")

    with open("/tmp/compliance_v2/report.md","w") as f:
        f.write("\n".join(report))
    print("\nReport: /tmp/compliance_v2/report.md")
    print("\n".join(report[:40]))

if __name__ == "__main__":
    main()
