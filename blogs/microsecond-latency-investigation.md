# Why is microsecond latency bimodal? A profiling story

While benchmarking MicroResolve's routing pipeline before launch, the same query against the same namespace consistently produced two distinct clusters of timing samples — fast (~10µs) and slow (~28µs), with rare ~240µs outliers. The 25× spread looked alarming for a system claiming "microsecond" performance.

This post walks through how I diagnosed it. The answer turned out not to be a code problem at all.

## The observation

30 samples of `"create a ticket"` routed against the unified demo namespace, after a 5-call warmup:

```
sorted: [9, 9, 10, 10, 10, 10, 10, 12, 13, 15,
         26, 26, 26, 26, 27, 28, 28, 29, 30, 30,
         31, 32, 32, 32, 36, 36, 37, 38, 40, 240]

min: 9µs    median: 28µs    p90: 38µs    p99: 240µs
```

Two clean clusters and one big outlier. Same query. Same code path. Why?

## The hypotheses

Four candidates, in order of how interesting they'd be if true:

1. **Multi-intent rounds**: routing iterates `score → confirm → consume → re-score`. Some queries take 1 round, others take 3. Different rounds = different total cost.
2. **HashMap variance**: Rust's default `HashMap` uses SipHash for DoS resistance — each lookup is ~100ns of pure hashing. ~10 lookups per query = ~1µs of hash overhead.
3. **AhoCorasick lazy state machine**: some implementations build state on first scan of a particular pattern. First call slow, subsequent fast.
4. **OS scheduler / CPU governor**: process gets descheduled, CPU clocks down between calls.

I'd been quietly assuming #1 (multi-intent rounds) was the cause because the bimodal shape looks like "two different code paths." Time to actually measure.

## The instrumentation

Added per-phase timing to the route handler — entity layer, L0 typo correction, L1 morphology, L2 single-pass scoring, L2 multi-intent rounds — plus the round count for that query:

```rust
let _t_total = std::time::Instant::now();
let _t_phase = std::time::Instant::now();
let query_for_l0 = if let Some(layer) = router.entity_layer() {
    layer.augment(&req.query)
} else { req.query.clone() };
let entity_us = _t_phase.elapsed().as_micros();
// ... same pattern for each subsequent phase ...
eprintln!("[timing] entity={}µs l0={}µs l1={}µs l2_single={}µs l2_multi={}µs(rounds={}) total={}µs",
    entity_us, l0_us, l1_us, l2_single_us, l2_multi_us, n_rounds, total_us);
```

Then 30 calls against the same query, after warmup. Here's a representative slice of the log:

```
[timing] entity=3µs l0=1µs l1=8µs l2_single=4µs l2_multi=7µs(rounds=2) total=26µs   ← slow cluster
[timing] entity=4µs l0=1µs l1=8µs l2_single=5µs l2_multi=8µs(rounds=2) total=28µs   ← slow cluster
[timing] entity=2µs l0=1µs l1=7µs l2_single=4µs l2_multi=7µs(rounds=2) total=22µs   ← slow cluster
[timing] entity=1µs l0=0µs l1=4µs l2_single=2µs l2_multi=3µs(rounds=2) total=12µs   ← FAST cluster
[timing] entity=1µs l0=0µs l1=3µs l2_single=2µs l2_multi=3µs(rounds=2) total=10µs   ← FAST cluster
[timing] entity=0µs l0=0µs l1=3µs l2_single=2µs l2_multi=3µs(rounds=2) total=10µs   ← FAST cluster
[timing] entity=1µs l0=0µs l1=3µs l2_single=1µs l2_multi=3µs(rounds=2) total=10µs   ← FAST cluster
[timing] entity=4µs l0=1µs l1=9µs l2_single=5µs l2_multi=8µs(rounds=2) total=28µs   ← slow cluster
```

## Reading the data

Three observations jump out immediately.

### 1. Rounds is constant — multi-intent rounds is NOT the cause

Every single sample shows `rounds=2`. Hypothesis #1 was wrong. The same query always takes 2 rounds (one to consume the dominant signal, one to verify nothing else clears the threshold).

### 2. Every phase scales together, proportionally

Compare a fast sample to a slow sample:

| Phase | Fast (total 10µs) | Slow (total 28µs) | Ratio |
|-------|-------------------|-------------------|-------|
| entity | 1µs | 4µs | 4× |
| L0 | 0µs | 1µs | — |
| L1 | 3µs | 8µs | 2.7× |
| L2 single | 2µs | 5µs | 2.5× |
| L2 multi | 3µs | 8µs | 2.7× |

If hypothesis #2 (HashMap) or #3 (AhoCorasick lazy state) were right, **one** phase would be slow on the slow samples — the one doing the relevant work. But every phase is slow proportionally.

### 3. The CPU itself is running 2-3× slower for the slow calls

This is the hallmark of CPU frequency scaling. Modern CPUs (Intel Speed Shift, AMD CPPC) drop to low power states when idle and ramp up under load. A sporadic test where calls are seconds apart looks like "always idle" to the governor — clock stays low. Calls that happen to land while the CPU is still spun up from a previous burst execute at full clock.

The 240µs outlier is the same effect plus a scheduler context switch — process briefly de-prioritized, kernel does some bookkeeping, then resumes.

## Verification

Quick test: pin the process to one core and force the performance governor.

```bash
sudo cpupower frequency-set -g performance
taskset -c 2 ./target/release/server
```

Result: bimodal distribution disappears. Median 22µs, p99 35µs. The cluster collapses into one tight band.

That's the smoking gun. **The bimodal shape is CPU power management, not anything in our code.**

## What this means for honest performance numbers

Three different "p99" numbers, all true:

| Setup | p99 |
|-------|-----|
| Default CPU governor, sporadic load | ~250µs |
| Performance governor + pinned core | ~50µs |
| Sustained load (1000+ QPS) | ~50µs (governor stays at full clock) |

For launch material the right thing to publish is the conservative one: **median 25µs, p99 250µs (default kernel/governor)**. If you deploy to a server that actually receives traffic continuously, you'll see closer to the second row.

## The thing I almost did wrong

The instinct when seeing bimodal latency is to **change something in the code** — switch HashMap to BTreeMap, switch SipHash to FxHash, refactor the scoring loop to reduce allocations. Each would be reasonable engineering. None of them would have helped here.

Worse: each "optimization" would have made the next benchmark slightly different (different exact numbers, slightly different distribution shape) and I would have congratulated myself for "fixing" the variance — when really the only thing that changed was when the CPU governor sampled the load.

## How to actually decide if your latency is good

Two rules I'm taking from this:

**1. Always profile per-phase, not just total.** A 10× variance in total latency means something different if every phase moved together (CPU/scheduler) versus one phase being the culprit (algorithm/data structure).

**2. Test under sustained load before changing code.** A single-call benchmark always reflects whatever power state the CPU happened to be in. If the variance disappears under continuous load, the code is fine; the variance is the OS.

Both rules are obvious in hindsight. Both took me an evening of "this can't possibly be this slow" to internalize.

## Honest numbers for MicroResolve

After all of the above, the actual production-relevant latency for MicroResolve's full routing pipeline (with entity layer enabled, multi-intent scoring, on a typical 4-intent query):

- **Median: 25 µs**
- **p90: 40 µs**
- **p99 (cold CPU): 250 µs**
- **p99 (warm CPU under sustained load): ~50 µs**

For an embedded library running inside an AI agent loop, that's the difference between "you can run it on every request" and "you can run it on every request and not notice."

The numbers we publish on the launch post are the conservative ones. Anyone benchmarking us will see better than that under realistic load. That's the right way around.
