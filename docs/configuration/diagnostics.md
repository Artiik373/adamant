# Diagnostics and measurement flags

Flags for measuring, debugging and testing the engine. All are off by default and are not settings.
Binary: **CLI** = `bin/basalt` one-shot (`--tokens IDS`), **serve** = `--serve` engine, **bench** = `basalt-bench`.

## One-shot generation

| Flag | Binary | Default | What it does |
|---|---|---|---|
| `--tokens`, `--tokens-file` | CLI | required | The prompt as token ids. |
| `--max-new` | CLI | 16 | Tokens to generate. |
| `--greedy` | CLI | on | Argmax decoding. |
| `--seed` | CLI | | Sample with this seed under `--top-k` 20, `--top-p` 0.95, `--temperature` 1.0; the same seed gives the same text on every model, and `--temperature 0` stays greedy. |
| `--stop-eos` | CLI | off | Stop at an end-of-turn token. |
| `--eos-ids` | engine | the pack's | The end-of-turn tokens. |
| `--gpu-plan` | engine | | Print the `--gpu`, `--gpu-mode` and `--gpu-share` choice as JSON and exit. |

## Timing and profiles

| Flag | Binary | What it does |
|---|---|---|
| `--stats` | CLI | Per-token wall time, host phases and the pool's breakdown. |
| `--no-pool` | CLI | Skip the CPU expert pool: the GPU-only floor. |
| `--no-capture` | CLI | Run layers directly, not as graphs. Needs `--no-pool`. |
| `--stage-timing` | CLI | Per-stage shares by kernel count. Needs `--no-capture`. |
| `--graph-only` | CLI | Time the per-layer `pre` graphs only. |
| `--gpu-only-full` | CLI | Replay every graph and the head, no pool: the per-token GPU floor. Not with `gpu-mode split`. |
| `--gpu-stages` | CLI | Time each layer's three graphs, print, exit. Not with `gpu-mode split`. |
| `--profile` | serve | GPU stage times per request. Not for A/B timing. |
| `--wait-stats` | serve | Host wait times in the decode loop. |
| `--decode-timing` | serve | One line per request on where decode time went. |
| `--draft-log FILE` | serve | One line per verify window (width, accepted, time), then an `X` line: the window's distinct experts per token prefix, all tiers, CPU pool and staged, summed over the layers. |

## Dumps and speculation tests

| Flag | Binary | What it does |
|---|---|---|
| `--dump-residual`, `--dump-mixed`, `--dump-final-r` | CLI | Write residual or head inputs as f32, for bisecting. |
| `--dump-routing` | CLI | Routed expert ids and weights per layer and position. |
| `--spec-oracle FILE` | CLI | With `--spec N` on a pack without a drafter: draft from an expected continuation. The verify exactness test. |
| `--prefill-until P` | CLI | Batch the first P prompt positions; run the rest teacher-forced. |
| `--spec-corrupt N` | CLI | Replace every Nth draft with a wrong token, to test rejection. |
| `--diag-batch-check` | CLI | With `--batch N` (2 or more) and a prompt: every session's windows alone against the same windows batched, bitwise ([batched decoding](../design/batching.md)). |
| `--diag-gpu-check` | CLI | With `--tokens-file`, `--max-new N` and two or more cards: the prompt and N greedy steps, each step's logits recorded (`--diag-gpu-trace FILE`) or compared with a reference run's (`--diag-gpu-ref FILE`, which feeds the reference's picks so every step sees the same context): bitwise steps, greedy identity, KLD (mean and worst step), top-1 ([several GPUs](../design/multi-gpu.md)). |
| `--diag-gpu-cut L` | CLI, serve | With `gpu-mode split`: the first layer of the second card, in place of the cut the shares make. One flag for every model. |
| `--diag-gpu-same` | serve | Flash-Next with `gpu-mode split` and `--diag-gpu-cut L`: both stages of the split on GPU 0, the split's reference. |
| `--diag-main-card-bits` | CLI, serve | Flash-Next: the other cards' stages run the main card's cuBLAS algorithms and plan by its SM count, so a peer's prompt stages and a split are bitwise against the main card (and `--diag-gpu-same`). A normal run lets each card use its own, which costs nothing and keeps the BF16 KLD level; the flag costs about 0.6% of a 64K prefill. |
| `--diag-fixed-experts` | CLI, serve | Flash-Next: a prompt's experts all run on the GPU and a window's verifiers see none in VRAM, so the CPU pool computes every window row; two placements of the same experts (a split and its `--diag-gpu-same` reference) then compute the same rows. |
| `--diag-teacher` | CLI | With `--spec-oracle`: teacher-forces the file's tokens and counts differing picks. |
| `--diag-tier-frozen` | CLI, serve | Fixed placement: the VRAM tier keeps what it holds, with no copies or promotion. |
| `--diag-drafter-gpu DEV` | CLI, serve | The card the drafter runs on. |
| `--force-zero-doorbell` | serve | Scheduling test on a card that cannot hold every expert. The output is wrong by design. |

## Experts and n-gram table

The per-layer n-gram (PLE) table stays on disk and its rows are read as needed. It is read from the model file unless
`--ple-gguf` names another GGUF that carries it, and either file may hold the table alone or among other tensors. Rows
are whichever of `IQ4_NL`, `Q5_0`, `Q8_0` or `BF16` the file has: 90 B, 110 B, 170 B or 320 B each.

| Flag | Binary | Value, default | What it does |
| --- | --- | --- | --- |
| `--ple-gguf` | engine | path, the model file | Read the table from this GGUF instead of the model file. |
| `--ple-io` | engine | `direct`, `ram`; `direct` | `direct`: unbuffered reads, the table never enters RAM or the file cache. `ram`: map the file with the whole table locked in RAM at start. |
| `--ple-row-cache` | engine | rows, 1048576 | A bounded cache of fetched rows, one row's bytes each; `0` turns it off. |
| `--ple-inflight` | engine | integer, 256 | Outstanding reads. |
| `--ple-delay-us` | engine | microseconds, 0 | Fault injection: each row read completes no earlier than this. |

## Quality gate and bench

| Flag or key | Binary | Default | What it does |
|---|---|---|---|
| `--quality` | bench | 20 chunks | Run the PPL/KLD/top-1 gate through decode. [method](../precision/quality.md) |
| `--kld-base FILE` | bench | | The llama.cpp KL-divergence base. |
| `--quality-t N` | bench | 3 | Tokens per verify window in the gate. |
| `--ppl-base`, `--ppl-out` | serve | | Base file for `ppl=` requests, and where they write rows. |
| `ppl=K`, `ppl_t=T` | request keys | | Score chunk K; window width T. |

The request keys go on a `GEN` line of the engine's stdin protocol, before the token ids.

| Bench flag | Default | What it does |
|---|---|---|
| `-t conv,struct,prose` | | Decode fixtures, run in that order. |
| `-p DEPTHS` | `-p 16384` with no `-t` or `-p` | Prefill depths, e.g. `32k`. |
| `-r N` | 1 | Repetitions; above 1 the table shows mean and deviation. |
| `--seed N` | | Seed every decode request. |
| `--set "FLAG VALUE"` | | Replace or add engine arguments (repeatable). |
| `--engine PATH`, `--fixtures DIR` | `basalt` beside the bench, `data/bench` | Binary and fixtures. |
| `-o md` or `json` | md | Output format. |
| `--power-label W` | read from the GPU | The power limit to print. |
| `--logs DIR`, `--no-windows`, `--print-command` | | Logs, window stats, command printing. |
| `--hardware` | | Measure CPU, RAM and each GPU's link, then exit. |
| `--community`, `--submitter NAME`, `--community-dir DIR` | | The community preset and its row, which names the model ([community runs](../community.md)). |

`basalt-device`, a helper in the build directory, takes `--list-devices` (every GPU the runtime sees and whether this
build runs on it), `--link-rates` (one `pcie_h2d BUS_ID GB_PER_S` line per GPU, which `--hardware` reads) and
`--selftest`; with no flag it prints the main GPU and the memory plan.

## Experiment keys (`--exp`)

`--exp NAME=VALUE[,...]` sets experiment switches. They carry no stability promise and go away with their experiment.

- `concur9_logit_hash=1`: print a hash of each accepted row's head logits on stderr.
- `spec_log=1`: with `--draft-log`, also write each row's target distribution.
- `expert_trace=N`: with `--draft-log FILE`, write `FILE.trace` for the first N windows: each layer's experts, tiers and FFN input rows, after the 4-bit router tables. The layout is in `decode/verify/execution.cpp`.
- `host_ev=1`: print GPU time per window's verify, commit and draft, and the idle gap.
