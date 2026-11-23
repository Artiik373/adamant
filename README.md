# Adamant

Adamant is a fork of [Basalt](https://github.com/jesdga95/basalt), itself based on Strata: a local inference engine for
large mixture-of-experts models (**Hy4-flash**, **Qwen3.8-Flash-Next**) on NVIDIA **Ampere** (RTX 30-series, sm_86) GPUs.
Ada Lovelace (sm_89) still builds and runs, but Ampere is the target every change is measured on.

> **Status: early development.** Expect breakage; flags, defaults and the pack layout can still change. The numbers below
> are development runs on one card, not the maintainer's headline numbers. Open validation work is listed in
> [Ampere execution and validation](docs/precision/ampere.md). Linux and Windows platform support is retained.

Experts that do not fit in VRAM live in system RAM, and optionally on a second GPU used as a peer expert tier.
A large MoE model runs on one workstation, behind an OpenAI- and Anthropic-compatible API.

<p align="center"><img src="docs/img/frontend.gif" alt="The web UI while a reply streams" width="720"></p>

**Help wanted:** have some spare compute to lend for testing? See [Contributing](CONTRIBUTING.md#other-ways-to-help).

<p align="center">
  <a href="https://github.com/sponsors/jesdga95"><img src="assets/donate/github.svg" height="44" alt="Sponsor on GitHub"></a>
  <a href="https://ko-fi.com/jesdga"><img src="assets/donate/kofi.svg" height="44" alt="Support on Ko-fi"></a>
  <a href="https://paypal.me/jesdga"><img src="assets/donate/paypal.svg" height="44" alt="Donate on PayPal"></a>
</p>

## Requirements

- Linux x86-64 with AVX2 (AVX-512 is used when present), or Windows 10 1903+ x64 ([platforms](docs/platforms.md)).
- An NVIDIA Ampere (compute capability 8.6) GPU with CUDA 13; Ada Lovelace (8.9) also builds.
- CMake 3.24 or newer, GCC 13 or newer (Linux), Python 3.10 or newer, and enough system RAM for the experts that do not fit in VRAM.

## Quick start

```sh
git clone <adamant remote> && cd adamant
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
                                        # Windows: add -G Ninja to the first line, from an x64 VS 2022 prompt ([platforms](docs/platforms.md#building-on-windows))
                                        # one model only: -DBASALT_MODELS=qwen4exp or hy4flash ([building](docs/building.md#building-only-some-models))
pip install -r requirements.txt
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt --local-dir models
```

The build targets sm_86 by default; `-DCMAKE_CUDA_ARCHITECTURES` accepts `86`, `89` or `86;89`, and anything else is
refused. What the port changes against upstream and what is still open is in
[Ampere execution and validation](docs/precision/ampere.md).

The binary and file-format names stay `basalt` and `.basalt`: the packs are the
same files Basalt reads, so a fork that renamed them would break every existing
model. Adamant is the project name, not a new format.

Put your settings in `basalt_config.json`: the `models` list names the packs, and each entry's own `gpu` and `kv` win over the
top-level `engine` block ([models and the router](docs/configuration/models.md)):

```json
{
  "version": 2,
  "models": [{ "model": "models/Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt", "gpu": "0,1", "kv": "int8" }],
  "engine": { "expert-cache": "auto", "prefill": "auto", "resident-experts": true },
  "server": { "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1. Every setting: [configuration](docs/configuration/index.md). More examples:
[example configurations](docs/configuration/examples.md).

## Measured on Ampere

Development measurements, not the maintainer's headline runs: one RTX 3090 (24 GB) at a **304 W** power limit,
`basalt-bench` fixtures, expert cache auto, the rest of the experts in page-locked RAM.

| Pack | Prefill 16k | Decode conv | struct | prose |
| --- | ---: | ---: | ---: | ---: |
| Hy4-flash AD-IQ2_S-IQ2_XXS, DFly2 drafter on | 1,377 t/s | 50 tok/s | 81 tok/s | 46 tok/s |

Draft acceptance 70%, 99.9% and 58%: the spread is drafting acceptance per content, not engine speed.

On Ada (RTX 4090 at 450 W, the GSQ-RCO IQ3_S Qwen3.8-Flash-Next pack, int8 KV), MTP drafter off → on: conv 115 → 190,
struct 116 → 372, prose 116 → 169 tok/s. An Adamant-versus-Strata comparison from those runs is in
[the comparison report](docs/benchmarks/strata-comparison.md).

## Models

| | Hy4-flash | Qwen3.8-Flash-Next |
|---|---|---|
| Packs | built locally from a `hy_v4_flash` GGUF with `tools/convert/convert.py --hf-dir` ([pack](docs/models/hy4-flash/pack.md)) | [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) |
| Context | default 8,192; `--max-context` sets it | default 8,192; `--max-context` sets it |
| Drafting | DFly2 drafter, appended with `convert.py --dfly FILE --into PACK`; `--spec` default 6 | MTP drafter in the pack, on; `--spec` default 8 |
| Guide | [guide](docs/models/hy4-flash.md) | [guide](docs/models/qwen38-flash-next.md) |
| Precision | | [precision](docs/precision/qwen38-flash-next.md) |

All models compared, with each one's context limit, flags and cards: [models](docs/models/index.md).

## Video

Qwen3.8-Flash-Next reads video; Hy4-flash does not. A video is named like an image in a chat message, and needs
`--vision` and FFmpeg on the machine. Flash-Next decodes frames on the GPU with NVDEC; the 27B decodes them on the CPU.
How a request names a video, and the limits: [video](docs/running.md#video).

On Ampere the NVFP4 packs run through the integer MMQ path rather than native FP4
tensor instructions, so their numerics differ from a Blackwell run; the IQ and K-quant
formats use the same kernels as upstream. A pack can be built locally from a GGUF, drafter included
([model file](docs/model-file.md), [Ampere execution](docs/precision/ampere.md)).

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [models and the router](docs/configuration/models.md),
  [sampling](docs/configuration/sampling.md), [image sources](docs/configuration/image-sources.md),
  [documents](docs/configuration/documents.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Models:** [models](docs/models/index.md), [Hy4-flash](docs/models/hy4-flash.md), [Flash-Next](docs/models/qwen38-flash-next.md),
  [the `.basalt` file](docs/model-file.md), [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build and develop:** [building](docs/building.md), [adding a model](docs/adding-a-model.md), [community runs](docs/community.md),
  [changelog](docs/changelog/index.md), [contributing](CONTRIBUTING.md).

## Credits

- [Adamant](https://github.com/nixcat-dev/adamant): the OG Adamant (MIT).
- [Basalt](https://github.com/jesdga95/basalt): the project Adamant forks (MIT).
- [Strata](https://github.com/Niko1221/Strata): the project Basalt forks (MIT).
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the model. The packs are under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), not Basalt's MIT license.
- [unsloth](https://huggingface.co/unsloth): the UD and Q8_0 GGUFs the packs are built from.
- IST Austria DAS Lab: the GSQ-RCO quants; please cite [GSQ](https://arxiv.org/abs/2604.18556) and [RCO](https://arxiv.org/abs/2605.00649).
- [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4): the NVFP4 experts.
- [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp): quantization formats and kernels extracted into `src/` (MIT, see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)).

## License

MIT, see [LICENSE](LICENSE).
