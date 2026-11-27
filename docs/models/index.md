# Models

A model is one `.basalt` file. The server reads the architecture from the file and picks the chat format, tool syntax and thinking
switch. Several entries can share one address ([router](../configuration/models.md)). What the model supports:

| | Qwen3.8-Flash-Next | Hy4-flash | Inkling 2 Mini |
| --- | --- | --- | --- |
| Architecture id | `qwen4exp` | `hy4flash` | `inkling2` |
| Packs | default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt), built from Tencent's BF16 checkpoint ([model file](../model-file.md#hy4-flash-nvfp4-pack)) | `Inkling-2-Mini-NVFP4-MX8`, [Hugging Face](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt), built from the official release ([guide](inkling2.md#pack)) |
| Layers, experts | 48 (36 GDN, 12 QSA); 512 per layer, top-10 | 42 (12 sparse MLA with an indexer, 30 sliding-window); layer 0 dense, then 512 per layer, top-16, plus one shared | 24 visited 48 times a token (a core of 12 run three times; 4 full attention, 20 sliding-window); 128 per layer, top-8 |
| Context | default 8,192; `--max-context` sets it | default 8,192; `--max-context` sets it, up to 1,048,576 | default 8,192; `--max-context` sets it, up to 1,048,576 |
| Drafting | MTP drafter in the pack, on; `--spec` default 8 | DFly2 block drafter in the pack, on; `--spec` default 6 | DSpark drafter in the pack, on; `--spec` default 8 |
| Chat, tools, thinking | yes; `<tool_call>` tools | yes; `<tool_calls>` tools | yes; JSON tool calls |
| Reasoning effort | `none`, `low`, `medium`, `high`, `max` (`high` = `max`); a number is refused | `none`, `low`, `medium`, `high`, `max` (all but `none` and `max` are `high`); a number is refused | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or 1 to 100 |
| Images | `--vision dynamic`, a GPU number or `cpu` | `--vision dynamic` only | `--vision dynamic` only |
| Video | with images, and FFmpeg on the machine ([video](../running.md#video)) | no | no |
| `--kv` | fp16 (default), int8, q4_0, k8v4 | fp16 (default), fp8 | fp16 (default), fp8, int8 |
| `gpu-mode peer` | up to 4 cards: experts and prompt layers | up to 4 cards: experts | 2 cards: the experts that spill from the main card |
| `gpu-mode split` | up to 8 cards | refused | 2 cards |
| Cards by default | the main card alone | the main card alone | the main card, experts in RAM when the context needs their room |
| Parallel slots | yes; each slot drafts | not measured | not measured |
| Lossy flags | `--relaxed-drafting`, `--route-prior`, `--route-tail-skip` (off) | none | none |
| Idle sleep, wake | yes (Linux) | yes (Linux) | yes (Linux) |
| `basalt-bench --quality` | yes | yes | yes |
| Guide | [guide](qwen38-flash-next.md) | [guide](hy4-flash.md) | [guide](inkling2.md) |
| Precision | [precision](../precision/qwen38-flash-next.md) | [precision](../precision/hy4-flash.md) | [precision](../precision/inkling2.md) |

The adapter registry in `src/models/builtin.cpp` is the list of architectures the engine can run; a model's own code lives in
`src/models/<model>` ([adding a model](../adding-a-model.md)).

GigaChat4-37B-A4B has its architecture and kernels in the tree but does not run yet ([status](gigachat4.md)).
