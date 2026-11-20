# Precision

**Adamant:** the port's paths on Ampere (sm_86) and their validation state are in
[Ampere execution and validation](ampere.md).
The optional [DFlash2 drafter](dflash2.md) has its own precision and validation record.

Adamant's own measurements are in the [README](../../README.md#measured-on-ampere). The pages below describe upstream
Basalt on Blackwell; their setup and numbers are not Adamant results.

Each model has one page for what it computes in and what each rounding costs. The method is shared.

- [Qwen3.8-Flash-Next](qwen38-flash-next.md): stages, prefill, the measured costs, drafting and the lossy flags.
- [How quality is measured](quality.md): the gate, the BF16 reference, greedy identity and reproduction commands.
