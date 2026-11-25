# MS pack (Qwen3.8-Flash-Next)

The MS pack (`Qwen3.8-FN-MS-60.basalt`, [format](../ms-trellis-qwen4exp.md)) codes the routed experts MS_C4 on 36 layers and
MS_C3 on 12, the dense spine and the head MS_C6; the n-gram table and the hyper-connections are the IQ4_XS pack's. Gate:
100 chunks, T 3, int8 KV, one RTX 3090 at 290 W, against a local BF16 base (the BF16 checkpoint layer by layer,
`ms_flashnext_eval.py --write-base`; wikitext-2 base PPL 3.2756, gsm8k 1.4628, chat 2.9924 over 64 windows of 512 tokens
from `eval_chat.pt`).

| Pack | experts | wiki2 KLD | wiki2 top-1 | gsm8k KLD | gsm8k top-1 | chat KLD | chat top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MS v3 (60 GB plan, chat calibration) | 53.2 GiB | 0.0614 | 92.04% | 0.0323 | 96.81% | 0.0723 | 95.38% |
| MS (60 GB plan, web calibration) | 53.2 GiB | 0.0741 | 91.13% | 0.0356 | 96.55% | | |
| UD-IQ4_XS | 56.2 GiB | 0.0835 | 90.69% | 0.0402 | 96.39% | 0.1087 | 93.14% |
| EXL3 3.68 bpw (experts 3.5 / 4 bits, dense 7, head 6) | 50.8 GiB | 0.0682 | 91.50% | 0.0352 | 96.50% | 0.1000 | 94.09% |
| WV-v5 3.5 bpw | | 0.2400 | 84.15% | 0.1144 | 93.33% | | |

qbench (22 conversations of exllamav3's `qbench_prompts.json`, 45,588 response tokens with thinking and tool calls;
[quality](quality.md#the-bf16-reference)), same gate settings. The EXL3 3.68 bpw pack reads 0.0090 here against 0.0087
published for it, so the gate matches the exllamav3 harness. Paired over conversations, EXL3 3.68 minus MS v3: dKLD
+0.0009 [+0.0005, +0.0018] (MS v3 lower in 19 of 22); IQ4_XS minus MS v3: +0.0052 [+0.0037, +0.0072] (21 of 22).

| Pack | KLD mean | median | p99 | top-1 | PPL (BF16 1.3850) |
| --- | --- | --- | --- | --- | --- |
| MS v3 | 0.00811 | 0.00042 | 0.0884 | 97.20% | 1.3922 |
| EXL3 3.68 bpw | 0.00903 | 0.00048 | 0.0932 | 97.17% | 1.3957 |
| UD-IQ4_XS | 0.01332 | 0.00077 | 0.1432 | 96.44% | 1.4036 |

MS v3 (`Qwen3.8-FN-MS-60-v3.basalt`) changes only the converter's inputs and search: 2560 x 512 calibration tokens
(on-policy Flash-Next chat with thinking 60%, agent and chat traces 20%, web 20%; `build_calib_pool.py --dspine`) instead
of 256 web sequences, each expert's Hessian shrunk toward the layer's (`--hess-tau 256`), and Viterbi window 64 / commit
8 (`--window 64`). Conversion: 7.5 h. Same format, plan and size as MS.

The dense MS GEMV of the decode windows runs on the tensor cores (`ms_mmvq.cu`): the 1MAD codes are exact in fp16, the
rotated input columns are staged as fp16 (each divided by its largest magnitude), the sums are fp32. Against the fp32
GEMV before it, paired on wikitext (100 chunks): dKLD +0.0005 [-0.0006, +0.0016], same top-1 -0.26 points [-0.51, -0.02];
decode +7 to +10% on short prompts (one RTX 3090 at 290 W). An exact variant (fp16 hi + lo pairs, two products) gave
about half of that gain.

The CPU tier's fp32 activations and the prompt path's Q8_0 experts against the fp16 / fp16-GEMM build before them, paired on
wikitext: dKLD +0.0006 [-0.0009, +0.0021].
