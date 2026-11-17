"""The quality gate's BF16 reference from the original safetensors, one module at a time.

Runs the unquantized model through exllamav3's module-by-module forward (the converter's own loop, nothing quantized),
so only the hidden states of the evaluation chunks stay in memory and no BF16 GGUF is needed. Writes the base file
`include/basalt/program/ppl_gate.hpp` reads: "_logits_", n_ctx (u32), n_vocab, n_chunk (i32), the chunk tokens, then
per chunk the scored rows (positions n_ctx/2 .. n_ctx-2): scale, min log-prob (f32) and n_vocab uint16 log-probs.

Needs exllamav3 with the model's architecture; run it with that environment's Python:
  python tools/bf16_reference.py --model DIR --text wiki.test.raw --out base.kld [--chunks 100] [--ctx 512]
"""
import argparse
import struct
import sys
import time

import numpy as np
import torch


def tokenize(model_dir: str, text_path: str):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    text = open(text_path, encoding="utf-8").read()
    return tok.encode(text, add_special_tokens=False).ids


def quantize_row(logits: torch.Tensor):
    """fp32 log-softmax of one row as (scale, min, uint16[n]); terms at -16 or below decode to the minimum"""
    lp = torch.log_softmax(logits.float(), dim=-1)
    lo = max(float(lp.min()), -16.0)
    hi = float(lp.max())
    scale = (hi - lo) / 65535.0 if hi > lo else 1.0
    q = torch.clamp(torch.round((lp.clamp(min=lo) - lo) / scale), 0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
    return scale, lo, q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args()

    from exllamav3 import Config, Model

    ids = tokenize(a.model, a.text)
    n_chunk = min(a.chunks, len(ids) // a.ctx)
    chunks = [torch.tensor(ids[c * a.ctx:(c + 1) * a.ctx], dtype=torch.long).unsqueeze(0) for c in range(n_chunk)]
    print(f"{len(ids)} tokens, {n_chunk} chunks of {a.ctx}", flush=True)

    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    device = torch.device("cuda:0")
    state = [c.clone() for c in chunks]
    preserve = [{} for _ in chunks]
    f = open(a.out, "wb")
    f.write(b"_logits_")
    head_written = False
    nll, rows = 0.0, 0
    t0 = time.time()
    with torch.inference_mode():
        for idx, module in enumerate(model.modules):
            last = idx == len(model.modules) - 1
            assert module.num_slices <= 1, f"{module.key}: sliced modules are not handled"
            defer = module.can_defer_load()
            if defer:
                module.config.stc.begin_deferred_load()
            try:
                module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
            finally:
                if defer:
                    module.config.stc.end_deferred_load()
            for i in range(n_chunk):
                params = {"attn_mode": "flash_attn_nc", "input_ids": chunks[i]}
                params.update(preserve[i])
                params["quant_preserve"] = dict(preserve[i])
                x = module.prepare_for_device(state[i], params)
                y = module.forward(x, params)
                preserve[i] = params["quant_preserve"]
                if last:   # the head: write the chunk's scored rows n_ctx/2 .. n_ctx-2 now
                    lg = y[0, a.ctx // 2:a.ctx - 1].float()
                    n_vocab = lg.shape[-1]
                    if not head_written:
                        f.write(struct.pack("<Iii", a.ctx, n_vocab, n_chunk))
                        for c in chunks:
                            f.write(c.squeeze(0).to(torch.int32).numpy().tobytes())
                        head_written = True
                    tgt = chunks[i][0, a.ctx // 2 + 1:a.ctx]
                    for r in range(lg.shape[0]):
                        scale, lo, q = quantize_row(lg[r])
                        f.write(struct.pack("<ff", scale, lo))
                        f.write(q.tobytes())
                        if n_vocab % 2:
                            f.write(bytes(2))
                        nll -= float(torch.log_softmax(lg[r], dim=-1)[tgt[r]])
                        rows += 1
                    state[i] = None
                else:
                    state[i] = y.cpu()
            module.unload()
            torch.cuda.empty_cache()
            # the loader keeps every shard it read open; release them (reopened on demand) or the open-file limit is hit
            for fn in list(config.stc.handles):
                config.stc.release_file(fn)
            print(f"  {idx + 1}/{len(model.modules)} {module.key}  {time.time() - t0:.0f} s", flush=True)

    f.close()
    print(f"wrote {a.out}: n_vocab {n_vocab}, {rows} scored rows, PPL {np.exp(nll / rows):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
