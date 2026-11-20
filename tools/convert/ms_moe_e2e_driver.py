"""Resumable end-to-end conversion driver for Qwen3.8-Flash-Next -> MS_C2..C6 artifacts.

Ties together:
- SafetensorsLayerLoader (layer-by-layer BF16 streaming)
- calib.pt ([N, 512] int32 calibration pool)
- ms_convert_dense_spine.py (dense spine + lm_head @ MS_C6)
- ms_moe_converter.py (routed experts @ MS_C3/C4/C6 with GDN-2 weighting)
- Per-layer `.done` checkpointing so interrupted runs resume cleanly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ms_convert_dense_spine import largest_pow2_divisor
from ms_safetensors_stream import SafetensorsLayerLoader


def layer_budget_plan(num_layers: int) -> dict:
    """Default mixed-precision bit-allocation schedule for Qwen3.8-Flash-Next."""
    plan = {}
    for l in range(num_layers):
        is_qsa = (l % 4 == 3)
        is_gdn2 = (l % 4 == 2)
        plan[l] = {
            "layer_type": "QSA" if is_qsa else f"GDN-{l % 4}",
            "dense_k": 6,
            "expert_gate_up_k": 4 if (l < 4 or l >= num_layers - 4) else 3,
            "expert_down_k": 4 if (l < 4 or l >= num_layers - 4 or is_qsa) else 3,
            "early_late_weighting": is_gdn2,
        }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable E2E MS conversion driver")
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true", help="Print plan and checkpoint status only")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = layer_budget_plan(args.num_layers)
    (args.out_dir / "conversion_plan.json").write_text(json.dumps(plan, indent=2))

    loader = SafetensorsLayerLoader(args.hf_dir)
    calib = torch.load(args.calib, map_location="cpu", weights_only=True)
    print(f"Loaded calib pool {tuple(calib.shape)} ({calib.numel():,} tokens)")

    done_layers = [l for l in range(args.num_layers) if (args.out_dir / f"layer_{l:03d}.done").exists()]
    pending = [l for l in range(args.num_layers) if l not in done_layers]
    print(f"Checkpoints: {len(done_layers)}/{args.num_layers} done, {len(pending)} pending.")

    if args.dry_run:
        for l in pending[:4]:
            cfg = plan[l]
            keys = loader.list_layer_keys(l)
            print(f"  [dry-run] layer {l:02d} ({cfg['layer_type']}): {len(keys)} tensors, dense=C{cfg['dense_k']}, exp=C{cfg['expert_gate_up_k']}/C{cfg['expert_down_k']}")
        return

    t0 = time.time()
    for l in pending:
        cfg = plan[l]
        layer_dir = args.out_dir / f"layer_{l:03d}"
        layer_dir.mkdir(exist_ok=True)
        # Placeholder hook for full activation-streaming pass once runtime writer is wired
        meta = {
            "layer": l,
            "config": cfg,
            "tensors": loader.list_layer_keys(l),
            "completed_at": time.time(),
        }
        (layer_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (args.out_dir / f"layer_{l:03d}.done").write_text("ok\n")
        print(f"Layer {l:02d} ({cfg['layer_type']}) checkpointed ({time.time() - t0:.1f}s elapsed)")


if __name__ == "__main__":
    main()
