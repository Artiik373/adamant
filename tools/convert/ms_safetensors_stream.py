"""Safetensors layer-by-layer streaming driver for Qwen3.8-Flash-Next MS conversion.

Designed for 1x RTX 3090 (24 GB VRAM) + 64 GB DDR5:
- Reads model.safetensors.index.json and maps only the tensors belonging to layer `l`
  onto GPU (~5.2 GB BF16 per layer), freeing them before layer `l + 1`.
- Accumulates per-expert Hessians (512 experts) in host DDR5 (pinned/paged float32)
  and processes experts in GPU micro-batches (default 32 experts at a time).
- Writes per-layer packed MS binary blobs + JSON manifest for Adamant (.basalt) packing.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open

import ms_moe_converter as ms


class SafetensorsLayerLoader:
    """Lazy, shard-aware loader that pulls one transformer layer at a time from NVMe."""

    def __init__(self, hf_dir: Path) -> None:
        self.hf_dir = hf_dir
        index_path = hf_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing {index_path}")
        index = json.loads(index_path.read_text())
        self.weight_map: Dict[str, str] = index["weight_map"]

    def list_layer_keys(self, layer_idx: int) -> List[str]:
        prefix = f"model.layers.{layer_idx}."
        alt_prefix = f"model.language_model.layers.{layer_idx}."
        return [
            k for k in self.weight_map
            if k.startswith(prefix) or k.startswith(alt_prefix)
        ]

    def load_tensor(self, key: str, device: str = "cpu") -> torch.Tensor:
        shard_path = self.hf_dir / self.weight_map[key]
        with safe_open(shard_path, framework="pt", device=device) as f:
            return f.get_tensor(key)

    def load_layer(self, layer_idx: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        keys = self.list_layer_keys(layer_idx)
        by_shard: Dict[str, List[str]] = {}
        for k in keys:
            by_shard.setdefault(self.weight_map[k], []).append(k)

        tensors: Dict[str, torch.Tensor] = {}
        for shard_name, shard_keys in by_shard.items():
            shard_path = self.hf_dir / shard_name
            with safe_open(shard_path, framework="pt", device=device) as f:
                for k in shard_keys:
                    tensors[k] = f.get_tensor(k)
        return tensors


def quantize_routed_experts_streaming(
    layer_tensors: Dict[str, torch.Tensor],
    expert_h_in: torch.Tensor,      # [num_experts, 2560, 2560] float32 in DDR5
    expert_h_mid: torch.Tensor,     # [num_experts, 640, 640] float32 in DDR5
    expert_counts: torch.Tensor,    # [num_experts] int64 token counts
    layer_idx: int,
    k: int,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, dict]:
    """Quantize all 512 routed experts of layer `layer_idx` in small GPU chunks."""
    lut = ms.build_1mad_lut(device)
    manifest: Dict[str, dict] = {}
    num_experts = int(expert_counts.shape[0])

    for e in range(num_experts):
        n_tok = int(expert_counts[e].item())
        for proj, h_host, blk_size, seed_off in (
            ("gate_proj", expert_h_in[e], 512, 1000),
            ("up_proj", expert_h_in[e], 512, 1000),
            ("down_proj", expert_h_mid[e], 128, 2000),
        ):
            # Match either model.layers.{l}.mlp.experts.{e}.{proj}.weight or language_model prefix
            match_key = next(
                (key for key in layer_tensors if key.endswith(f".experts.{e}.{proj}.weight")),
                None,
            )
            if match_key is None:
                continue

            w = layer_tensors[match_key].to(device=device, dtype=torch.float32)
            h = h_host.to(device=device, dtype=torch.float32)
            h_den = ms.denoise_hessian_rmt(h, n_samples=n_tok)
            seed = seed_off + layer_idx * 1024 + e
            w_rot, h_rot, _ = ms.rotate_weight_and_hessian(w, h_den, block_size=blk_size, seed=seed)
            hu = ms.cholesky_inv_upper(h_rot)

            scales_fp16, states, _ = ms.viterbi_code_matrix(w_rot, hu, k=k, lut=lut)
            blob = ms.pack_per_row_ms(scales_fp16, states, k=k)

            rel_path = f"layer_{layer_idx:02d}_e{e:03d}_{proj}.ms{k}"
            (out_dir / rel_path).write_bytes(blob)
            manifest[match_key] = {
                "file": rel_path,
                "k": k,
                "shape": list(w.shape),
                "hadamard_block": blk_size,
                "hadamard_seed": seed,
                "calib_tokens": n_tok,
            }

            del w, h, h_den, w_rot, h_rot, hu, scales_fp16, states
        if (e + 1) % 32 == 0:
            torch.cuda.empty_cache()

    gc.collect()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Qwen3.8-Flash-Next safetensors layers for MS streaming")
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    loader = SafetensorsLayerLoader(args.hf_dir)
    keys = loader.list_layer_keys(args.layer)
    print(f"Layer {args.layer}: {len(keys)} tensors found in safetensors index.")


if __name__ == "__main__":
    main()
