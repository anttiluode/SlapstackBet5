# Bet 4 — SDS into Gabor packet space

**Question:** does a recognizable object crystallize out of ~256 hard-gated
Gabor packets under nothing but distilled gradients from frozen Stable
Diffusion? This is the go/no-go for the compositional-editing program
(tractor / beach / dusk / permanence).

## Files
- `bet4_gabor_sds.py` — everything: renderer, gates, recon mode, SDS mode.
- `runs/cpu_check/triptych.png` — build-time verification: target | 93-atom
  reconstruction | same atoms shifted by (+0.35, −0.20) in position only.

## Design commitments (from the Slapstack ledger)
1. **Envelope-relative phase.** `carrier = cos(2πf·u + φ)` with `u` measured
   from the envelope center. Translation never touches φ — phase is intrinsic
   texture, not pose. Direct fix for the verified amortization phase-collapse:
   here there is no encoder at all, pure test-time optimization.
2. **Hard-concrete gates per atom**, L0 pressure, hard threshold at eval.
   Ledger reports open-atom count so results read "tractor from K atoms".
3. **Additive render → sigmoid.** Known limitation: no occlusion model yet
   (that is bets 2–3 territory: depth-ordered compositing over groups).

## Run order on the GPU box
```bash
pip install torch diffusers transformers accelerate pillow numpy

# 1. Capacity upper bound (~5 min): can atoms even REPRESENT a tractor?
python bet4_gabor_sds.py --mode recon --target some_tractor_photo.jpg \
    --atoms 256 --iters 2000 --render-size 256 --out runs/recon_tractor

# 2. The bet (~30–60 min at 256px):
python bet4_gabor_sds.py --mode sds --prompt "a photo of a red tractor, centered, high quality" --atoms 256 --iters 3000 --render-size 256 --cfg 50.0 --sd-model "sd2-community/stable-diffusion-2-1-base" --out runs/sds_tractor

```

## Go / no-go criteria
- **GO:** `final_hardgates.png` is a recognizable tractor (shape + wheels
  discernible; color sane) with a finite open-atom count. Blurry is fine —
  256 atoms is a low-capacity canvas; we want objecthood, not photorealism.
- **NO-GO:** oriented-texture soup after the full schedule AND a retry with
  `--seed 1` and `--cfg 100`. Record both in the ledger before calling it.

## Knobs that matter (in likely order)
- `--cfg` 30–100. DreamFusion needed ~100; latent-space SDS often works at
  ~50. If soup: raise it.
- Timestep annealing `--t-max-end` (default 0.5). If it locks in a bad coarse
  layout early, raise back toward 0.8.
- `--atoms` 256 → 512 if recon mode says capacity is the binding constraint.
- `--l0-weight` down to 0 if gates close too aggressively during SDS chaos.

## Known failure modes to expect (write them down, don't hide them)
- SDS mode-seeking → oversaturated colors (classic). Acceptable for go/no-go.
- Janus-style incoherence doesn't apply (2D), but multi-object hallucination
  does; "a photo of a single tractor, centered" is a fair prompt tweak.
- If VAE-encode gradients are noisy at 256→512 upsample, try
  `--render-size 512` directly (more memory, cleaner gradients).

## Status honesty
`recon` mode executed and verified (CPU, toy target: 11.5→27.4 dB in 400
iters, gates pruned 96→93, translation equivariance confirmed visually).
`sds` mode follows the standard diffusers SDS recipe but was NOT executed in
the build environment (no GPU / no SD weights). First run = smoke test.
