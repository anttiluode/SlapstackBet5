# SlapstackBet5 — Object Permanence in Gabor Packet Space

![pic1](/runs/recon_tractor/final_hardgates.png)

> "we prompt a tractor, a tractor appears in gabor space, background as a
> beach in miami, a beach appears, at dusk, it turns to dusk, **the tractor
> stays**."

![pic2](/runs/bet5_dusk/final_hardgates.png)

This repo contains Bets 4 and 5 of the Slapstack program: text-to-image and
image editing performed directly on a sparse set of Gabor packets — no
pixel-space generation, no trained text-to-atom model, no encoder. A scene
is a finite set of atoms (position, orientation, scale, frequency,
**envelope-relative phase**, amplitude, color) behind hard-concrete gates.
Identity lives in the intrinsic channels; pose lives in a Sim(2) group
element; edits are gradient masks over parameter subsets.

**Both bets are now GO.** An entire editable tractor scene is a 15 KB file.

![pic3](/runs/glide/glide.gif)

## Results

### Bet 4 — GO: text conjures an object from random atoms
Pure Score Distillation Sampling (frozen SD 2.1-base, cfg 50, timestep
annealing 0.98→0.50, native 512 rendering) assembled a recognizable red
tractor — chassis, cab, exhaust stack, wheels, treeline — from 256 randomly
initialized Gabor packets. No target image, no paired data, no trained
text→atom model. Separately, recon mode established the capacity bound:
205 open atoms reconstruct a real tractor photo at 23.5 dB PSNR.

### Bet 5 — GO: the tractor stays
Loaded the 23.5 dB recon atoms, froze all geometry channels
(`xy, theta, sigma_u, sigma_v, freq`) plus gates, and ran SDS with
*"a photo of a red tractor at dusk, golden hour, warm light"* — gradients
touching only `phase, amp, color, bg_bias`.

```
geometry fingerprint before/after: e475df55520bf8b2 / e475df55520bf8b2
    -> IDENTICAL — permanence held by construction
open atoms: 205/256 (unchanged, gates frozen)
```

![pic4](/runs/bet5_beach/final_hardgates.png)

The output is unmistakably the *same tractor* — same pose, same wheels,
same cab, same exhaust stack — relit: warm highlights, lit lamps, a
twilight atmosphere replacing the flat green-screen background. Appearance
channels alone expressed a relight while geometry was bitwise untouchable.
This is the empirical form of the fiber-bundle claim: **illumination acts
on the amplitude/chroma fibers; geometry is the base space.**

What makes this different from pixel-space editing: permanence here is not
a property the optimizer was coaxed into preserving (attention surgery,
inversion tricks, careful prompting). It is a boolean mask. The tractor
*cannot* move, because the tensors that encode its shape are excluded from
the optimizer and SHA-256 fingerprinted before and after to prove it.

Honest reading of the result: the relight is real but reads as moody
teal-warm twilight rather than textbook golden hour, and because this run
used no group masks, object and background were relit together. Both are
expected: SD 2.1's mode-seeking pulls color statistics toward its
prompt-mean, and separating "relight the scene" from "relight the object"
is precisely what the group machinery (Experiment 4 below) exists for.
Also note the trainable `phase` channel means *texture* could in principle
have been rewritten even with shape frozen; that it wasn't abused is mildly
interesting and worth a follow-up ablation (`--freeze geometry,gates,phase`).

## Ledger

**Verified (GPU, this repo):**
- Bet 4 GO: SDS assembles a recognizable object from random atoms.
- Bet 5 GO: frozen-geometry relight; fingerprint bitwise identical;
  appearance channels sufficient to express a lighting edit.
- 205 atoms / 23.5 dB / 15 KB capacity bound for a real tractor photo.
- Native-512 rendering is causal for SDS gradient quality (the 256→512
  bilinear upsample destroyed structure through the VAE).
- Additive-render-into-sigmoid saturates under color-token pressure
  ("solid red" collapse); LR rebalance (color 2e-3, bg 5e-4) escapes it.

**Verified (CPU test battery, `python tests.py`, ~1 min, no SD needed):**
- Bet 4 `atoms.pt` files load directly (N inferred, group buffer defaulted).
- Sim(2) cameras exactly equivariant: camera shift == pixel roll to
  machine precision (MSE ~4e-15). Texture is phase-locked to envelopes.
- Geometry freeze is bitwise under optimization (SHA-256 verified) while
  appearance trains.
- Per-atom group masks: masked atoms bitwise frozen, others train.
- Leaky soft clamp keeps a nonzero escape gradient at any saturation depth.

**Killed:**
- "SDS gradients are too chaotic to organize sparse atoms" (Bet 4).
- "Appearance channels can't express a relight without geometry's help"
  (Bet 5).
- "Plain tanh soft clamp has no dead zones" — false; tanh underflows in
  fp32 for |pre| ≳ 40. Caught by test 5, fixed with a 0.02·pre linear leak.

**Open:**
1. Random Sim(2) cameras cure the SDS zoom-crop trap (Experiment 2 — the
   camera code is tested for equivariance but has not yet run under real
   SDS gradients).
2. Normalized SDS loss + gate warmup re-engages pruning in from-scratch
   SDS mode ("tractor from K atoms" is still unmeasured; the Bet 5 run
   had gates frozen by design, so it does not answer this).
3. Two-group editing: background repaints to a beach around a bitwise-
   frozen tractor group (Experiment 4).
4. Phase ablation: does freezing phase alongside geometry change the
   relight quality? (Separates lighting from texture rewriting.)
5. SD's mode-seeking oversaturation: VSD or multi-step guidance are the
   known upgrades — not to be reached for until they are the actual
   bottleneck.

## Repo contents

```
bet5_gabor_sds.py   current main script: renderer, gates, Sim(2) cameras,
                    freeze/group machinery, recon / sds / render modes
tests.py            CPU verification battery — run this first
bet4_gabor_sds.py   Bet 4 script kept for provenance (superseded by bet5:
                    hard clamp, no cameras, no freeze/groups, unnormalized
                    L0 that never pruned in sds mode)
README_bet4.md      Bet 4 documentation, kept for provenance
runs/               ledgers and snapshots from the GO runs
```

## Install

```bash
pip install torch diffusers transformers accelerate pillow numpy
python tests.py     # should print 5x PASS
```

## Reproduce the GO runs

```bash
# Capacity bound (needs any tractor photo):
python bet5_gabor_sds.py --mode recon --target tractor.jpg \
    --atoms 256 --iters 2000 --render-size 256 --out runs/recon_tractor

# Bet 4 (from-scratch SDS):
python bet5_gabor_sds.py --mode sds \
    --prompt "a photo of a red tractor, centered, high quality" \
    --atoms 256 --iters 2000 --render-size 512 --cfg 50 --no-camera \
    --sd-model sd2-community/stable-diffusion-2-1-base \
    --out runs/sds_tractor

# Bet 5 (the permanence run, exactly as executed):
python bet5_gabor_sds.py --mode sds \
    --init-atoms runs/recon_tractor/atoms.pt \
    --freeze geometry,gates \
    --prompt "a photo of a red tractor at dusk, golden hour, warm light" \
    --iters 1500 --render-size 512 --cfg 50 --no-camera \
    --out runs/bet5_dusk
```

## Next experiments

```bash
# 2. Zoom-trap cure: from-scratch SDS with random cameras ON (the default)
python bet5_gabor_sds.py --mode sds \
    --prompt "a photo of a red tractor in a green field" \
    --atoms 256 --iters 3000 --render-size 512 --cfg 50 \
    --out runs/bet5_cameras
# watch: composition (whole + centered?) and open-atom count (pruning
# should engage after the 400-iter gate warmup -> "tractor from K atoms")

# 3. The glide demo (equivariance as a visible object):
python bet5_gabor_sds.py --mode render \
    --init-atoms runs/recon_tractor/atoms.pt \
    --render-size 512 --gif --camera "1.3,0.25,0.2,-0.1" --out runs/glide

# 4. Two slots — the beach (background trains, tractor bitwise frozen):
python bet5_gabor_sds.py --mode sds \
    --init-atoms runs/recon_tractor/atoms.pt \
    --assign-group-rect="-1,-1,1,1:1" \ 
    --assign-group-rect="-0.55,-0.5,0.55,0.6:0" \
    --train-groups 1 --freeze gates \
    --prompt "a tractor on a beach in miami, ocean, sand, blue sky" \
    --iters 1500 --render-size 512 --cfg 50 --no-camera \
    --out runs/bet5_beach
# rect assignment is a crude stand-in for real grouping (common-fate
# clustering is Bets 2/3 territory) — tune the inner box to your tractor.

# 5. Phase ablation (lighting vs texture):
#    rerun the dusk command with --freeze geometry,gates,phase
```

## Honesty note

Bet 4 and Bet 5 results above ran on real GPU + SD 2.1 (community mirror
`sd2-community/stable-diffusion-2-1-base`). The CPU battery verified the
mechanical claims (equivariance, freezing, masking, saturation escape)
before any GPU time was spent. Random cameras under real SDS gradients and
group-masked SDS remain untested — the next runs are smoke tests for those
paths. Ledgers for all runs are committed under `runs/`.

---
*Slapstack lineage: hard-concrete gates, envelope-relative phase, honest
ledgers. Do not hype. Do not lie. Just show.*
