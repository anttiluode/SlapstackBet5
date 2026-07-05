# Slapstack Bet 5 — Object Permanence in Gabor Packet Space

> "we prompt a tractor, a tractor appears in gabor space, background as a
> beach in miami, a beach appears, at dusk, it turns to dusk, **the tractor
> stays**."

Bet 4 answered the go/no-go: a recognizable tractor crystallized out of 256
hard-gated Gabor packets under nothing but score-distillation gradients from
a frozen Stable Diffusion model. Bet 5 tests the claim that makes the whole
program worth running: **permanence and editability by construction.**

The scene is a finite set of Gabor atoms, each with position, orientation,
scale, frequency, **envelope-relative phase**, amplitude, and color, behind a
hard-concrete gate. Identity lives in the intrinsic channels; pose lives in
a Sim(2) group element. Because the factorization is architectural, "the
tractor stays" is not a hoped-for emergent property — it is a frozen tensor.

## Ledger

**Verified (Bet 4, GPU):**
- 205 open atoms reconstruct a real tractor photo at 23.5 dB; the whole
  editable scene is a 15 KB file (~2,500 floats).
- Pure SDS (SD 2.1-base, cfg 50, timestep annealing 0.98→0.50) assembles a
  recognizable tractor from random atoms. **Bet 4: GO.**
- Render resolution is causal for SDS gradient quality: native 512 rendering
  fixed the failure that 256→512 bilinear upsampling caused.
- Additive-render-into-sigmoid can saturate under color-token pressure
  ("solid red" collapse); LR rebalance (color 2e-3, bg 5e-4) escapes it.

**Verified (this repo, CPU test battery — `python tests.py`):**
- Bet 4 `atoms.pt` files load directly (N inferred, group buffer defaulted).
- Sim(2) cameras are **exactly** equivariant: a camera shift equals a
  pixel-space roll to machine precision (MSE ~4e-15). Texture is
  phase-locked to envelopes; pose never touches phase.
- Freezing geometry gives **bitwise** permanence under optimization
  (SHA-256 fingerprint identical after training steps) while appearance
  channels train freely.
- Per-atom group masks: masked atoms bitwise frozen, others train.
- The leaky soft clamp keeps a nonzero escape gradient even at absurd
  saturation (bg_bias=50) — the solid-red trap always has an exit ramp.

**Killed:**
- "SDS gradients are too chaotic to organize sparse atoms" (Bet 4).
- "Plain tanh soft clamp has no dead zones" — false; tanh gradient
  underflows in fp32 for |pre| ≳ 40. Caught by test 5, fixed with a
  0.02·pre linear leak. Kept on the ledger as a warning.

**Open (what this repo exists to test):**
1. Frozen-geometry relighting reads as *the same tractor at dusk*.
2. Random Sim(2) cameras cure the SDS zoom-crop trap.
3. Normalized SDS loss + gate warmup re-engages pruning in SDS mode
   (Bet 4 ran 256/256 open the whole way — L0 was numerically invisible).
4. Two-group editing: background repaints to a beach while a frozen
   tractor group survives untouched.

## Install

```bash
pip install torch diffusers transformers accelerate pillow numpy
python tests.py     # ~1 min, CPU, no SD weights needed — should print 5x PASS
```

## What's new vs Bet 4

| Change | Why |
|---|---|
| Leaky soft clamp `4·tanh(pre/4) + 0.02·pre` | Hard clamp fixed saturation but had zero gradient outside the corridor; plain tanh dies numerically at depth. The leak guarantees an escape gradient at any saturation. |
| Random Sim(2) cameras during SDS | DreamFusion's cure for the zoom-crop trap, nearly free here: cameras are parameter arithmetic on atoms. Only a complete, centered object scores well under every random view. |
| SDS loss normalized per-element + `--gate-warmup` | In Bet 4 the raw SDS loss dwarfed L0 → gates never closed. Now "tractor from K atoms" is measurable in SDS mode. |
| `--init-atoms` | Load any previous atoms.pt; N inferred from the file. Reconstruction as initialization = semi-amortized refinement in generative clothing. |
| `--freeze` channel groups | `geometry`, `appearance`, or any of position / orientation / scale / frequency / phase / amp / color / bg / gates. Frozen = excluded from the optimizer; geometry runs are fingerprint-verified. |
| `--train-groups` + `--assign-group-rect` | Per-atom gradient masks. The two-slot tractor/background experiment is a flag, not a rewrite. |
| `--mode render` with `--camera` / `--gif` | The "glide" demo: object translates, zooms, rotates with texture riding the envelopes. Zero re-optimization. |

## The experiments, in order

### 1. Permanence (the bet): dusk relight with frozen geometry

Uses the `atoms.pt` from your 23.5 dB Bet 4 recon run.

```bash
python bet5_gabor_sds.py --mode sds \
    --init-atoms runs/recon_tractor/atoms.pt \
    --freeze geometry,gates \
    --prompt "a photo of a red tractor at dusk, golden hour, warm light" \
    --iters 1500 --render-size 512 --cfg 50 --no-camera \
    --out runs/bet5_dusk
```

`--no-camera` here: the object is already well-composed; cameras are for
forming objects from scratch. The script prints the geometry fingerprint
before/after — it must be IDENTICAL (permanence by construction), so the
scientific question is only whether appearance channels alone can express
"dusk". GO: same tractor, relit. NO-GO: scene refuses to read as dusk after
a seed retry and `--cfg 100`.

### 2. Zoom-trap cure: from-scratch SDS with random cameras

```bash
python bet5_gabor_sds.py --mode sds \
    --prompt "a photo of a red tractor in a green field" \
    --atoms 256 --iters 3000 --render-size 512 --cfg 50 \
    --out runs/bet5_cameras
```

Cameras are ON by default in sds mode (`--cam-zoom 0.3 --cam-shift 0.25
--cam-rot 0.15`). Compare composition against the Bet 4 run that anchored a
zoomed front-left crop. Also watch the open-atom count: with the normalized
loss and warmup, gates should start closing after iteration ~400.

### 3. The glide demo

```bash
python bet5_gabor_sds.py --mode render \
    --init-atoms runs/recon_tractor/atoms.pt \
    --render-size 512 --gif --camera "1.3,0.25,0.2,-0.1" \
    --out runs/glide
```

Writes `identity.png`, `camera.png`, and `glide.gif` — the tractor sweeping
across the frame with texture phase-locked to its envelopes. This is the
equivariance test as a visible object.

### 4. Two slots: the beach (preview of Bet 6)

Assign background atoms to group 1 (rect coordinates in [-1,1]; tune the
box around your tractor), then let ONLY the background train:

```bash
python bet5_gabor_sds.py --mode sds \
    --init-atoms runs/recon_tractor/atoms.pt \
    --assign-group-rect "-1,-1,1,1:1" \
    --assign-group-rect "-0.55,-0.5,0.55,0.6:0" \
    --train-groups 1 --freeze gates \
    --prompt "a tractor on a beach in miami, ocean, sand" \
    --iters 1500 --render-size 512 --cfg 50 --no-camera \
    --out runs/bet5_beach
```

(Second rect overwrites the first inside the box: everything is group 1
except a central window of group 0 = tractor. Rect assignment is a crude
stand-in for real grouping — bets 2/3, common-fate clustering — but it is
enough to test whether subset-masked SDS repaints a background around a
bitwise-frozen object.)

## Files

```
bet5_gabor_sds.py   everything: renderer, gates, cameras, freeze/groups,
                    recon / sds / render modes
tests.py            CPU verification battery (run this first)
```

## Honesty note

The recon / render / freeze / group / camera machinery in this repo was
executed and verified on CPU before shipping (see test battery output in
ledger above). The sds loop reuses the Bet-4 loop that ran on real GPU +
SD 2.1, but the *new* pieces (cameras under real SDS gradients, normalized
L0, frozen-channel SDS) have not yet seen a real run. Experiment 1's first
launch is a smoke test. Known open risk: SD 2.1's mode-seeking mean did a
lot of compositional work in Bet 4 and oversaturation will persist; VSD or
multi-step guidance are the known upgrades, not to be reached for until
they're the actual bottleneck.

---
*Slapstack lineage: hard-concrete gates, envelope-relative phase, honest
ledgers. Do not hype. Do not lie. Just show.*
