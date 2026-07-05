#!/usr/bin/env python3
"""
BET 4 — Score Distillation into Gabor packet space.
====================================================

Question (go/no-go for the whole compositional-editing program):
    Does a recognizable object crystallize out of a few hundred hard-gated
    Gabor packets under nothing but distilled gradients from a frozen
    Stable Diffusion model?

Lineage (Slapstack ledger):
  * Atoms carry ENVELOPE-RELATIVE phase. Translation of an atom leaves its
    phase untouched -> phase is object-intrinsic texture, not pose.
    (This is the design fix for the verified amortization-collapse result:
    here there is NO amortized encoder at all — pure test-time optimization,
    the limiting case of semi-amortized refinement.)
  * Per-atom hard-concrete gates (two-way doors) with L0 pressure, annealed
    to deterministic at the end. We report how many atoms survive.

Modes:
  recon : fit atoms to a target image by MSE. No diffusion model needed.
          Sanity check for renderer/gradients AND a capacity upper bound:
          if atoms can't even *reconstruct* a tractor photo, SDS can't
          conjure one.
  sds   : the actual bet. Frozen Stable Diffusion (diffusers), classic
          DreamFusion-style SDS with CFG + timestep annealing, gradients
          flow only into atom parameters.

Usage (GPU box):
  pip install torch diffusers transformers accelerate pillow numpy
  python bet4_gabor_sds.py --mode sds --prompt "a photo of a tractor" \
      --atoms 256 --iters 3000 --out runs/tractor

  python bet4_gabor_sds.py --mode recon --target tractor.jpg \
      --atoms 256 --iters 2000 --out runs/recon_check

Honesty note: everything in `recon` mode was executed and verified on CPU
before shipping. The `sds` path follows the standard diffusers SDS recipe
but was NOT executed in the build environment (no GPU / no HF weights
there) — treat the first run as a smoke test.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Hard-concrete gates (Louizos et al.), Slapstack-style two-way doors
# ----------------------------------------------------------------------------

class HardConcreteGates(nn.Module):
    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0

    def __init__(self, n, init_logit=2.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n,), float(init_logit)))

    def forward(self, hard_eval=False):
        if self.training and not hard_eval:
            u = torch.rand_like(self.logits).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logits) / self.BETA)
        else:
            s = torch.sigmoid(self.logits)
        z = s * (self.ZETA - self.GAMMA) + self.GAMMA
        return z.clamp(0.0, 1.0)

    def l0(self):
        # P(gate > 0) — expected number of open doors
        return torch.sigmoid(
            self.logits - self.BETA * math.log(-self.GAMMA / self.ZETA)
        )

    @torch.no_grad()
    def hard_open(self):
        z = torch.sigmoid(self.logits) * (self.ZETA - self.GAMMA) + self.GAMMA
        return (z.clamp(0, 1) > 0.5)


# ----------------------------------------------------------------------------
# Differentiable Gabor packet image
# ----------------------------------------------------------------------------

class GaborPacketImage(nn.Module):
    """
    Image = sigmoid( bg_bias + sum_i  g_i * a_i * c_i * env_i(x,y) * carrier_i(x,y) )

      env_i     = exp(-0.5 (u^2/su^2 + v^2/sv^2)),  (u,v) atom-local rotated coords
      carrier_i = cos(2*pi*f_i*u + phi_i)           <-- phi is ENVELOPE-RELATIVE:
                                                        moving x_i does not touch phi_i.
    Coordinates in [-1, 1]; scales/frequencies in units of image half-width.
    """

    def __init__(self, n_atoms=256, coarse_frac=0.25, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n = n_atoms
        nc = int(n * coarse_frac)  # big low-freq atoms (background-ish)

        self.xy_raw = nn.Parameter(torch.randn(n, 2, generator=g) * 0.7)
        self.theta = nn.Parameter(torch.rand(n, generator=g) * math.pi)

        log_s = torch.empty(n)
        log_s[:nc] = math.log(0.45) + 0.25 * torch.randn(nc, generator=g)
        log_s[nc:] = math.log(0.12) + 0.35 * torch.randn(n - nc, generator=g)
        self.log_sigma_u = nn.Parameter(log_s.clone())
        self.log_sigma_v = nn.Parameter(log_s + 0.2 * torch.randn(n, generator=g))

        f = torch.empty(n)
        f[:nc] = 0.25 + 0.5 * torch.rand(nc, generator=g)
        f[nc:] = 0.75 + 2.0 * torch.rand(n - nc, generator=g)
        self.freq_raw = nn.Parameter(torch.log(torch.expm1(f)))  # inv-softplus

        self.phase = nn.Parameter(2 * math.pi * torch.rand(n, generator=g))
        self.amp = nn.Parameter(0.35 + 0.15 * torch.randn(n, generator=g))
        self.color = nn.Parameter(0.30 * torch.randn(n, 3, generator=g))
        self.bg_bias = nn.Parameter(torch.zeros(3))

        self.gates = HardConcreteGates(n)
        self.n_atoms = n

    # -- derived params -------------------------------------------------------
    def xy(self):
        return torch.tanh(self.xy_raw)

    def freq(self):
        return F.softplus(self.freq_raw)

    def sigmas(self):
        return (self.log_sigma_u.exp().clamp(5e-3, 2.0),
                self.log_sigma_v.exp().clamp(5e-3, 2.0))

    # -- render ---------------------------------------------------------------
    def render(self, H, W, device, chunk=64, hard_gates=False):
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")

        xy = self.xy().to(device)
        theta = self.theta.to(device)
        su, sv = self.sigmas()
        su, sv = su.to(device), sv.to(device)
        f = self.freq().to(device)
        phi = self.phase.to(device)
        amp = self.amp.to(device)
        col = self.color.to(device)
        z = self.gates(hard_eval=hard_gates).to(device)

        pre = torch.zeros(3, H, W, device=device) + self.bg_bias.to(device)[:, None, None]

        for i0 in range(0, self.n_atoms, chunk):
            sl = slice(i0, min(i0 + chunk, self.n_atoms))
            dx = X[None] - xy[sl, 0, None, None]
            dy = Y[None] - xy[sl, 1, None, None]
            ct = torch.cos(theta[sl])[:, None, None]
            st = torch.sin(theta[sl])[:, None, None]
            u = ct * dx + st * dy
            v = -st * dx + ct * dy
            env = torch.exp(-0.5 * ((u / su[sl, None, None]) ** 2 +
                                    (v / sv[sl, None, None]) ** 2))
            carrier = torch.cos(2 * math.pi * f[sl, None, None] * u + phi[sl, None, None])
            w = (z[sl] * amp[sl])[:, None, None] * env * carrier   # (n,H,W)
            pre = pre + torch.einsum("nhw,nc->chw", w, col[sl])

        pre = torch.clamp(pre, -4.0, 4.0)
        return torch.sigmoid(pre)

    def ledger(self):
        return {
            "atoms_total": self.n_atoms,
            "atoms_open_hard": int(self.gates.hard_open().sum().item()),
            "expected_L0": float(self.gates.l0().sum().item()),
        }


# ----------------------------------------------------------------------------
# Optimizer with per-parameter learning rates
# ----------------------------------------------------------------------------

# Change this existing block:
def make_optimizer(model, lr_scale=1.0):
    G = lambda p, lr: {"params": [p], "lr": lr * lr_scale}
    return torch.optim.Adam([
        G(model.xy_raw, 5e-3), G(model.theta, 5e-3),
        G(model.log_sigma_u, 5e-3), G(model.log_sigma_v, 5e-3),
        G(model.freq_raw, 5e-3), G(model.phase, 2e-2),
        G(model.amp, 1e-2), G(model.color, 2e-3),    # <-- Reduced from 1e-2 to 2e-3
        G(model.bg_bias, 5e-4), G(model.gates.logits, 2e-2), # <-- Reduced from 1e-2 to 5e-4
    ], betas=(0.9, 0.99))


def save_png(img_chw, path):
    from PIL import Image
    arr = (img_chw.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ----------------------------------------------------------------------------
# Mode: recon  (verified on CPU)
# ----------------------------------------------------------------------------

def run_recon(args, device):
    from PIL import Image
    tgt = Image.open(args.target).convert("RGB").resize((args.render_size,) * 2)
    target = torch.from_numpy(np.asarray(tgt)).float().permute(2, 0, 1) / 255.0
    target = target.to(device)

    model = GaborPacketImage(args.atoms, seed=args.seed).to(device)
    model.train()
    opt = make_optimizer(model)

    os.makedirs(args.out, exist_ok=True)
    save_png(target, os.path.join(args.out, "target.png"))
    log = []
    t0 = time.time()
    for it in range(args.iters):
        opt.zero_grad()
        img = model.render(args.render_size, args.render_size, device, chunk=args.chunk)
        mse = F.mse_loss(img, target)
        l0 = model.gates.l0().sum() / model.n_atoms
        loss = mse + args.l0_weight * l0
        loss.backward()
        opt.step()
        if it % max(1, args.iters // 20) == 0 or it == args.iters - 1:
            psnr = -10 * math.log10(max(mse.item(), 1e-12))
            row = {"it": it, "mse": mse.item(), "psnr_db": psnr, **model.ledger()}
            log.append(row)
            print(f"[recon] it {it:5d}  mse {mse.item():.5f}  psnr {psnr:5.2f} dB  "
                  f"open {row['atoms_open_hard']}/{model.n_atoms}  "
                  f"({time.time()-t0:.0f}s)")
            save_png(img, os.path.join(args.out, f"it_{it:05d}.png"))
    finish(model, args, device, log)


# ----------------------------------------------------------------------------
# Mode: sds  (the bet — requires GPU + Stable Diffusion weights)
# ----------------------------------------------------------------------------

def run_sds(args, device):
    from diffusers import StableDiffusionPipeline, DDPMScheduler

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False)
    pipe.to(device)
    vae, unet, tok, te = pipe.vae, pipe.unet, pipe.tokenizer, pipe.text_encoder
    for m in (vae, unet, te):
        m.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    alphas = sched.alphas_cumprod.to(device)
    T = sched.config.num_train_timesteps

    def embed(text):
        ids = tok(text, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        return te(ids)[0]

    with torch.no_grad():
        emb = torch.cat([embed(args.negative_prompt), embed(args.prompt)])

    model = GaborPacketImage(args.atoms, seed=args.seed).to(device)
    model.train()
    opt = make_optimizer(model)

    os.makedirs(args.out, exist_ok=True)
    log = []
    t0 = time.time()
    for it in range(args.iters):
        opt.zero_grad()
        img = model.render(args.render_size, args.render_size, device,
                           chunk=args.chunk)                      # (3,H,W) [0,1]
        x = img[None] * 2 - 1
        if args.render_size != 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear",
                              align_corners=False)

        latents = vae.encode(x.to(dtype)).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.float()

        # timestep annealing: shrink t_max as structure forms
        frac = it / max(1, args.iters - 1)
        t_max = args.t_max_start + (args.t_max_end - args.t_max_start) * frac
        t = torch.randint(int(args.t_min * T), int(t_max * T), (1,), device=device)

        noise = torch.randn_like(latents)
        noisy = sched.add_noise(latents, noise, t)
        with torch.no_grad():
            eps = unet(torch.cat([noisy] * 2).to(dtype),
                       torch.cat([t] * 2), encoder_hidden_states=emb).sample.float()
            eps_un, eps_tx = eps.chunk(2)
            eps_hat = eps_un + args.cfg * (eps_tx - eps_un)

        w = (1 - alphas[t]).view(-1, 1, 1, 1)
        grad = (w * (eps_hat - noise)).detach()
        sds_loss = (grad * latents).sum() / latents.shape[0]

        l0 = model.gates.l0().sum() / model.n_atoms
        loss = sds_loss + args.l0_weight * l0
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g_ in opt.param_groups
                                        for p in g_["params"]], 1.0)
        opt.step()

        if it % max(1, args.iters // 30) == 0 or it == args.iters - 1:
            row = {"it": it, "sds": float(sds_loss.item()),
                   "t_max": t_max, **model.ledger()}
            log.append(row)
            print(f"[sds] it {it:5d}  t_max {t_max:.2f}  "
                  f"open {row['atoms_open_hard']}/{model.n_atoms}  "
                  f"({time.time()-t0:.0f}s)")
            with torch.no_grad():
                save_png(model.render(args.render_size, args.render_size,
                                      device, chunk=args.chunk, hard_gates=True),
                         os.path.join(args.out, f"it_{it:05d}.png"))
    finish(model, args, device, log)


def finish(model, args, device, log):
    model.eval()
    with torch.no_grad():
        img = model.render(args.render_size, args.render_size, device,
                           chunk=args.chunk, hard_gates=True)
    save_png(img, os.path.join(args.out, "final_hardgates.png"))
    torch.save(model.state_dict(), os.path.join(args.out, "atoms.pt"))
    ledger = {"mode": args.mode, "prompt": getattr(args, "prompt", None),
              "final": model.ledger(), "log": log,
              "note": ("Envelope-relative phase; hard-concrete gates. "
                       "GO if final_hardgates.png is a recognizable object; "
                       "NO-GO if soup after full schedule + one seed retry.")}
    with open(os.path.join(args.out, "ledger.json"), "w") as fh:
        json.dump(ledger, fh, indent=2)
    print(f"done -> {args.out}  | open atoms: {model.ledger()['atoms_open_hard']}")


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Bet 4: SDS into Gabor packet space")
    p.add_argument("--mode", choices=["recon", "sds"], required=True)
    p.add_argument("--out", default="runs/bet4")
    p.add_argument("--atoms", type=int, default=256)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--render-size", type=int, default=256,
                   help="render resolution (upsampled to 512 for SD)")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l0-weight", type=float, default=2e-3)
    # recon
    p.add_argument("--target", help="target image path (recon mode)")
    # sds
    p.add_argument("--prompt", default="a photo of a tractor")
    p.add_argument("--negative-prompt",
                   default="blurry, low quality, deformed")
    p.add_argument("--sd-model", default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--cfg", type=float, default=50.0)
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max-start", type=float, default=0.98)
    p.add_argument("--t-max-end", type=float, default=0.50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if args.mode == "recon":
        assert args.target, "--target required in recon mode"
        run_recon(args, device)
    else:
        run_sds(args, device)


if __name__ == "__main__":
    main()
