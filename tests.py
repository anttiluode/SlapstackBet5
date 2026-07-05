#!/usr/bin/env python3
"""
Bet 5 verification battery. CPU-only, no diffusion model needed, ~1 min.

    python tests.py

Covers: Bet-4 atoms.pt compatibility, exact Sim(2) equivariance, geometry
freeze permanence, per-atom group masking, and saturation escape gradient.
"""
import math
import torch

from bet5_gabor_sds import (GaborPacketImage, load_atoms, resolve_frozen,
                            make_optimizer, group_mask, apply_grad_masks,
                            geometry_fingerprint)

dev = torch.device("cpu")
torch.manual_seed(0)


def make_trained(n=64, iters=120):
    """Tiny recon so tests run on a non-random atom set."""
    tgt = torch.zeros(3, 64, 64)
    tgt[0, 16:40, 12:44] = 0.8          # a red slab
    tgt[2, 40:60, :] = 0.6              # blue band
    m = GaborPacketImage(n, seed=0)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(iters):
        opt.zero_grad()
        ((m.render(64, 64, dev) - tgt) ** 2).mean().backward()
        opt.step()
    return m


def test_bet4_compat():
    m = make_trained()
    sd = m.state_dict()
    sd.pop("group")                     # Bet 4 files have no group buffer
    torch.save(sd, "/tmp/bet4_style.pt")
    m2 = load_atoms("/tmp/bet4_style.pt")
    assert m2.n_atoms == 64 and (m2.group == 0).all()
    print("PASS bet4_compat: N inferred, group buffer defaulted")


def test_equivariance():
    m = make_trained(); m.eval()
    S = 96
    px = 2.0 / (S - 1)
    dx, dy = 20 * px, -10 * px          # exact 20px right, 10px up
    with torch.no_grad():
        a = m.render(S, S, dev, hard_gates=True)
        b = m.render(S, S, dev, hard_gates=True, camera=(1.0, 0.0, dx, dy))
    rolled = torch.roll(a, shifts=(-10, 20), dims=(1, 2))
    crop = (slice(None), slice(15, S - 15), slice(25, S - 25))
    err = ((b[crop] - rolled[crop]) ** 2).mean().item()
    assert err < 1e-6, f"equivariance broken: {err:.2e}"
    print(f"PASS equivariance: camera shift == pixel roll (MSE {err:.1e})")


def test_freeze_permanence():
    m = make_trained(); m.train()
    frozen = resolve_frozen("geometry,gates")
    opt = make_optimizer(m, frozen, "sds")
    fp0 = geometry_fingerprint(m)
    col0 = m.color.detach().clone()
    tgt = torch.rand(3, 64, 64)
    for _ in range(5):
        opt.zero_grad()
        ((m.render(64, 64, dev) - tgt) ** 2).mean().backward()
        opt.step()
    assert geometry_fingerprint(m) == fp0, "geometry moved despite freeze"
    assert not torch.equal(col0, m.color), "appearance did not train"
    print(f"PASS freeze_permanence: geometry fingerprint {fp0} bitwise stable")


def test_group_masking():
    m = make_trained(); m.train()
    m.group[:32] = 1
    opt = make_optimizer(m, set(), "sds")
    mask = group_mask(m, "0")
    before = m.xy_raw.detach().clone()
    tgt = torch.rand(3, 64, 64)
    for _ in range(5):
        opt.zero_grad()
        ((m.render(64, 64, dev) - tgt) ** 2).mean().backward()
        apply_grad_masks(m, mask, dev)
        opt.step()
    assert torch.equal(before[:32], m.xy_raw[:32]), "masked atoms moved"
    assert not torch.equal(before[32:], m.xy_raw[32:]), "trainable atoms stuck"
    print("PASS group_masking: group-1 atoms bitwise frozen, group-0 trained")


def test_saturation_escape():
    m = GaborPacketImage(8, seed=1)
    m.bg_bias.data.fill_(50.0)          # brutal saturation
    m.render(32, 32, dev).sum().backward()
    g = m.bg_bias.grad.abs().sum().item()
    assert g > 1e-3, f"saturation escape gradient too small: {g:.2e}"
    print(f"PASS saturation_escape: |grad| = {g:.2e} at bg_bias=50")


if __name__ == "__main__":
    test_bet4_compat()
    test_equivariance()
    test_freeze_permanence()
    test_group_masking()
    test_saturation_escape()
    print("\nall tests pass")
