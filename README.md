# Mental Accounts for Actions: EWA-VQ-ODT

Lightweight add-on for Decision/Online Decision Transformers that keeps a tiny per-action memory (“experience-weighted attraction”) and injects it as a small bias into attention on action tokens. Works with continuous actions via a compact vector-quantized (VQ) codebook and direct grid lookup.

> 📄 The anonymized paper is in `docs/`. Please keep the repo anonymized during review.

## Highlights

* **Plug-and-play**: Additive pre-softmax column bias; no changes to DT/ODT backbone or loss.
* **Fast**: Constant-time action routing; GPU-vectorized updates.
* **Interpretable**: Per-code “mental accounts” of recent success/failure.

## Quick start

```bash
git clone https://github.com/<anon-org>/ewa-vq-odt.git
cd ewa-vq-odt
python -m venv .venv && source .venv/bin/activate   # (or use conda)
pip install -r requirements.txt
```

### Run quick experiments

```bash
# Hopper (3D actions)
python run_quick1_ewa.py --env hopper-medium-v2

# Walker2d (6D actions)
python run_quick2_ewa.py --env walker2d-medium-replay-v2
```

Defaults (see scripts/args): `beta=0.05`, `phi=0.05`, `delta=0.8`, `codes=27`, context length `K∈{5,20}`.


## Citation (anonymized)

If you use this code, please cite the anonymized manuscript in `docs/` (BibTeX provided inside). We’ll update with camera-ready details post-review.

## License

See `LICENSE`.
