#!/usr/bin/env python3
"""Download orbital + auxiliary bases from the Basis Set Exchange as JSON."""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASIS_DIR = Path(__file__).resolve().parent.parent / "data" / "basis_sets"
BSE_API = "https://www.basissetexchange.org/api/basis"

# (BSE API id, local filename)
DOWNLOADS = [
    # Orbital
    ("sto-3g", "sto-3g.json"),
    ("6-31g", "6-31g.json"),
    ("6-31g*", "6-31gs.json"),
    ("def2-svp", "def2-svp.json"),
    ("def2-tzvp", "def2-tzvp.json"),
    ("def2-svpd", "def2-svpd.json"),
    ("def2-qzvpp", "def2-qzvpp.json"),
    ("cc-pvdz", "cc-pvdz.json"),
    ("cc-pvtz", "cc-pvtz.json"),
    ("cc-pvqz", "cc-pvqz.json"),
    # Auxiliaries (JK-fit preferred for hybrids; RIFIT kept for optional use)
    ("def2-universal-jkfit", "def2-universal-jkfit.json"),
    ("cc-pvtz-jkfit", "cc-pvtz-jkfit.json"),
    ("cc-pvqz-jkfit", "cc-pvqz-jkfit.json"),
    ("cc-pvdz-rifit", "cc-pvdz-rifit.json"),
    ("cc-pvtz-rifit", "cc-pvtz-rifit.json"),
    ("cc-pvqz-rifit", "cc-pvqz-rifit.json"),
]


def fetch(bse_id: str, out_path: Path) -> None:
    enc = urllib.parse.quote(bse_id, safe="")
    url = f"{BSE_API}/{enc}/format/json/?version=1"
    print(f"  {bse_id} -> {out_path.name} ...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    out_path.write_text(json.dumps(data, indent=4) + "\n")
    print(f"OK ({out_path.stat().st_size} bytes)")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Download BSE JSON basis sets")
    p.add_argument("--force", action="store_true", help="Re-download existing files")
    args = p.parse_args()

    BASIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {BASIS_DIR}")
    n_ok = n_skip = n_fail = 0
    for bse_id, fname in DOWNLOADS:
        out = BASIS_DIR / fname
        if out.exists() and not args.force:
            print(f"  {fname}: skip (exists)")
            n_skip += 1
            continue
        try:
            fetch(bse_id, out)
            n_ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            n_fail += 1
    print(f"\nDone: {n_ok} downloaded, {n_skip} skipped, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
