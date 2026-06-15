"""T_XS runner — build panel+features once, run combos, write FULL trade ledger + cells.

A "combo" = (family, H). Knobs are IS-tuned per window. Output per run dir:
  - trades.parquet : FULL per-rebalance-bar pnl ledger (combo_id, window_id, phase,
                     bar_idx, pnl_bp). The standing rule: everything computable off one run.
  - cells.parquet  : per (combo,window,phase) aggregate (n_reb, pnl_bp, pos_bp, neg_bp).
  - summary.csv    : per-combo OOS sharpe/pf/totals.
  - combos.parquet : the structural axes per combo_id.

Cross-sectional combos are FEW (families × H), so we don't need a 32-proc pool over
combos; instead each combo's model fit is the heavy part (GPU). We run combos serially
(GPU is the shared resource) but the SIM fan-out over knobs is batched on GPU inside
xs_wfo. nproc kept for the lgbm/CPU path.
"""
import os
import sys
import time
import csv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import xs_common as xc
import xs_data as xd
import xs_features as xf
import xs_wfo as xw

_TRADE_SCHEMA = pa.schema([
    ("combo_id", pa.uint32()), ("window_id", pa.uint16()), ("phase", pa.uint8()),
    ("bar_idx", pa.uint32()), ("pnl_bp", pa.float32()),
])


class _Ledger:
    def __init__(self, path):
        self.w = pq.ParquetWriter(path, _TRADE_SCHEMA, compression="zstd"); self.n = 0

    def add(self, recs):
        if not recs:
            return
        self.w.write_table(pa.table({
            "combo_id": pa.array([r[0] for r in recs], pa.uint32()),
            "window_id": pa.array([r[1] for r in recs], pa.uint16()),
            "phase": pa.array([r[2] for r in recs], pa.uint8()),
            "bar_idx": pa.array([r[3] for r in recs], pa.uint32()),
            "pnl_bp": pa.array([float(r[4]) * 1e4 for r in recs], pa.float32()),
        }, schema=_TRADE_SCHEMA))
        self.n += len(recs)

    def close(self):
        self.w.close()


def _cells(recs):
    agg = {}
    for (cid, wid, ph, bi, pnl) in recs:
        k = (cid, wid, ph); bp = pnl * 1e4
        a = agg.get(k)
        if a is None:
            agg[k] = [1, bp, bp if bp > 0 else 0.0, -bp if bp < 0 else 0.0]
        else:
            a[0] += 1; a[1] += bp
            if bp > 0:
                a[2] += bp
            else:
                a[3] += -bp
    return [(k[0], k[1], k[2], v[0], v[1], v[2], v[3]) for k, v in agg.items()]


def _write_cells(cells, path):
    if not cells:
        pq.write_table(pa.table({"combo_id": pa.array([], pa.uint32())}), path); return
    pq.write_table(pa.table({
        "combo_id": pa.array([c[0] for c in cells], pa.uint32()),
        "window_id": pa.array([c[1] for c in cells], pa.uint16()),
        "phase": pa.array([c[2] for c in cells], pa.uint8()),
        "n_reb": pa.array([c[3] for c in cells], pa.uint32()),
        "pnl_bp": pa.array([float(c[4]) for c in cells], pa.float32()),
        "pos_bp": pa.array([float(c[5]) for c in cells], pa.float32()),
        "neg_bp": pa.array([float(c[6]) for c in cells], pa.float32()),
    }), path, compression="zstd")


def build_combos(families=None, horizons=None):
    families = families or ["lgbm", "mlp", "xtransformer"]
    # richer horizon set = more distinct pair-switch holding periods = more strategies
    # AND more combos to fan across cores. Env-overridable (XS_HORIZONS="6,24,72").
    if horizons is None:
        env = os.environ.get("XS_HORIZONS", "")
        horizons = [int(x) for x in env.split(",")] if env else [4, 6, 8, 12, 24, 48, 72, 120, 168, 336]
    combos = []
    cid = 0
    for fam in families:
        for H in horizons:
            combos.append(dict(combo_id=cid, family=fam, H=int(H))); cid += 1
    return combos


# fork-COW shared globals for the worker pool (set BEFORE forking in run()).
# X is ~5.5GB; workers inherit it read-only via copy-on-write → no per-task pickling,
# no 10x-RAM blowup. Each task ships only the tiny spec dict.
_W = {}


def _combo_worker(spec):
    try:
        summ, recs = xw.run_combo(_W["panel"], _W["X"], _W["names"], spec)
        return spec, summ, recs, None
    except Exception as e:
        return spec, None, [], f"{type(e).__name__}: {e}"


def run(name, combos, out_root=None, limit=None, nproc=None):
    import multiprocessing as mp
    out_root = out_root or xc.OUT_ROOT
    if limit:
        combos = combos[:limit]
    nproc = nproc or int(os.environ.get("XS_NPROC", "8"))
    outdir = f"{out_root}/{name}"
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    print(f"[{name}] building panel...", flush=True)
    panel = xd.build_panel()
    print(f"[{name}] panel T={panel['T']} N={panel['N']} | features...", flush=True)
    X, names, _ = xf.build_features(panel)
    nproc = max(1, min(nproc, len(combos)))
    print(f"[{name}] features {X.shape} | {len(combos)} combos | nproc={nproc} | "
          f"IS={xc.IS} WALK={xc.WALK} NS_KNOB={xc.NS_KNOB} GPU={xc.gpu_enabled()}", flush=True)

    ledger = None if os.environ.get("XS_NO_LEDGER") == "1" else _Ledger(f"{outdir}/trades.parquet")
    cells = []; summaries = []

    def _consume(spec, summ, recs, err, ct):
        if err:
            print(f"  [warn] combo {spec['combo_id']} {spec['family']}/H{spec['H']}: {err}", flush=True)
            return
        if recs:
            cells.extend(_cells(recs))
            if ledger is not None:
                ledger.add(recs)
        if summ is not None:
            summaries.append(summ)
            print(f"  [{spec['family']}/H{spec['H']}] OOS Sharpe={summ['oos_sharpe']:.3f} "
                  f"PF={summ['oos_pf']:.3f} bars={summ['oos_bars']} {time.time()-ct:.0f}s", flush=True)

    _W["panel"], _W["X"], _W["names"] = panel, X, names   # set BEFORE fork → COW shared
    if nproc <= 1:
        for spec in combos:
            ct = time.time()
            s, summ, recs, err = _combo_worker(spec)
            _consume(s, summ, recs, err, ct)
    else:
        ct0 = time.time()
        # maxtasksperchild=1: respawn each worker after EVERY combo so all per-combo
        # [T,N] label/score arrays + the LGBM predict copy of the flattened cube are
        # fully released back to the OS (no cross-combo creep toward the RAM floor). The
        # ~5.5 GB normed cube is re-inherited read-only via fork-COW on each respawn, so
        # this costs only fork time, not a fresh copy.
        with mp.get_context("fork").Pool(nproc, maxtasksperchild=1) as pool:
            for s, summ, recs, err in pool.imap_unordered(_combo_worker, combos):
                _consume(s, summ, recs, err, ct0)
    if ledger is not None:
        ledger.close()
    _write_cells(cells, f"{outdir}/cells.parquet")
    pq.write_table(pa.table({
        "combo_id": pa.array([c["combo_id"] for c in combos], pa.uint32()),
        "family": pa.array([c["family"] for c in combos]),
        "H": pa.array([c["H"] for c in combos], pa.int32()),
    }), f"{outdir}/combos.parquet", compression="zstd")
    if summaries:
        with open(f"{outdir}/summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader()
            for s in summaries:
                w.writerow(s)
    el = time.time() - t0
    ntr = ledger.n if ledger is not None else 0
    med_pf = np.median([s["oos_pf"] for s in summaries if np.isfinite(s["oos_pf"])]) if summaries else 0.0
    print(f"[{name}] DONE {el:.0f}s | combos={len(summaries)}/{len(combos)} | "
          f"med OOS PF={med_pf:.3f} | ledger rows={ntr:,}", flush=True)


if __name__ == "__main__":
    fam = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    lim = int(sys.argv[2]) if len(sys.argv) > 2 and int(sys.argv[2]) < 999999 else None
    out = sys.argv[3] if len(sys.argv) > 3 else None
    npr = int(sys.argv[4]) if len(sys.argv) > 4 else None
    run("XS_main", build_combos(families=fam), out_root=out, limit=lim, nproc=npr)
