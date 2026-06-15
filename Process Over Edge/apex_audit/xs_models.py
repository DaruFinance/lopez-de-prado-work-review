"""T_XS cross-sectional scorers — produce a causal per-bar per-pair SCORE [T,N].

The model learns to RANK pairs against each other at each bar (the pair-switch is the
strategy). Two families behind a uniform interface (.fit(Xtr,ytr,vtr) / .score(X,v)):

  - 'lgbm'  : LightGBM regressor on the pooled (bar,pair) cross-section rows → fast,
              trees-win-for-tabular (the survey's repeated finding). CPU, deterministic.
  - 'mlp'   : GPU panel MLP over the per-(bar,pair) feature vector. Batched, deterministic.
  - 'xtransformer' : GPU per-bar set-attention (Transformer encoder ACROSS pairs at a bar)
              so a pair's score depends on the whole cross-section that bar. This is the
              architecture that genuinely needs the GPU and fits the cross-sectional thesis.

Label y [T,N] = forward cross-sectional return rank (built in xs_wfo, purged). Models
fit on rows where valid & finite(y). Scores are emitted for ALL bars; invalid pairs get
-inf so the sim excludes them.

Determinism: torch seeded + CUBLAS workspace set in xs_common; lgbm deterministic flags.
"""
import os
import numpy as np
import xs_common as xc


def _rows(X, y, valid):
    """Flatten [T,N,F] + [T,N] → pooled rows where valid & finite(y)."""
    T, N, F = X.shape
    m = valid & np.isfinite(y)
    Xr = X[m]                       # [R,F]
    yr = y[m].astype(np.float64)    # [R]
    return Xr, yr, m


class _LGBM:
    def __init__(self, hp, seed):
        import lightgbm as lgb
        self.lgb = lgb
        self.params = dict(
            objective="regression", n_estimators=int(hp.get("n_estimators", 300)),
            num_leaves=int(hp.get("num_leaves", 31)), max_depth=int(hp.get("max_depth", 6)),
            learning_rate=float(hp.get("lr", 0.05)), subsample=float(hp.get("subsample", 0.8)),
            colsample_bytree=float(hp.get("colsample", 0.7)), reg_lambda=float(hp.get("reg_lambda", 1.0)),
            n_jobs=int(os.environ.get("XS_LGBM_JOBS", hp.get("n_jobs", 1))), random_state=seed, deterministic=True,
            force_col_wise=True, verbosity=-1,
        )
        self.model = None

    def fit(self, X, y, valid):
        Xr, yr, _ = _rows(X, y, valid)
        if Xr.shape[0] < 500:
            return self
        self.model = self.lgb.LGBMRegressor(**self.params).fit(Xr, yr)
        return self

    def score(self, X, valid):
        T, N, F = X.shape
        out = np.full((T, N), -np.inf, np.float32)
        if self.model is None:
            return out
        # Tiled predict to bound RAM: avoid materializing full T*N float64 at once.
        # Each tile is TILE_T bars at a time; float64 cost = TILE_T*N*F*8 bytes.
        TILE_T = int(os.environ.get("XS_SCORE_TILE", "4096"))
        for a in range(0, T, TILE_T):
            b = min(a + TILE_T, T)
            Xslice = X[a:b].reshape((b - a) * N, F)  # float32 view
            p = self.model.predict(Xslice).reshape(b - a, N).astype(np.float32)
            out[a:b][valid[a:b]] = p[valid[a:b]]
        return out


class _TorchPanel:
    """GPU MLP or cross-pair Transformer. kind in {'mlp','xtransformer'}."""
    def __init__(self, kind, hp, seed, nf):
        import torch
        import torch.nn as nn
        self.torch = torch; self.nn = nn
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        self.kind = kind
        self.dev = xc.torch_device()
        h = int(hp.get("hidden", 64)); self.epochs = int(hp.get("epochs", 12))
        self.lr = float(hp.get("lr", 2e-3)); self.bs = int(hp.get("batch", 8192))
        self.nf = nf
        if kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(nf, h), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1),
            ).to(self.dev)
        else:  # xtransformer: encode the SET of pairs at a bar
            self.dmodel = h
            self.inp = nn.Linear(nf, h).to(self.dev)
            layer = nn.TransformerEncoderLayer(d_model=h, nhead=int(hp.get("heads", 4)),
                                               dim_feedforward=2 * h, dropout=0.1,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=int(hp.get("layers", 2))).to(self.dev)
            self.out = nn.Linear(h, 1).to(self.dev)

    def _params(self):
        if self.kind == "mlp":
            return self.net.parameters()
        return list(self.inp.parameters()) + list(self.enc.parameters()) + list(self.out.parameters())

    def fit(self, X, y, valid):
        torch = self.torch
        opt = torch.optim.Adam(self._params(), lr=self.lr)
        lossf = self.nn.MSELoss()
        if self.kind == "mlp":
            Xr, yr, _ = _rows(X, y, valid)
            if Xr.shape[0] < 500:
                self.net = None; return self
            Xt = torch.from_numpy(Xr).to(self.dev); yt = torch.from_numpy(yr).float().to(self.dev).view(-1, 1)
            n = Xt.shape[0]
            for _ in range(self.epochs):
                perm = torch.randperm(n, device=self.dev)
                for a in range(0, n, self.bs):
                    idx = perm[a:a + self.bs]
                    opt.zero_grad(); p = self.net(Xt[idx]); loss = lossf(p, yt[idx])
                    loss.backward(); opt.step()
            return self
        # xtransformer: per-bar sequences (pad to max alive, mask)
        T, N, F = X.shape
        rows = [t for t in range(T) if (valid[t] & np.isfinite(y[t])).sum() >= 8]
        if len(rows) < 50:
            self.enc = None; return self
        for _ in range(self.epochs):
            np.random.shuffle(rows)
            for t in rows:
                m = valid[t] & np.isfinite(y[t])
                xb = torch.from_numpy(X[t][m]).to(self.dev).unsqueeze(0)   # [1,n,F]
                yb = torch.from_numpy(y[t][m]).float().to(self.dev).view(1, -1, 1)
                opt.zero_grad()
                h = self.enc(self.inp(xb)); p = self.out(h)
                loss = lossf(p, yb); loss.backward(); opt.step()
        return self

    def score(self, X, valid):
        torch = self.torch
        T, N, F = X.shape
        out = np.full((T, N), -np.inf, np.float32)
        with torch.no_grad():
            if self.kind == "mlp":
                if self.net is None:
                    return out
                flat = torch.from_numpy(X.reshape(T * N, F)).to(self.dev)
                pred = self.net(flat).cpu().numpy().reshape(T, N)
                out[valid] = pred[valid].astype(np.float32)
                return out
            if self.enc is None:
                return out
            for t in range(T):
                m = valid[t]
                if m.sum() < 1:
                    continue
                xb = torch.from_numpy(X[t][m]).to(self.dev).unsqueeze(0)
                p = self.out(self.enc(self.inp(xb))).cpu().numpy().reshape(-1)
                row = out[t]; row[m] = p.astype(np.float32); out[t] = row
        return out


def make_model(family, hp, seed, nf):
    if family == "lgbm":
        return _LGBM(hp, seed)
    if family in ("mlp", "xtransformer"):
        return _TorchPanel(family, hp, seed, nf)
    # --- model-zoo families (paper completeness; new modules, same .fit/.score) ---
    if family == "xattn":                       # batched cross-pair attention (GPU flagship)
        import xs_models_attn as _attn
        return _attn.make_attn_model(hp, seed, nf)
    if family in ("lstm", "gru", "tcn"):        # temporal sequence (GPU)
        import xs_models_seq as _seq
        return _seq.make_seq_model(family, hp, seed, nf)
    import xs_models_zoo as _zoo                 # tree/linear zoo (CPU)
    if family in _zoo.ZOO_FAMILIES:
        return _zoo.make_model(family, hp, seed, nf)
    raise ValueError(f"unknown family {family}")


HP_GRID = {
    "lgbm": dict(n_estimators=[200, 300, 500], num_leaves=[31, 63], max_depth=[4, 6, 8],
                 lr=[0.03, 0.05, 0.1], subsample=[0.7, 0.9], colsample=[0.6, 0.8], reg_lambda=[0.5, 2.0]),
    "mlp": dict(hidden=[64, 128], lr=[1e-3, 2e-3], epochs=[8, 15], batch=[8192]),
    "xtransformer": dict(hidden=[64, 128], heads=[4], layers=[2, 3], lr=[1e-3, 2e-3], epochs=[6, 10]),
}


def sample_hp(rng, family):
    g = HP_GRID.get(family)
    if g is None:
        # delegate to the zoo modules' own grids
        if family == "xattn":
            import xs_models_attn as _a; return _a.sample_hp(rng, family)
        if family in ("lstm", "gru", "tcn"):
            import xs_models_seq as _s; return _s.sample_hp(rng, family)
        import xs_models_zoo as _z; return _z.sample_hp(rng, family)
    return {k: (v[int(rng.integers(len(v)))] if len(v) else None) for k, v in g.items()}
