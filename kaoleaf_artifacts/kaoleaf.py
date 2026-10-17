"""
KaoLEAF: Knowledge-Augmented Optimization and Learned Embeddings for Automated Feature Representation
Core package (model + preprocessing + training + explainability + scorer).

Everything the notebook needs to be reproduced / audited / deployed lives here.
"""
import math, json, copy, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
# 1. Feature policy
# ----------------------------------------------------------------------------
CAT_COLS = ["occupation_status", "product_type", "loan_intent"]

# Semantic ordering of feature tokens: the local operator (1-D conv) slides over this axis,
# so neighbouring tokens belong to the same financial "block".
SEMANTIC_ORDER = [
    # block 1: applicant profile
    "age", "occupation_status", "years_employed", "employment_share",
    # block 2: credit behaviour / history
    "credit_score", "credit_history_years", "credit_history_share",
    "defaults_on_file", "delinquencies_last_2yrs", "derogatory_marks", "risk_events",
    # block 3: repayment capacity
    "annual_income", "savings_assets", "current_debt",
    "debt_to_income_ratio", "savings_to_loan", "total_debt_to_income",
    # block 4: the credit request
    "product_type", "loan_intent", "loan_amount", "interest_rate",
    "loan_to_income_ratio", "payment_to_income_ratio", "annual_interest_burden",
]
BLOCK_OF = {}
for _f in SEMANTIC_ORDER[:4]:   BLOCK_OF[_f] = "Profile"
for _f in SEMANTIC_ORDER[4:11]: BLOCK_OF[_f] = "Credit behaviour"
for _f in SEMANTIC_ORDER[11:17]: BLOCK_OF[_f] = "Capacity"
for _f in SEMANTIC_ORDER[17:]:  BLOCK_OF[_f] = "Request"

PRICING_FEATURES = ["interest_rate", "payment_to_income_ratio", "annual_interest_burden"]

BANDS = [  # (lower prob, upper prob, code, label)
    (0.00, 0.20, "E", "Very weak"),
    (0.20, 0.40, "D", "Weak"),
    (0.40, 0.60, "C", "Moderate"),
    (0.60, 0.80, "B", "Strong"),
    (0.80, 1.01, "A", "Very strong"),
]


def engineer(df):
    """Deterministic feature engineering (row-wise, no statistics learned -> no leakage)."""
    d = df.copy()
    inc = d["annual_income"].clip(lower=1)
    adult_years = (d["age"] - 17).clip(lower=1)
    d["total_debt_to_income"] = (d["current_debt"] + d["loan_amount"]) / inc
    d["savings_to_loan"] = d["savings_assets"] / (d["loan_amount"] + 1)
    d["employment_share"] = d["years_employed"] / adult_years
    d["credit_history_share"] = d["credit_history_years"] / adult_years
    d["risk_events"] = d["defaults_on_file"] + d["delinquencies_last_2yrs"] + d["derogatory_marks"]
    d["annual_interest_burden"] = d["loan_amount"] * d["interest_rate"] / 100.0 / inc
    return d


def fmt_value(v):
    """Format a feature value for reason codes: numerics -> 3 significant digits, categoricals (str) -> as-is."""
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return f"{float(v):.3g}"
    return str(v)


def format_reason(feature, value, attribution):
    """One printable reason-code line; safe for numeric AND categorical features."""
    return f"{feature} = {fmt_value(value)}  ({attribution:+.2f} logit)"


def to_score(p):
    """KaoLEAF Score: calibrated approval probability mapped linearly to 300-850."""
    return np.round(300 + 550 * np.clip(np.asarray(p, dtype=float), 0, 1)).astype(int)


def risk_band(p):
    p = np.asarray(p, dtype=float)
    out = np.empty(len(p), dtype=object)
    for lo, hi, code, label in BANDS:
        out[(p >= lo) & (p < hi)] = f"{code} - {label}"
    return out


# ----------------------------------------------------------------------------
# 2. Preprocessing (fit on train only, picklable for inference)
# ----------------------------------------------------------------------------
class Prep:
    def __init__(self, drop=(), seed=42):
        self.drop = list(drop)
        self.seed = seed
        self.order = [f for f in SEMANTIC_ORDER if f not in self.drop]
        self.cat_cols = [f for f in self.order if f in CAT_COLS]
        self.num_cols = [f for f in self.order if f not in CAT_COLS]

    def fit(self, df_eng):
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                      subsample=10**9, random_state=self.seed)
        self.qt.fit(df_eng[self.num_cols].values.astype(float))
        self.levels = {c: sorted(df_eng[c].astype(str).unique().tolist()) for c in self.cat_cols}
        # IG reference point = the *median applicant* expressed in quantile space. NOTE: this is NOT the zero vector:
        # for zero-inflated counts (delinquencies, derogatory marks, defaults) the median is 0 and the quantile transform
        # maps 0 -> about -5 (clipped), so a literal zero baseline would sit in a region of feature space no applicant occupies.
        med = df_eng[self.num_cols].median().values.astype(float)[None, :]
        self.base_num = np.clip(np.nan_to_num(self.qt.transform(med), nan=0.0), -5, 5).astype("float32")
        return self

    @property
    def cat_sizes(self):
        return [len(self.levels[c]) for c in self.cat_cols]

    @property
    def order_idx(self):
        names = self.num_cols + self.cat_cols
        return [names.index(f) for f in self.order]

    @property
    def feature_names(self):  # order of columns in attribution arrays
        return self.num_cols + self.cat_cols

    def transform(self, df_eng):
        xn = self.qt.transform(df_eng[self.num_cols].values.astype(float))
        xn = np.clip(np.nan_to_num(xn, nan=0.0), -5, 5).astype("float32")
        cols = []
        for c in self.cat_cols:
            mp = {l: i for i, l in enumerate(self.levels[c])}
            cols.append(df_eng[c].astype(str).map(mp).fillna(len(mp)).astype("int64").values)  # unknown -> UNK
        xc = np.stack(cols, 1) if cols else np.zeros((len(df_eng), 0), dtype="int64")
        return xn, xc

    def flat(self, xn, xc):
        """Flat matrix for sklearn/GBDT baselines: quantile-numerics + one-hot categoricals."""
        parts = [xn]
        for j, c in enumerate(self.cat_cols):
            oh = np.zeros((len(xc), len(self.levels[c])), dtype="float32")
            ok = xc[:, j] < len(self.levels[c])
            oh[np.where(ok)[0], xc[ok, j]] = 1.0
            parts.append(oh)
        return np.hstack(parts)

    @property
    def flat_names(self):
        names = list(self.num_cols)
        for c in self.cat_cols:
            names += [f"{c}={l}" for l in self.levels[c]]
        return names


# ----------------------------------------------------------------------------
# 3. KaoLEAF architecture
# ----------------------------------------------------------------------------
class PeriodicEmbedding(nn.Module):
    """Per-feature learned periodic embedding of a scalar (Gorishniy et al. style), plus a linear path."""
    def __init__(self, n, d, k=8, sigma=0.5, mode="periodic"):
        super().__init__()
        self.mode = mode
        self.lin_w = nn.Parameter(torch.randn(n, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n, d))
        if mode == "periodic":
            self.c = nn.Parameter(torch.randn(n, k) * sigma)
            self.w = nn.Parameter(torch.randn(n, 2 * k, d) / math.sqrt(2 * k))

    def forward(self, x):  # x: (B, n)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            e = x.unsqueeze(-1) * self.lin_w + self.b
            if self.mode == "periodic":
                v = 2 * math.pi * self.c.unsqueeze(0) * x.unsqueeze(-1)       # (B, n, k)
                f = torch.cat([v.sin(), v.cos()], dim=-1)                      # (B, n, 2k)
                e = e + F.relu(torch.einsum("bnk,nkd->bnd", f, self.w))
        return e


class MHA(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.h, self.drop = heads, drop
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, return_attn=False):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if return_attn:
            a = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
            a = a.softmax(-1)
            o = a @ v
        else:
            a = None
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.out(o), a


class GEGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class AttnBlock(nn.Module):
    def __init__(self, d, heads, drop, ff_mult=2):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = MHA(d, heads, drop)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * ff_mult * d), GEGLU(), nn.Dropout(drop), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, return_attn=False):
        h, a = self.mha(self.ln1(x), return_attn)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, a


class LocalBlock(nn.Module):
    """Gated depthwise-separable 1-D convolution over the ordered feature tokens (local operator)."""
    def __init__(self, d, ks, drop):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, ks, padding=ks // 2, groups=d)
        self.pw = nn.Conv1d(d, 2 * d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # (B, T, d)
        h = self.ln(x).transpose(1, 2)
        h = F.glu(self.pw(self.dw(h)), dim=1).transpose(1, 2)
        return x + self.drop(h)


class KaoLEAF(nn.Module):
    def __init__(self, n_num, cat_sizes, order_idx, d=64, heads=4, layers=3, k=8, sigma=0.5,
                 drop=0.1, ks=3, use_local=True, emb_mode="periodic"):
        super().__init__()
        self.n_num, self.n_cat = n_num, len(cat_sizes)
        self.T = n_num + self.n_cat
        self.use_local = use_local
        self.num_emb = PeriodicEmbedding(n_num, d, k, sigma, emb_mode)
        self.cat_emb = nn.ModuleList([nn.Embedding(s + 1, d) for s in cat_sizes])   # +1 = UNK
        self.feat_emb = nn.Parameter(torch.randn(self.T, d) * 0.02)
        perm = torch.tensor(order_idx, dtype=torch.long)
        self.register_buffer("perm", perm)
        self.register_buffer("inv", torch.argsort(perm))
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.local = nn.ModuleList([LocalBlock(d, ks, drop) if use_local else nn.Identity() for _ in range(layers)])
        self.attn = nn.ModuleList([AttnBlock(d, heads, drop) for _ in range(layers)])
        self.final_ln = nn.LayerNorm(d)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.proj = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU())
        self.head = nn.Linear(d, 1)
        # self-supervised heads (stage 1)
        self.rec_num = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.rec_b = nn.Parameter(torch.zeros(n_num))
        self.rec_cat = nn.ModuleList([nn.Linear(d, s + 1) for s in cat_sizes])

    def tokenize(self, xnum, xcat):
        parts = [self.num_emb(xnum)]
        for j, emb in enumerate(self.cat_emb):
            parts.append(emb(xcat[:, j]).unsqueeze(1))
        e = torch.cat(parts, dim=1)                      # (B, T, d) in [num..., cat...] order
        return e[:, self.perm] + self.feat_emb           # semantic order

    def forward(self, xnum, xcat, mask=None, return_attn=False):
        tok = self.tokenize(xnum, xcat)
        if mask is not None:                             # mask given in [num..., cat...] order
            m = mask[:, self.perm].unsqueeze(-1)
            tok = torch.where(m, (self.mask_tok + self.feat_emb).unsqueeze(0).expand_as(tok).to(tok.dtype), tok)
        B = tok.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1).to(tok.dtype), tok], dim=1)
        attns = []
        for loc, att in zip(self.local, self.attn):
            if self.use_local:
                x = torch.cat([x[:, :1], loc(x[:, 1:])], dim=1)
            x, a = att(x, return_attn)
            if a is not None:
                attns.append(a.mean(1))                  # head-averaged (B, T+1, T+1)
        x = self.final_ln(x)
        cls, toks = x[:, 0], x[:, 1:]
        w = ((toks * self.pool_q).sum(-1) / math.sqrt(toks.shape[-1])).softmax(-1)
        pooled = (w.unsqueeze(-1) * toks).sum(1)
        z = self.proj(torch.cat([cls, pooled], dim=-1))
        return {"logit": self.head(z).squeeze(-1), "z": z, "tokens": toks, "attn": attns, "pool_w": w}

    def mfm_loss(self, xnum, xcat, mask_p=0.15):
        """Stage 1: masked feature modelling (no labels used)."""
        B = xnum.shape[0]
        mask = torch.rand(B, self.T, device=xnum.device) < mask_p
        if not mask.any():
            mask[:, 0] = True
        out = self.forward(xnum, xcat, mask=mask)
        h = out["tokens"][:, self.inv]                    # back to [num..., cat...] order
        pred = ((h[:, :self.n_num] * self.rec_num).sum(-1) + self.rec_b).float()
        mn = mask[:, :self.n_num]
        loss = ((pred - xnum) ** 2)[mn].mean() if mn.any() else pred.sum() * 0
        lc, cnt = 0.0, 0
        for j, head in enumerate(self.rec_cat):
            mj = mask[:, self.n_num + j]
            if mj.any():
                lc = lc + F.cross_entropy(head(h[mj, self.n_num + j]).float(), xcat[mj, j])
                cnt += 1
        return loss + (lc / max(cnt, 1) if cnt else 0.0)


class TabMLP(nn.Module):
    """Plain tabular MLP baseline (same inputs as KaoLEAF)."""
    def __init__(self, n_num, cat_sizes, hidden=(512, 256, 128), drop=0.15, emb_d=8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s + 1, emb_d) for s in cat_sizes])
        layers, i = [], n_num + emb_d * len(cat_sizes)
        for h in hidden:
            layers += [nn.Linear(i, h), nn.ReLU(), nn.Dropout(drop)]
            i = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(i, 1)

    def forward(self, xnum, xcat, **kw):
        x = torch.cat([xnum] + [e(xcat[:, j]) for j, e in enumerate(self.embs)], dim=1)
        return {"logit": self.head(self.body(x)).squeeze(-1)}


def build_model(cfg, prep):
    return KaoLEAF(len(prep.num_cols), prep.cat_sizes, prep.order_idx, **cfg)


# ----------------------------------------------------------------------------
# 4. Training
# ----------------------------------------------------------------------------
def environment_info():
    """Versions of everything that can change numerical results (saved next to the weights)."""
    import platform, sklearn, scipy
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for name in ("xgboost", "lightgbm", "catboost", "optuna", "shap"):
        try: info[name] = __import__(name).__version__
        except Exception: pass
    return info


def _sched(opt, steps, warm):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))


def pretrain_mfm(model, xn, xc, device, epochs=25, lr=1e-3, wd=1e-4, bs=512, mask_p=0.15, seed=0, verbose=True):
    torch.manual_seed(seed)
    model.to(device).train()
    use_amp = device.type == "cuda"
    Xn, Xc = torch.as_tensor(xn, device=device), torch.as_tensor(xc, device=device)
    n = len(Xn)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(n / bs)
    sched = _sched(opt, steps, max(1, steps // 20))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model.mfm_loss(Xn[idx], Xc[idx], mask_p)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()); nb += 1
        hist.append(tot / nb)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  [pretrain] epoch {ep+1:3d}/{epochs}  MFM loss {hist[-1]:.4f}")
    return hist


@torch.no_grad()
def predict_logits(model, xn, xc, device, bs=4096):
    model.eval().to(device)
    out = []
    for i in range(0, len(xn), bs):
        a = torch.as_tensor(xn[i:i + bs], device=device)
        b = torch.as_tensor(xc[i:i + bs], device=device)
        out.append(model(a, b)["logit"].float().cpu())
    return torch.cat(out).numpy()


def train_supervised(model, xn_tr, xc_tr, y_tr, xn_va, xc_va, y_va, device, epochs=60, lr=6e-4,
                     wd=1e-4, bs=512, patience=10, seed=0, head_lr_mult=3.0, verbose=True, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    use_amp = device.type == "cuda"
    Xn, Xc = torch.as_tensor(xn_tr, device=device), torch.as_tensor(xc_tr, device=device)
    Y = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    n = len(Xn)
    head_p = [p for k, p in model.named_parameters() if k.startswith("head")]
    other = [p for k, p in model.named_parameters() if not k.startswith("head")]
    groups = [{"params": other, "lr": lr}] + ([{"params": head_p, "lr": lr * head_lr_mult}] if head_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    steps = epochs * math.ceil(n / bs)
    sched = _sched(opt, steps, max(1, steps // 20))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, bad = -1.0, None, 0
    hist = {"train_loss": [], "val_auc": [], "val_loss": []}
    yv = np.asarray(y_va)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logit = model(Xn[idx], Xc[idx])["logit"]
            loss = F.binary_cross_entropy_with_logits(logit.float(), Y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()); nb += 1
        lv = predict_logits(model, xn_va, xc_va, device)
        auc = roc_auc_score(yv, lv)
        vl = float(F.binary_cross_entropy_with_logits(torch.tensor(lv), torch.tensor(yv, dtype=torch.float32)))
        hist["train_loss"].append(tot / nb); hist["val_auc"].append(auc); hist["val_loss"].append(vl)
        if auc > best + 1e-5:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or ep == epochs - 1 or bad == 0 and ep < 3):
            print(f"  [{tag}] epoch {ep+1:3d}  train {tot/nb:.4f}  val loss {vl:.4f}  val AUC {auc:.5f}  best {best:.5f}")
        if bad >= patience:
            if verbose: print(f"  [{tag}] early stop at epoch {ep+1}")
            break
    model.load_state_dict(best_state)
    hist["best_val_auc"] = best
    return hist


# ----------------------------------------------------------------------------
# 5. Explainability
# ----------------------------------------------------------------------------
def integrated_gradients(model, xnum, xcat, steps=32, baseline=None):
    """IG of the logit w.r.t. the (quantile-normalised) numeric inputs.
    baseline: (1, n_num) or (B, n_num) tensor = median applicant in quantile space (Prep.base_num); None -> zeros."""
    model.eval()
    if baseline is None:
        base = torch.zeros_like(xnum)
    else:
        base = torch.as_tensor(baseline, dtype=xnum.dtype, device=xnum.device).expand_as(xnum)
    total = torch.zeros_like(xnum)
    for s in range(steps):                                   # midpoint rule
        a = (s + 0.5) / steps
        xi = (base + a * (xnum - base)).detach().requires_grad_(True)
        g, = torch.autograd.grad(model(xi, xcat)["logit"].sum(), xi)
        total += g
    return ((xnum - base) * total / steps).detach()


@torch.no_grad()
def categorical_occlusion(model, xnum, xcat):
    """Attribution of each categorical feature = logit(x) - logit(x with that feature set to UNK)."""
    model.eval()
    base = model(xnum, xcat)["logit"]
    cols = []
    for j, emb in enumerate(model.cat_emb):
        x2 = xcat.clone()
        x2[:, j] = emb.num_embeddings - 1
        cols.append(base - model(xnum, x2)["logit"])
    return torch.stack(cols, 1) if cols else torch.zeros(len(xnum), 0, device=xnum.device)


def attributions(models, prep, xn, xc, device, steps=32, temperature=1.0, bs=1024):
    """Ensemble-mean attributions in calibrated-logit units. Columns = prep.feature_names."""
    outs = []
    base = getattr(prep, "base_num", None)                   # median-applicant baseline (older pickles: zeros)
    for i in range(0, len(xn), bs):
        a = torch.as_tensor(xn[i:i + bs], device=device)
        b = torch.as_tensor(xc[i:i + bs], device=device)
        acc = None
        for m in models:
            m.eval().to(device)
            an = integrated_gradients(m, a, b, steps, baseline=base)
            ac = categorical_occlusion(m, a, b)
            cur = torch.cat([an, ac], 1)
            acc = cur if acc is None else acc + cur
        outs.append((acc / len(models) / temperature).cpu().numpy())
    return np.vstack(outs)


# ----------------------------------------------------------------------------
# 6. Deployable scorer
# ----------------------------------------------------------------------------
class KaoLEAFScorer:
    """Load saved artifacts and score raw applicant records (DataFrame / dict)."""

    def __init__(self, models, prep, temperature, cfg, device):
        self.models, self.prep, self.T, self.cfg, self.device = models, prep, float(temperature), cfg, device

    @staticmethod
    def save(path, models, prep, temperature, cfg, extra=None):
        os.makedirs(path, exist_ok=True)
        torch.save({"states": [{k: v.cpu() for k, v in m.state_dict().items()} for m in models],
                    "cfg": cfg, "temperature": float(temperature)}, os.path.join(path, "kaoleaf_weights.pt"))
        joblib.dump(prep, os.path.join(path, "kaoleaf_prep.joblib"))
        meta = {"cfg": cfg, "temperature": float(temperature), "n_members": len(models),
                "features": prep.order, "dropped_features": prep.drop,
                "score_map": "300 + 550 * calibrated_probability", "bands": BANDS}
        meta.update(extra or {})
        with open(os.path.join(path, "kaoleaf_config.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path, device="cpu"):
        device = torch.device(device)
        ck = torch.load(os.path.join(path, "kaoleaf_weights.pt"), map_location=device)
        prep = joblib.load(os.path.join(path, "kaoleaf_prep.joblib"))
        models = []
        for st in ck["states"]:
            m = build_model(ck["cfg"], prep)
            m.load_state_dict(st)
            models.append(m.to(device).eval())
        return cls(models, prep, ck["temperature"], ck["cfg"], device)

    def _prep(self, df):
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        eng = engineer(df)
        return eng, *self.prep.transform(eng)

    def predict_logit(self, df):
        eng, xn, xc = self._prep(df)
        lg = np.mean([predict_logits(m, xn, xc, self.device) for m in self.models], axis=0)
        return lg / self.T

    def predict(self, df):
        lg = self.predict_logit(df)
        p = 1 / (1 + np.exp(-lg))
        return pd.DataFrame({"p_approve": p, "kaoleaf_score": to_score(p), "risk_band": risk_band(p)})

    def explain(self, df, top_k=4, steps=32):
        """Per-applicant reason codes from Integrated Gradients (numeric) + occlusion (categorical)."""
        eng, xn, xc = self._prep(df)
        A = attributions(self.models, self.prep, xn, xc, self.device, steps, self.T)
        names = self.prep.feature_names
        res = []
        for i in range(len(eng)):
            order = np.argsort(A[i])
            val = lambda j: eng.iloc[i][names[j]]            # float for numerics, str for categoricals
            neg = [(names[j], val(j), float(A[i, j])) for j in order[:top_k] if A[i, j] < 0]
            pos = [(names[j], val(j), float(A[i, j])) for j in order[::-1][:top_k] if A[i, j] > 0]
            res.append({"reasons_against_approval": neg, "reasons_for_approval": pos})
        return res, A
