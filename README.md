<div align="center">

# 🍃 KaoLEAF

### Knowledge-Augmented Optimization and Learned Embeddings for Automated Feature Representation

**Teerapong Panboonyuen**
Government Savings Bank Artificial Intelligence Laboratory (GSBAIL)

[![Project page](https://img.shields.io/badge/project%20page-kaopanboonyuen.github.io%2FKaoLEAF-111111)](https://kaopanboonyuen.github.io/KaoLEAF)
[![Open in Colab](https://img.shields.io/badge/Colab-run%20the%20notebook-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kaopanboonyuen/KaoLEAF/blob/main/KaoLEAF_Loan_Approval_Notebook.ipynb)
[![Status](https://img.shields.io/badge/status-manuscript%20in%20preparation-lightgrey)](#citation)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

</div>

KaoLEAF is an attention-based model for credit decisions on tabular applicant data, released by GSBAIL. It reads an applicant as a sequence of financial concepts, learns a reusable risk embedding, and returns a **calibrated 300–850 score**, a **five-level risk band (A–E)** and **additive reason codes** that a credit officer can read.

> **Status.** This repository reserves the idea and shares a reproducible reference implementation. The paper is in preparation, so numbers and wording may change before publication.

---

## Contents

- [Highlights](#highlights)
- [Results](#results)
- [Method](#method)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the paper numbers](#reproducing-the-paper-numbers)
- [Explainability and audit](#explainability-and-audit)
- [Limitations and governance](#limitations-and-governance)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Highlights

- **Knowledge-ordered tokens.** 24 engineered features become tokens arranged by financial meaning: profile → credit behaviour → capacity → request.
- **Learned periodic embeddings** for numeric features, a **gated depthwise-separable convolution** for local mixing, and **multi-head self-attention** for global interactions.
- **Two-stage learning.** Self-supervised masked-feature modelling, then supervised fine-tuning, repeated over a 5-seed deep ensemble.
- **Calibrated by design.** Temperature scaling on validation, a 300–850 score, and five monotone bands.
- **Auditable.** Integrated Gradients with a completeness check, attention rollout, a SHAP cross-check, ICE/PDP monotonicity tests and a slice/fairness audit.
- **Separated layers.** Representation, outcome prediction and decision policy are independent, so thresholds and hard rules can change without retraining.
- **Reproducible.** Fixed seeds, a SHA-256 check on the dataset and weights, a run manifest, a model card and a reload-and-verify test.

---

## Results

Benchmark: *Loan Approval (US and Canada)*, 50,000 applicants, stratified 70/10/20 split, **10,000 held-out test applicants**. Every model receives the same features, and the tree baselines were tuned and early-stopped on the same validation set.

| Model | Test AUC | PR-AUC | KS | ECE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| **KaoLEAF-Fusion (KaoLEAF + GBDT)** | **0.9852** | **0.9881** | **0.8617** | **0.0096** | **0.0484** |
| LightGBM (Optuna) | 0.9850 | 0.9879 | 0.8607 | 0.0108 | 0.0487 |
| CatBoost | 0.9849 | 0.9879 | 0.8607 | 0.0129 | 0.0489 |
| XGBoost | 0.9842 | 0.9874 | 0.8581 | 0.0137 | 0.0501 |
| HistGradientBoosting | 0.9835 | 0.9866 | 0.8533 | 0.0111 | 0.0511 |
| **KaoLEAF (5-seed ensemble, calibrated)** | 0.9825 | 0.9858 | 0.8478 | 0.0107 | 0.0529 |
| FT-Transformer (plain) | 0.9798 | 0.9837 | 0.8439 | 0.0108 | 0.0564 |
| MLP (embeddings) | 0.9744 | 0.9794 | 0.8199 | 0.0106 | 0.0638 |
| Random Forest | 0.9737 | 0.9789 | 0.8216 | 0.0561 | 0.0683 |
| Extra Trees | 0.9667 | 0.9740 | 0.7964 | 0.0953 | 0.0848 |
| Logistic Regression | 0.9427 | 0.9511 | 0.7289 | 0.0080 | 0.0947 |

**How to read this honestly.**

- KaoLEAF-Fusion ranks first, but its margin over tuned LightGBM is small (+0.0002 AUC), and the 95% bootstrap intervals of the top models overlap.
- The standalone KaoLEAF ensemble is competitive but sits behind the tuned gradient-boosted trees. It outperforms the plain FT-Transformer (+0.0027 AUC), the MLP (+0.0081) and logistic regression (+0.0397).
- In the single-seed ablation, only the periodic embedding shows a consistent gap. Other differences are inside seed noise.
- The case for KaoLEAF is ranking that is competitive with boosting, **plus** a reusable embedding, calibrated scores, stable bands and faithful explanations.

### Score bands (test set)

| Band | Score range | Applicants | Share | Mean predicted | Observed approval |
|---|---|---:|---:|---:|---:|
| A · Very strong | 740–850 | 4,632 | 46.3% | 97.5% | 97.8% |
| B · Strong | 631–740 | 609 | 6.1% | 70.4% | 71.1% |
| C · Moderate | 520–630 | 464 | 4.6% | 50.3% | 57.8% |
| D · Weak | 410–520 | 504 | 5.0% | 29.7% | 33.9% |
| E · Very weak | 300–410 | 3,791 | 37.9% | 2.3% | 2.7% |

Observed approval rises monotonically from E to A and tracks the predicted rate in every band.

---

## Method

Let an applicant be $x=(x^{\text{num}}_{1..m},\,x^{\text{cat}}_{1..c})$. Numerics are mapped to $\tilde x_j\approx\mathcal N(0,1)$ by a quantile transform fitted on the training split only.

**1. Feature tokenisation (learned periodic embeddings).**

$$
\mathbf e_j=\operatorname{ReLU}\Big(W_j\big[\sin(2\pi \mathbf c_j\tilde x_j)\,;\,\cos(2\pi \mathbf c_j\tilde x_j)\big]\Big)+\tilde x_j\,\mathbf a_j+\mathbf b_j,\qquad \mathbf t_j=\mathbf e_j+\mathbf p_j
$$

Categorical features use $\mathbf t_j=E_j[x_j]+\mathbf p_j$. A `[CLS]` token is prepended to give $H^{(0)}\in\mathbb R^{(m+c+1)\times d}$.

**2. Local operator** over the knowledge-ordered token axis:

$$
\tilde H=H+\operatorname{Drop}\big(\operatorname{GLU}(\operatorname{PW}(\operatorname{DW}_{k}(\operatorname{LN}(H))))\big)
$$

**3. Global dependency modelling** (pre-LN attention with a GEGLU feed-forward, repeated for $L$ layers):

$$
H'=\tilde H+\operatorname{MHA}(\operatorname{LN}\tilde H),\qquad H''=H'+\operatorname{FFN}(\operatorname{LN}H')
$$

**4. Behaviour embedding and prediction.**

$$
\mathbf z=\operatorname{GELU}\!\big(W[\operatorname{LN}(h_{\text{cls}})\,;\,\textstyle\sum_i\alpha_i h_i]\big),\qquad f(x)=\mathbf w^\top\mathbf z+b,\qquad \hat p=\sigma\!\big(f(x)/T\big)
$$

**5. Training.** Stage 1 minimises masked-feature reconstruction (MSE on numerics plus cross-entropy on categoricals) without labels. Stage 2 minimises BCE with AdamW, warm-up and cosine decay, AMP, and early stopping on validation AUC. The final logit is the mean over $S=5$ members.

**6. Calibration and score.** $T^\star=\arg\min_T\operatorname{NLL}(\sigma(\bar f/T))$ on validation (fitted $T^\star=1.044$), then

$$
\text{Score}=\operatorname{round}(300+550\,\hat p),\qquad \text{bands }A\text{–}E\text{ at }\hat p\in\{0.8,0.6,0.4,0.2\}
$$

**7. Explanations.** Integrated Gradients in logit space, starting from the median applicant mapped into quantile space (not the zero vector):

$$
\mathrm{IG}_j(x)=(\tilde x_j-\tilde x^{0}_j)\int_0^1\partial_j f\big(\tilde x^{0}+\alpha(\tilde x-\tilde x^{0})\big)\,d\alpha
$$

Categorical features use UNK-occlusion.

**8. KaoLEAF-Fusion.** A logistic stacker over the logits of KaoLEAF and the three strongest gradient-boosted models, fit on validation only. The test set is never touched.

Default configuration: `d=64`, `heads=4`, `layers=3`, `k=8` frequencies, `σ=0.5`, dropout `0.1`, conv kernel `3`, about **190K parameters** per member.

---

## Repository structure

```text
KaoLEAF/
├── index.html                          # project page (GitHub Pages)
├── README.md
├── kaoleaf.bib                         # BibTeX
├── KaoLEAF_Loan_Approval_Notebook.ipynb  # full benchmark, EDA, XAI, audit
├── dataset/
│   └── Loan_Approval_US_CA_DATASET.csv.zip   # copy used by the notebook (see Acknowledgements)
└── kaoleaf_artifacts/
    ├── kaoleaf.py                      # model, preprocessing, training, XAI, scorer
    ├── model/
    │   ├── kaoleaf_weights.pt          # 5 ensemble members + temperature
    │   ├── kaoleaf_prep.joblib         # fitted preprocessor
    │   └── kaoleaf_config.json         # config, features, bands, hashes
    ├── figures/                        # EDA, results, XAI and audit figures
    ├── leaderboard.csv
    ├── metrics.json
    ├── test_predictions.csv            # score, band and probability per test applicant
    ├── run_manifest.json               # library versions, seed, GPU
    └── MODEL_CARD.md
```

---

## Installation

```bash
git clone https://github.com/kaopanboonyuen/KaoLEAF.git
cd KaoLEAF
python -m venv .venv && source .venv/bin/activate

pip install torch numpy pandas scikit-learn scipy joblib matplotlib seaborn
# only needed to re-run the baselines and the SHAP cross-check:
pip install xgboost lightgbm catboost optuna shap
```

The reported runs used Python 3.13, PyTorch 2.11 (CUDA 13.0) and a Colab T4 GPU. The code also runs on CPU, more slowly. Exact versions are in `kaoleaf_artifacts/run_manifest.json`.

---

## Quick start

### Score an applicant with the released weights

```python
import sys
sys.path.insert(0, "kaoleaf_artifacts")
import kaoleaf as kl

scorer = kl.KaoLEAFScorer.load("kaoleaf_artifacts/model", device="cpu")

applicant = dict(
    age=29, occupation_status="Employed", years_employed=4.0, annual_income=52000,
    credit_score=655, credit_history_years=6.0, savings_assets=3500, current_debt=9000,
    defaults_on_file=0, delinquencies_last_2yrs=1, derogatory_marks=0,
    product_type="Personal Loan", loan_intent="Education", loan_amount=15000,
    interest_rate=15.5, debt_to_income_ratio=0.31, loan_to_income_ratio=0.29,
    payment_to_income_ratio=0.10,
)

res = scorer.predict(applicant)
print(res.p_approve[0], res.kaoleaf_score[0], res.risk_band[0])
# P(approve) = 1.000 | KaoLEAF Score = 850 | band = A - Very strong

why, _ = scorer.explain(applicant, top_k=4)
for feature, value, attribution in why[0]["reasons_against_approval"]:
    print(kl.format_reason(feature, value, attribution))
```

This is the notebook's demo applicant. It is one example, not a benchmark.

### Keep decisions as policy, not as model weights

```python
def decision_policy(p, defaults_on_file, approve=0.60, review=0.40):
    if defaults_on_file == 1:
        return "DECLINE (policy rule: default on file)"
    return "APPROVE" if p >= approve else "MANUAL REVIEW" if p >= review else "DECLINE"
```

In the data, `defaults_on_file = 1` is never approved, so it is enforced as a rule rather than learned.

> **Security note.** `kaoleaf_prep.joblib` is a pickle-based file. Load it only from sources you trust.

---

## Reproducing the paper numbers

1. Open the notebook in Colab (badge above) and select **Runtime → Change runtime type → T4 GPU**.
2. Run all cells. A smoke test with `FAST_DEV = True` finishes in minutes. Set it to `False` for the full numbers.
3. The notebook downloads the dataset, checks its SHA-256 against the version used in the paper, trains everything, and writes `kaoleaf_artifacts.zip`.

| Item | How it is handled |
|---|---|
| Seeds | Global `SEED=42`, ensemble members use `SEED+s`, `cudnn.deterministic=True`. GPU kernels can still shift AUC by about 1e-4, so report mean ± std over seeds. |
| Splits | Stratified 70/10/20. The test set is used only for final reporting. |
| Leakage control | Quantile transform and category levels are fit on train only. Engineered ratios are row-wise. |
| Dataset integrity | SHA-256 of the zip is checked and stored in the model card. |
| Environment | Versions and GPU are written to `run_manifest.json`. |
| Reload check | Saved weights are reloaded into a fresh scorer and compared with in-memory predictions (max difference 0). |

Switches in the configuration cell: `DROP_FEATURES` (for example `["age"]` for a fair-lending-safe variant), `N_SEEDS`, `OPTUNA_TRIALS`, `RUN_AUTOGLUON`.

---

## Explainability and audit

- **Global importance** from mean |Integrated Gradients|, shown per feature and per financial block.
- **Cross-check** against LightGBM SHAP: Spearman ρ = 0.72 between two model families and two attribution methods.
- **Reason codes** per applicant, in calibrated-logit units, in the structure of an adverse-action list.
- **Permutation importance and attention rollout.** Attention is reported as a descriptive diagnostic, not a causal explanation. The faithful additive explanation is Integrated Gradients.
- **Monotonicity audit** with ICE/PDP for credit score (↑), DTI (↓), delinquencies (↓) and credit history (↑).
- **Slice audit** by age group, occupation, product and loan intent, with the adverse-impact ratio as a screening statistic, not a legal verdict.
- **Feature-policy sensitivity.** Re-training without `age` and without pricing features, to check how much the model needs them.

All figures are in `kaoleaf_artifacts/figures/` and the full discussion is in the notebook.

---

## Limitations and governance

- **Cross-sectional data.** The benchmark has one row per applicant. The local operator here is a 1-D convolution over *ordered financial features*, not over time. Temporal modelling is not evaluated.
- **Likely simulated data.** The public dataset appears synthetic or realistic-simulated, so results show method behaviour, not real bank performance.
- **Protected attribute.** `age` is protected under US (ECOA) and Canadian lending rules. A variant without it is supported and recommended for any deployment.
- **Possible leakage.** Pricing variables (`interest_rate`, `payment_to_income_ratio`) may be outputs of underwriting in production.
- **Single dataset, single split.** Bootstrap intervals reflect test sampling only, not split variance.
- **Not validated for production lending.** No GSB customer data is used in this repository.

See [`kaoleaf_artifacts/MODEL_CARD.md`](kaoleaf_artifacts/MODEL_CARD.md) for the full model card.

---

## Roadmap

- [ ] Multi-seed, multi-split reporting for the paper
- [ ] Monotonic-constraint variant for regulatory use
- [ ] Knowledge distillation from gradient-boosted trees into KaoLEAF
- [ ] Longer Stage-1 pre-training on larger unlabeled pools
- [ ] Temporal encoder on longitudinal data, planned for GSB data, which cannot be public

---

## Citation

The manuscript is in preparation. Until then, please cite the project:

```bibtex
@misc{panboonyuen2026kaoleaf,
  title  = {{KaoLEAF}: Knowledge-Augmented Optimization and Learned Embeddings
            for Automated Feature Representation},
  author = {Panboonyuen, Teerapong},
  year   = {2026},
  note   = {Government Savings Bank Artificial Intelligence Laboratory (GSBAIL).
            Manuscript in preparation.
            Code and project page: https://kaopanboonyuen.github.io/KaoLEAF}
}
```

---

## Acknowledgements

We thank **parthpatel2130** for releasing the [Realistic Loan Approval Dataset (US and Canada)](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada) on Kaggle, on which all results in this repository were obtained. Please follow the dataset's license terms on its Kaggle page.

---

## License

_Add a `LICENSE` file and name it here (for example MIT or Apache-2.0) before making the repository public._