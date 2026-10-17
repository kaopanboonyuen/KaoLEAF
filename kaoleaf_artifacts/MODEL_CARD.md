# Model Card — KaoLEAF (Knowledge-Augmented Optimization and Learned Embeddings for Automated Feature Representation)
**Version:** 1.0 · **Date:** 2026-10-17 · **Author:** T. Panboonyuen (GSBAIL)

## Intended use
Research benchmark / reference implementation of an attention-based credit-decision model with calibrated scores and explanations.
**Not validated for production lending.** Public dataset is synthetic/realistic-simulated; no GSB customer data used (AI-governance & ethics).

## Data
Loan Approval (US & Canada), Kaggle (parthpatel2130). n=50,000; sha256 `aa6435d06ea46ec256b79a2780faee807ba63334df4c4ce507aa9bc2097ae61f`. Split 70/10/20 stratified, seed 42.
Features used: age, occupation_status, years_employed, employment_share, credit_score, credit_history_years, credit_history_share, defaults_on_file, delinquencies_last_2yrs, derogatory_marks, risk_events, annual_income, savings_assets, current_debt, debt_to_income_ratio, savings_to_loan, total_debt_to_income, product_type, loan_intent, loan_amount, interest_rate, loan_to_income_ratio, payment_to_income_ratio, annual_interest_burden. Dropped: none.

## Model
Periodic feature tokeniser → gated 1-D conv over semantically ordered tokens → 3× self-attention (d=64) → embedding → logistic head.
Stage 1: masked-feature modelling (25 ep). Stage 2: supervised (60 ep max, early stopping). 5-member ensemble, temperature T=1.044.
Score = round(300 + 550·p). Bands: A ≥0.8, B ≥0.6, C ≥0.4, D ≥0.2, E <0.2. Weights sha256 `d8086702aaaa871a35187e61e7a4a71f21f1e49bc642c430088753cfce5f2560`.

## Test performance (calibrated ensemble)
AUC=0.9825, PR-AUC=0.9858, KS=0.8478, LogLoss=0.1679, Brier=0.0529, ECE=0.0107, Acc@0.5=0.9241, F1@0.5=0.9309
Best model by test AUC: **KaoLEAF-Fusion (LEAF + GBDT)** (0.9852). Best baseline: LightGBM (Optuna) (0.9850).

## Explainability
Integrated Gradients (numeric) + UNK-occlusion (categorical), attention rollout, SHAP cross-check (Spearman ρ=0.72), ICE/PDP & monotonicity audit.

## Known limitations / governance
* Cross-sectional data: temporal behaviour modelling not evaluated here.
* `age` is a protected attribute (AUC with/without: see sensitivity table). Recommend `DROP_FEATURES=["age"]` for deployment.
* Pricing features may leak underwriting outcome in production.
* `defaults_on_file=1` is a deterministic decline in the data; enforced as policy.
* Single dataset, single split; bootstrap CIs reflect test sampling only, not split variance.
