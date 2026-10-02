# LayaETF

Leakage-safe research harness for testing Laya as a pairwise top-1/top-2 decision layer for ETF_trader / Hybrid24.

The public repository intentionally contains only anonymized A/B feature states and randomized case identifiers. It does **not** publish ticker symbols, dates, or forward returns.

The GitHub Actions pilot compares two Laya formulations on the same sample:

- `noul_neutral`: yes/no with neutral A/B labels to reduce the documented true/false label-bias risk.
- `choice`: explicit two-option A-vs-B choice.

Initial smoke sample: 140 independent pairs, stratified 20 per year over 2020-2026, with each pair also evaluated after swapping A/B (280 requests per formulation).
