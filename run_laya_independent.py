#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

SCORE_LEVELS = [
    "very weak evidence; strong risk of relative underperformance",
    "weak evidence; more likely to underperform than outperform",
    "slightly weak evidence",
    "neutral or mixed evidence",
    "slightly strong evidence",
    "strong evidence; more likely to outperform than underperform",
    "very strong evidence; strong relative-outperformance setup",
]

SINGLE_CONTEXT = (
    "Assess one anonymized ETF candidate independently. The horizon is the next 21 trading sessions. "
    "The candidate identity and date are hidden. Use only the contemporaneous numerical features in the state. "
    "Higher hybrid24_score and rank features mean stronger model preference. Positive past return and momentum, "
    "trend quality, drawdown, volatility and breadth are contextual evidence; do not use any one feature mechanically. "
    "Judge prospective relative strength versus a typical peer, not absolute certainty of a positive market return."
)


def questions(variant: str):
    if variant == "score7":
        return {
            "quality": {
                "type": "score",
                "instructions": SINGLE_CONTEXT + " Rate the strength of evidence for relative outperformance.",
                "criteria": SCORE_LEVELS,
            }
        }
    if variant == "noul_single":
        return {
            "outperform_typical": {
                "type": "noul",
                "instructions": SINGLE_CONTEXT + " Is this candidate more likely than not to outperform a typical peer over the horizon?",
                "criteria": {
                    "true": "more likely than not to outperform a typical peer",
                    "false": "not more likely than not to outperform a typical peer",
                },
            }
        }
    raise ValueError(variant)


def ece(y: np.ndarray, p: np.ndarray, k: int = 10):
    edges = np.linspace(0, 1, k + 1)
    bins = np.clip(np.digitize(p, edges, right=True) - 1, 0, k - 1)
    total = 0.0
    rows = []
    for j in range(k):
        m = bins == j
        if not m.any():
            continue
        mp = float(p[m].mean())
        wr = float(y[m].mean())
        n = int(m.sum())
        total += n / len(y) * abs(mp - wr)
        rows.append({"bin": j, "n": n, "mean_p": mp, "win_rate": wr})
    return float(total), rows


def score_distribution(answer: dict) -> np.ndarray:
    probs = answer["probabilities"]
    out = np.array([float(probs[str(i)]) for i in range(len(SCORE_LEVELS))], dtype=float)
    out = np.clip(out, 0.0, None)
    s = out.sum()
    if s <= 0:
        raise ValueError("score distribution has no mass")
    return out / s


def ordinal_pair_probability(a: np.ndarray, b: np.ndarray) -> float:
    # Independent latent ordinal scores: P(S_A>S_B) + 0.5*P(tie).
    outer = np.outer(a, b)
    win = float(np.tril(outer, k=-1).sum())  # row index A > col index B
    tie = float(np.trace(outer))
    return win + 0.5 * tie


def odds_pair_probability(pa: float, pb: float) -> float:
    # Convert two independent probabilities-vs-typical into a Bradley-Terry-style pair probability.
    eps = 1e-6
    pa = min(max(float(pa), eps), 1 - eps)
    pb = min(max(float(pb), eps), 1 - eps)
    la = math.log(pa / (1 - pa))
    lb = math.log(pb / (1 - pb))
    d = max(min(la - lb, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-d))


def build_single_states(requests_path: str):
    rows = []
    with open(requests_path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if not x["case_id"].endswith("-o"):
                continue
            pair_id = x["case_id"][:-2]
            state = x["state"]
            for side in ("A", "B"):
                rows.append(
                    {
                        "single_id": f"{pair_id}-{side}",
                        "pair_id": pair_id,
                        "side": side,
                        # Deliberately remove candidate_A/candidate_B naming.
                        "state": {"candidate": state[f"candidate_{side}"]},
                    }
                )
    return rows


def run_variant(agent, single_rows, variant, batch_size, max_len, head_max_len):
    qs = questions(variant)
    states = [r["state"] for r in single_rows]
    t0 = time.perf_counter()
    if hasattr(agent, "predict_batch"):
        results = agent.predict_batch(
            states,
            qs,
            batch_size=batch_size,
            sort_by_length=True,
            max_len=max_len,
            head_max_len=head_max_len,
        )
    else:
        results = [
            agent.predict(s, qs, max_len=max_len, head_max_len=head_max_len)
            for s in states
        ]
    infer_s = time.perf_counter() - t0

    pred_rows = []
    for meta, res in zip(single_rows, results):
        row = {k: meta[k] for k in ("single_id", "pair_id", "side")}
        if variant == "score7":
            ans = res["answers"]["quality"]
            dist = score_distribution(ans)
            row["score"] = float(ans["score"])
            row["dist"] = dist.tolist()
        else:
            ans = res["answers"]["outperform_typical"]
            row["p_typical"] = float(ans["noul"])
        pred_rows.append(row)
    return pred_rows, infer_s


def evaluate(pred_rows, labels_path, variant):
    lab = pd.read_csv(labels_path)
    ymap = (
        lab[lab.orientation.eq("original")][["pair_id", "a_wins"]]
        .drop_duplicates("pair_id")
        .set_index("pair_id")["a_wins"]
        .astype(int)
        .to_dict()
    )

    by_pair = {}
    for r in pred_rows:
        by_pair.setdefault(r["pair_id"], {})[r["side"]] = r

    pair_rows = []
    for pair_id, sides in by_pair.items():
        if set(sides) != {"A", "B"} or pair_id not in ymap:
            continue
        a, b = sides["A"], sides["B"]
        if variant == "score7":
            da = np.asarray(a["dist"], dtype=float)
            db = np.asarray(b["dist"], dtype=float)
            p = ordinal_pair_probability(da, db)
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "a_wins": ymap[pair_id],
                    "p_A": p,
                    "score_A": a["score"],
                    "score_B": b["score"],
                    "delta": a["score"] - b["score"],
                }
            )
        else:
            p = odds_pair_probability(a["p_typical"], b["p_typical"])
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "a_wins": ymap[pair_id],
                    "p_A": p,
                    "p_typical_A": a["p_typical"],
                    "p_typical_B": b["p_typical"],
                    "delta": a["p_typical"] - b["p_typical"],
                }
            )

    df = pd.DataFrame(pair_rows)
    y = df.a_wins.to_numpy(int)
    p = df.p_A.to_numpy(float)
    metrics = {
        "variant": variant,
        "n_pairs": int(len(df)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_p_A": float(p.mean()),
        "empirical_A_win_rate": float(y.mean()),
        "fraction_choose_A": float((p >= 0.5).mean()),
        "fraction_exact_tie": float(np.isclose(p, 0.5, atol=1e-12).mean()),
        "delta_mean": float(df.delta.mean()),
        "delta_std": float(df.delta.std(ddof=0)),
    }
    metrics["ece10"], cal = ece(y, p)

    # Confidence here means distance from an exactly indifferent pair probability.
    df["confidence"] = (df.p_A - 0.5).abs()
    df["q"] = pd.qcut(df.confidence.rank(method="first"), 5, labels=False) + 1
    q = (
        df.groupby("q")
        .agg(n=("a_wins", "size"), mean_confidence=("confidence", "mean"), accuracy=("a_wins", lambda s: np.nan))
        .reset_index()
    )
    # Compute accuracy per confidence quintile explicitly.
    q_rows = []
    for qq, g in df.groupby("q"):
        pred = (g.p_A >= 0.5).astype(int)
        q_rows.append(
            {
                "q": int(qq),
                "n": int(len(g)),
                "mean_confidence": float(g.confidence.mean()),
                "accuracy": float((pred == g.a_wins).mean()),
                "mean_p_A": float(g.p_A.mean()),
                "a_win_rate": float(g.a_wins.mean()),
            }
        )
    return df, metrics, cal, q_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="data/pilot_requests.jsonl")
    ap.add_argument("--labels", default="data/pilot_labels_70.csv")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--outdir", default="results_independent")
    a = ap.parse_args()

    import laya

    single_rows = build_single_states(a.requests)
    assert len(single_rows) == 140, len(single_rows)
    # Leakage guard on what is actually passed to Laya.
    serialized = json.dumps([r["state"] for r in single_rows])
    for banned in ("candidate_A", "candidate_B", "ticker", "signal_date", "fwd_ret", "a_wins"):
        assert banned not in serialized, banned

    t0 = time.perf_counter()
    agent = laya.load(a.model, device="cpu")
    load_s = time.perf_counter() - t0

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"model": a.model, "n_single_candidates": len(single_rows), "model_load_s": load_s, "variants": {}}

    for variant in ("score7", "noul_single"):
        pred_rows, infer_s = run_variant(
            agent, single_rows, variant, a.batch_size, a.max_len, a.head_max_len
        )
        pair_df, metrics, cal, quintiles = evaluate(pred_rows, a.labels, variant)
        metrics["inference_s"] = infer_s
        metrics["candidate_requests_per_s"] = float(len(single_rows) / infer_s)
        metrics["max_len"] = a.max_len
        metrics["head_max_len"] = a.head_max_len
        summary["variants"][variant] = metrics

        # JSON-friendly single-candidate predictions.
        (out / f"single_predictions_{variant}.json").write_text(json.dumps(pred_rows, indent=2) + "\n")
        pair_df.to_csv(out / f"pair_predictions_{variant}.csv", index=False)
        (out / f"report_{variant}.json").write_text(
            json.dumps({"metrics": metrics, "calibration_bins": cal, "confidence_quintiles": quintiles}, indent=2) + "\n"
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
