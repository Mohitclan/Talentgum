"""Fit a rubric to teacher grades: each criterion's metric->score map, then the criterion weights.

Input: assessment results (the JSON `ras assess --out` or the API writes) plus a CSV of teacher grades.
The CSV has one row per recording, a `result` column (path to the result JSON), and one column per
criterion id with the teacher's grade (0-100, or 1-5 with --scale 5). An optional `overall` column
holds the teacher's overall grade.

For each criterion, the metric value is re-computed from the saved scoring inputs, so recordings
assessed under an older rubric still count. The map is a monotone (isotonic) fit of grade on value,
stored as piecewise points. With an overall grade, the weights are fitted by non-negative least
squares on the calibrated criterion scores. Too few graded recordings leave that part of the rubric
unchanged. Nothing here is a neural network: a few dozen graded recordings per song are enough.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from ras.core.errors.analyze import Analysis
from ras.core.rubric import Rubric
from ras.core.rubric.engine import METRICS, MapSpec

MIN_PER_CRITERION = 10


def _value(c, analysis: Analysis) -> float | None:
    for m in c.choices():
        if m.metric == c.metric:
            return METRICS[m.metric](analysis, m.params)
    return None


def calibrate(rubric: Rubric, grades_csv: Path, scale: float = 100.0,
              min_n: int = MIN_PER_CRITERION) -> tuple[dict, dict]:
    """Returns (new rubric as a dict, report)."""
    rows = list(csv.DictReader(Path(grades_csv).open()))
    analyses = []
    for row in rows:
        res = json.loads(Path(row["result"]).read_text())
        analyses.append(Analysis.from_scoring_inputs(res.get("scoring_inputs", {})))
    out = rubric.model_dump(by_alias=True)
    report: dict = {"recordings": len(rows), "criteria": {}}
    calibrated_scores = np.full((len(rows), len(rubric.criteria)), np.nan)

    for j, c in enumerate(rubric.criteria):
        pairs = []
        for i, (row, a) in enumerate(zip(rows, analyses)):
            g = row.get(c.id, "").strip()
            v = _value(c, a)
            if g and v is not None:
                pairs.append((i, float(v), float(g) * 100.0 / scale))
        info: dict = {"n": len(pairs)}
        crit = out["criteria"][j]
        if len(pairs) >= min_n:
            idx, x, y = (np.array(t) for t in zip(*pairs))
            rho = spearmanr(x, y).statistic
            info["spearman"] = None if np.isnan(rho) else round(float(rho), 3)
            iso = IsotonicRegression(increasing=bool(rho >= 0), out_of_bounds="clip", y_min=0, y_max=100).fit(x, y)
            qs = np.unique(np.quantile(x, np.linspace(0, 1, 7)))
            points = [[round(float(q), 4), round(float(iso.predict([q])[0]), 1)] for q in qs]
            before = np.array([c.map.apply(v) for v in x])
            new_map = MapSpec(type="piecewise", points=[tuple(p) for p in points])
            after = np.array([new_map.apply(v) for v in x])
            info["mae_before"] = round(float(np.mean(np.abs(before - y))), 2)
            info["mae_after"] = round(float(np.mean(np.abs(after - y))), 2)
            crit["map"] = {"type": "piecewise", "points": points}
            calibrated_scores[idx.astype(int), j] = after
        else:
            info["kept"] = f"fewer than {min_n} graded recordings"
            for i, a in enumerate(analyses):
                v = _value(c, a)
                if v is not None:
                    calibrated_scores[i, j] = c.map.apply(v)
        report["criteria"][c.id] = info

    overall = [(i, float(r["overall"]) * 100.0 / scale) for i, r in enumerate(rows) if r.get("overall", "").strip()]
    if len(overall) >= max(3 * len(rubric.criteria), 30):
        idx = [i for i, _ in overall]
        X = calibrated_scores[idx]
        col_mean = np.nanmean(np.where(np.isnan(X), np.nan, X), axis=0)
        X = np.where(np.isnan(X), np.nan_to_num(col_mean, nan=50.0), X)
        w, _ = nnls(X, np.array([g for _, g in overall]))
        if w.sum() > 0:
            w = w / w.sum()
            w = np.maximum(w, 0.005)  # keep every criterion visible in the report
            w = w / w.sum()
            for j, c in enumerate(out["criteria"]):
                c["weight"] = round(float(w[j]), 4)
            drift = 1.0 - sum(c["weight"] for c in out["criteria"])  # rounding: keep the sum exactly 1
            top = out["criteria"][int(np.argmax(w))]
            top["weight"] = round(top["weight"] + drift, 4)
            report["weights"] = "fitted"
    else:
        report["weights"] = f"kept (needs {max(3 * len(rubric.criteria), 30)} recordings with an overall grade)"
    out["version"] = int(out.get("version", 1)) + 1
    out["description"] = (out.get("description", "") + f" Calibrated on {len(rows)} teacher-graded recordings.").strip()
    Rubric.model_validate(out)
    return out, report
