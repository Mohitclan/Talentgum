from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ras.core.align import match_notes
from ras.core.errors import analyze
from ras.core.rubric import MapSpec, Rubric, load_rubric, score_analysis
from ras.synth.performance import PerfSpec, simulate_performance

RUBRIC = Path(__file__).resolve().parents[1] / "data" / "rubrics" / "dev_placeholder.json"


def _analysis(ref, **spec):
    t = simulate_performance(ref, PerfSpec(**spec))
    return analyze(ref, t.notes, match_notes(ref, t.notes))


def _rubric(criteria):
    return Rubric.model_validate({"rubric_id": "t", "criteria": criteria,
                                  "score_bands": {"low": [0, 50], "high": [50, 100]}})


def test_placeholder_rubric_is_valid():
    r = load_rubric(RUBRIC)
    assert abs(sum(c.weight for c in r.criteria) - 1) < 1e-9


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        _rubric([{"id": "a", "weight": 0.5, "metric": "note_accuracy_pct"}])


def test_unknown_metric_rejected():
    with pytest.raises(ValidationError):
        _rubric([{"id": "a", "weight": 1.0, "metric": "vibes"}])


def test_maps():
    lin = MapSpec(**{"type": "linear", "from": [40, 100], "to": [0, 100]})
    assert lin.apply(40) == 0 and lin.apply(100) == 100 and lin.apply(70) == 50
    assert lin.apply(10) == 0 and lin.apply(120) == 100  # clipped
    inverse = MapSpec(**{"type": "linear", "from": [30, 0], "to": [0, 100]})  # lower is better
    assert inverse.apply(0) == 100 and inverse.apply(30) == 0
    step = MapSpec(type="step", points=[(0, 100), (2, 50), (5, 0)])
    assert (step.apply(0), step.apply(1), step.apply(2), step.apply(9)) == (100, 100, 50, 0)
    pw = MapSpec(type="piecewise", points=[(0, 0), (10, 100)])
    assert pw.apply(5) == 50


def test_perfect_performance_scores_high(twinkle):
    res = score_analysis(load_rubric(RUBRIC), _analysis(twinkle, seed=1))
    assert res.total > 90 and res.band == "excellent"
    assert res.assessed_weight == 1.0


def test_more_mistakes_score_lower(twinkle):
    rubric = load_rubric(RUBRIC)
    good = score_analysis(rubric, _analysis(twinkle, seed=1)).total
    bad = score_analysis(rubric, _analysis(twinkle, seed=1, n_wrong_pitch=10, n_missing=10)).total
    assert bad < good - 20


def test_unmeasurable_criterion_is_not_guessed(twinkle):
    # a keyboard with flat velocity cannot be assessed for dynamics; the total must ignore it
    rubric = _rubric([
        {"id": "notes", "weight": 0.6, "metric": "note_accuracy_pct"},
        {"id": "dyn", "weight": 0.4, "metric": "dynamic_variation_vel_std",
         "map": {"type": "linear", "from": [0, 15], "to": [0, 100]}},
    ])
    flat = _analysis(twinkle, seed=1, velocity_std=0.0)
    res = score_analysis(rubric, flat)
    assert res.criteria["dyn"]["status"] == "not_assessed" and res.criteria["dyn"]["score"] is None
    assert res.assessed_weight == 0.6
    assert res.total == res.criteria["notes"]["score"]  # renormalised over what was assessed
    varied = score_analysis(rubric, _analysis(twinkle, seed=1, velocity_std=10.0))
    assert varied.criteria["dyn"]["status"] == "assessed"


def test_metric_values_are_plain_numbers(twinkle):
    res = score_analysis(load_rubric(RUBRIC), _analysis(twinkle, seed=1))
    for c in res.criteria.values():
        assert c["value"] is None or isinstance(c["value"], float)
    assert not np.isnan(res.total)
