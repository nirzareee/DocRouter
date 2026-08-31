"""Tests for routing policies and their evaluation.

The evaluator replays measured results rather than re-parsing, which makes it
cheap to compare policies -- and makes it easy to get the accounting subtly
wrong in ways that flatter a policy. These tests pin the accounting.
"""

from __future__ import annotations

import json

import pytest

from docrouter.router import (
    always,
    evaluate_oracle,
    evaluate_policy,
    load_features,
    load_results,
    text_layer_rule,
)

FEATURES = {
    "clean_a": {"doc_key": "clean_a", "has_text_layer": True, "feature_time_ms": 900},
    "clean_b": {"doc_key": "clean_b", "has_text_layer": True, "feature_time_ms": 900},
    "scan_a": {"doc_key": "scan_a", "has_text_layer": False, "feature_time_ms": 75},
}

def _r(doc, backend, q, t, c=0.0):
    return {"doc_id": doc, "backend": backend, "overall": q, "wall_time_s": t,
            "cost_usd": c}

RESULTS = {
    ("clean_a", "pylib"): _r("clean_a", "pylib", 0.60, 10.0),
    ("clean_a", "docling"): _r("clean_a", "docling", 0.68, 180.0),
    ("clean_b", "pylib"): _r("clean_b", "pylib", 0.54, 10.0),
    ("clean_b", "docling"): _r("clean_b", "docling", 0.67, 180.0),
    ("scan_a", "pylib"): _r("scan_a", "pylib", 0.00, 0.05),
    ("scan_a", "docling"): _r("scan_a", "docling", 0.65, 320.0),
}


class TestFixedPolicies:
    def test_always_picks_one_backend(self):
        r = evaluate_policy(always("pylib"), FEATURES, RESULTS)
        assert r.choices == {"pylib": 3}
        assert r.mean_quality == pytest.approx((0.60 + 0.54 + 0.00) / 3)

    def test_always_expensive_costs_more(self):
        cheap = evaluate_policy(always("pylib"), FEATURES, RESULTS)
        rich = evaluate_policy(always("docling"), FEATURES, RESULTS)
        assert rich.mean_quality > cheap.mean_quality
        assert rich.total_seconds > cheap.total_seconds


class TestRulesRouter:
    def test_routes_scanned_to_expensive_backend(self):
        r = evaluate_policy(text_layer_rule("pylib", "docling"), FEATURES, RESULTS)
        assert r.choices == {"pylib": 2, "docling": 1}

    def test_beats_always_cheap_on_quality(self):
        cheap = evaluate_policy(always("pylib"), FEATURES, RESULTS)
        rules = evaluate_policy(text_layer_rule(), FEATURES, RESULTS)
        assert rules.mean_quality > cheap.mean_quality

    def test_beats_always_expensive_on_time(self):
        rich = evaluate_policy(always("docling"), FEATURES, RESULTS)
        rules = evaluate_policy(text_layer_rule(), FEATURES, RESULTS)
        assert rules.total_seconds < rich.total_seconds

    def test_policy_sees_only_features(self):
        # A policy that could see scores would be leaking: the router runs
        # before parsing, when quality is unknown by definition.
        seen = {}

        def spy(features):
            seen.update(features)
            return "pylib"

        evaluate_policy(spy, FEATURES, RESULTS)
        assert "overall" not in seen and "wall_time_s" not in seen


class TestOracle:
    def test_oracle_is_an_upper_bound(self):
        oracle = evaluate_oracle(FEATURES, RESULTS, ["pylib", "docling"])
        for policy in (always("pylib"), always("docling"), text_layer_rule()):
            assert oracle.mean_quality >= evaluate_policy(
                policy, FEATURES, RESULTS
            ).mean_quality - 1e-9

    def test_oracle_picks_cheap_when_it_wins(self):
        results = dict(RESULTS)
        results[("clean_a", "pylib")] = _r("clean_a", "pylib", 0.90, 10.0)
        oracle = evaluate_oracle(FEATURES, results, ["pylib", "docling"])
        assert oracle.choices.get("pylib", 0) >= 1


class TestAccounting:
    def test_missing_result_raises_rather_than_skips(self):
        # Silently skipping a document would let a policy look good by being
        # evaluated on fewer documents than its competitors.
        partial = {k: v for k, v in RESULTS.items() if k[0] != "scan_a"}
        with pytest.raises(KeyError, match="scan_a"):
            evaluate_policy(always("pylib"), FEATURES, partial)

    def test_routing_overhead_is_counted(self):
        r = evaluate_policy(text_layer_rule(), FEATURES, RESULTS)
        assert r.routing_overhead_s == pytest.approx((900 + 900 + 75) / 1000)

    def test_all_policies_see_the_same_documents(self):
        n = {
            evaluate_policy(p, FEATURES, RESULTS).n_documents
            for p in (always("pylib"), always("docling"), text_layer_rule())
        }
        assert n == {3}


class TestLoading:
    def test_round_trips_through_jsonl(self, tmp_path):
        rp = tmp_path / "r.jsonl"
        fp = tmp_path / "f.jsonl"
        rp.write_text("\n".join(json.dumps(v) for v in RESULTS.values()))
        fp.write_text("\n".join(json.dumps(v) for v in FEATURES.values()))
        assert len(load_results(rp)) == len(RESULTS)
        assert len(load_features(fp)) == len(FEATURES)
