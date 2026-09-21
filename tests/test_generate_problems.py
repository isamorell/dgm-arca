"""The worked generator has to be right: it is both the dataset and the ground truth."""

import pytest

from rlm.generate_problems import ShippingCostGenerator, describe
from rlm.verifier import NumericVerifier

GEN = ShippingCostGenerator()


def solve(**overrides) -> str:
    params = {
        "weight_kg": 5.0,
        "distance_km": 250,
        "remote": False,
        "express": False,
        "customer_tier": "none",
        "declared_value": 0.0,
        "order_value": 30.0,
    }
    params.update(overrides)
    return GEN.solve(params)[0]


def test_reference_implementation_on_hand_computed_cases():
    # 4.50 + 0.80*(5-2) = 6.90 of weight, + 3.00 for the 100-500 km band.
    assert solve() == "9.90"
    # 4.50 base, no distance fee, remote x1.25, express x1.6, gold -12%, insurance 1.2% of 200.
    assert (
        solve(
            weight_kg=1.0,
            distance_km=50,
            remote=True,
            express=True,
            customer_tier="gold",
            declared_value=200.0,
            order_value=80.0,
        )
        == "10.32"
    )
    # Order over 60 and not express: the weight charge is waived, the 6.50 band is not.
    assert solve(weight_kg=12.0, distance_km=600, customer_tier="gold", order_value=75.0) == "5.72"


def test_weight_tiers_are_continuous_at_the_boundaries():
    assert GEN.weight_charge(2.0) == (4.50, "tier_0")
    assert GEN.weight_charge(10.0)[0] == pytest.approx(10.90)
    assert GEN.weight_charge(30.0)[0] == pytest.approx(21.90)
    assert GEN.weight_charge(31.0)[1] == "tier_3"


def test_insurance_has_a_floor_and_is_not_discounted():
    cheap = solve(declared_value=50.0)  # 1.2% of 50 = 0.60, below the 1.50 floor
    assert float(cheap) == pytest.approx(9.90 + 1.50)


def test_generation_is_deterministic_and_deduplicated():
    first = GEN.generate(60, "train", seed=7)
    second = GEN.generate(60, "train", seed=7)
    assert [p.question for p in first] == [p.question for p in second]
    assert len({GEN.key(p.params) for p in first}) == 60


def test_ood_split_holds_out_the_heavy_tier():
    train = GEN.generate(80, "train", seed=1)
    ood = GEN.generate(40, "ood", seed=1)
    assert all(p.params["weight_kg"] <= 30 for p in train)
    assert all(p.params["weight_kg"] > 30 for p in ood)
    assert all(p.branches["weight_tier"] == "tier_3" for p in ood)


def test_statements_are_diverse_and_do_not_leak_the_answer():
    stats = describe(GEN.generate(200, "train", seed=3))
    assert stats["n_templates"] == 5
    assert stats["answer_leaked_in_statement"] == 0
    # Every branch of the reference implementation is exercised by the sample.
    assert set(stats["branches"]["weight_tier"]) >= {"tier_0", "tier_1", "tier_2"}
    assert set(stats["branches"]["free_base"]) == {"True", "False"}


def test_answers_are_checkable_with_the_numeric_verifier():
    verifier = NumericVerifier(tolerance=0.01)
    for problem in GEN.generate(20, "test", seed=5):
        assert verifier.is_correct(problem.answer, problem.answer)
        assert not verifier.is_correct("0.00", problem.answer) or problem.answer == "0.00"
