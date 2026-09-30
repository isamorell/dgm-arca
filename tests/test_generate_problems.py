"""The medication generator has to be right: it is both the dataset and the ground truth."""

from collections import Counter
from fractions import Fraction

import pytest

from rlm.generate_problems import (
    ANSWER_UNIT,
    BOUNDS,
    DECIMALS,
    DEFAULT_SEED,
    MASS_TO_MG,
    UNIT_ALIASES,
    MedicationGenerator,
    describe,
    fmt,
    render_time,
    round_half_up,
)
from rlm.verifier import MedicationVerifier, NumericVerifier

GEN = MedicationGenerator()
CONTINUOUS = {"infusion", "infusion_inverse"}


def solve(**params) -> str:
    """Answer text for a hand-written problem (decimals follow the family)."""
    params.setdefault("decimals", DECIMALS[params["family"]])
    params.setdefault("decimal_comma", False)
    return GEN.solve(params)[0]


def families(problems) -> Counter:
    return Counter(p.params["family"] for p in problems)


# ----------------------------------------------------------------------------- reference solver
def test_reference_implementation_on_hand_computed_cases():
    # 500 mL in 4 h -> 125 mL/h.
    assert solve(family="pump", volume_ml=500, time_min=240) == "125.0"
    # 750 mL in 5 h (300 min) with a 20 drops/mL set: 750 * 20 / 300 = 50 drops/min.
    assert solve(family="drops", volume_ml=750, time_min=300, drop_factor=20) == "50"
    # 60 kg x 0.05 g/kg/day = 3 g = 3000 mg per day, / 4 doses = 750 mg.
    assert (
        solve(family="dose_day", weight_kg=60, dose_value=0.05, dose_unit="g", n_doses=4)
        == "750.0"
    )
    # 75 kg x 2.5 mg/kg = 187.5 mg; vial 500 mg / 10 mL = 50 mg/mL -> 3.75 mL.
    assert (
        solve(family="volume", weight_kg=75, dose_value=2.5, dose_unit="mg", conc_mass=500,
              conc_mass_unit="mg", conc_volume_ml=10)
        == "3.75"
    )
    # 65 kg x 5 mcg/kg/min = 19.5 mg/h; bag 4 mg / 50 mL = 0.08 mg/mL -> 243.75 -> 243.8 (half up).
    assert (
        solve(family="infusion", weight_kg=65, dose_mcg_kg_min=5, bag_mass_mg=4, bag_volume_ml=50)
        == "243.8"
    )
    # The inverse of the previous one: 243.75 mL/h gives back 5 mcg/kg/min.
    assert (
        solve(family="infusion_inverse", weight_kg=65, rate_ml_h=243.75, bag_mass_mg=4,
              bag_volume_ml=50)
        == "5.00"
    )
    # 0.5 g = 500 mg at 5 mg/mL -> 100 mL final; minus 5 mL used to reconstitute -> 95 mL diluent.
    dilution = dict(family="dilution", dose_value=0.5, dose_unit="g", target_conc_mg_ml=5)
    assert solve(**dilution, ask="final") == "100.0"
    assert solve(**dilution, ask="diluent", recon_volume_ml=5) == "95.0"


def test_unit_conversions_are_continuous_at_the_boundaries():
    # The three mass units hang from one table: 1 g = 1000 mg and 1 mcg = 0.001 mg.
    assert MASS_TO_MG["g"] == 1000
    assert MASS_TO_MG["mg"] == 1
    assert MASS_TO_MG["mcg"] == Fraction(1, 1000)
    # The same dose written in g, mg or mcg gives the same answer.
    base = dict(family="dose_day", weight_kg=20, n_doses=4)
    assert (
        solve(**base, dose_value=0.05, dose_unit="g")
        == solve(**base, dose_value=50, dose_unit="mg")
        == solve(**base, dose_value=50000, dose_unit="mcg")
    )
    # A vial written in g or in mg is the same vial.
    vial = dict(family="volume", weight_kg=10, dose_value=5, dose_unit="mg", conc_volume_ml=10)
    assert solve(**vial, conc_mass=0.5, conc_mass_unit="g") == solve(
        **vial, conc_mass=500, conc_mass_unit="mg"
    )


def test_rounding_is_half_up_not_bankers():
    # 250 mL x 10 drops/mL / 200 min = 12.5 drops/min -> 13 (round() would give 12).
    assert solve(family="drops", volume_ml=250, time_min=200, drop_factor=10) == "13"
    # 11 kg x 5 mg / 8 doses = 6.875 mg -> 6.9.
    assert solve(family="dose_day", weight_kg=11, dose_value=5, dose_unit="mg", n_doses=8) == "6.9"
    assert round_half_up(Fraction(5, 2), 0) == 3
    assert round_half_up(Fraction(25, 100), 1) == Fraction(3, 10)


def test_diluent_is_the_final_volume_minus_the_reconstitution_and_stays_positive():
    final = solve(family="dilution", dose_value=1, dose_unit="g", target_conc_mg_ml=10, ask="final")
    diluent = solve(family="dilution", dose_value=1, dose_unit="g", target_conc_mg_ml=10,
                    ask="diluent", recon_volume_ml=5)
    assert float(final) - float(diluent) == pytest.approx(5)
    # A reconstitution volume larger than the final volume is not a valid problem.
    with pytest.raises(ValueError):
        solve(family="dilution", dose_value=100, dose_unit="mg", target_conc_mg_ml=20,
              ask="diluent", recon_volume_ml=10)


def test_branches_report_level_and_conversion():
    final = dict(family="dilution", dose_value=1, dose_unit="g", target_conc_mg_ml=10, ask="final",
                 decimals=1, decimal_comma=False)
    diluent = {**final, "ask": "diluent", "recon_volume_ml": 5}
    assert GEN.solve(final)[1]["level"] == "1"
    assert GEN.solve(diluent)[1]["level"] == "2"
    assert GEN.solve(final)[1]["mass_unit_conversion"] == "True"  # written in g


# ----------------------------------------------------------------------------- generation
def test_generation_is_deterministic_and_deduplicated():
    first = GEN.generate(60, "train", seed=7)
    second = GEN.generate(60, "train", seed=7)
    assert [p.question for p in first] == [p.question for p in second]
    assert len({GEN.key(p.params) for p in first}) == 60


def test_default_seed_is_reproducible_and_another_seed_is_not():
    a = MedicationGenerator().generate(60, "train", seed=DEFAULT_SEED)
    b = MedicationGenerator().generate(60, "train", seed=DEFAULT_SEED)
    c = MedicationGenerator().generate(60, "train", seed=DEFAULT_SEED + 1)
    assert [(p.question, p.answer) for p in a] == [(p.question, p.answer) for p in b]
    assert [p.question for p in a] != [p.question for p in c]


def test_families_are_balanced():
    assert set(families(GEN.generate(100, "train")).values()) == {20}  # 5 families x 20
    # The remainder of the division goes to the first families.
    assert sorted(families(GEN.generate(103, "train")).values()) == [20, 20, 21, 21, 21]


def test_ood_split_holds_out_the_continuous_infusion():
    train = GEN.generate(80, "train", seed=1)
    ood = GEN.generate(40, "ood", seed=1)
    assert all(p.params["family"] not in CONTINUOUS for p in train)
    assert all(p.params["family"] in CONTINUOUS for p in ood)
    assert all(p.branches["level"] == "4" for p in ood)


def test_weight_mode_holds_out_the_low_weights():
    generator = MedicationGenerator(ood_mode="weight")
    train = generator.generate(80, "train", seed=1)
    ood = generator.generate(40, "ood", seed=1)
    assert all(p.params.get("weight_kg", 99) >= 10 for p in train)
    assert all(p.params["weight_kg"] < 10 for p in ood)
    assert all(p.branches["low_weight"] == "True" for p in ood)


def test_avoid_keeps_test_disjoint_from_train():
    generator = MedicationGenerator()
    train = generator.generate(200, "train")
    generator.avoid = {generator.key(p.params) for p in train}
    test = generator.generate(60, "test")
    assert not generator.avoid & {generator.key(p.params) for p in test}


def test_answers_have_the_asked_decimals_and_stay_inside_the_bounds():
    for p in GEN.generate(300, "train", seed=4):
        decimals = p.params["decimals"]
        assert "," not in p.answer  # stored with a dot, whatever the statement uses
        assert (len(p.answer.split(".")[1]) == decimals) if decimals else ("." not in p.answer)
        low, high = BOUNDS[p.params["family"]]
        assert low <= float(p.answer) <= high


# ----------------------------------------------------------------------------- statements
def test_statements_are_diverse_and_do_not_leak_the_answer():
    problems = GEN.generate(200, "train", seed=3)
    stats = describe(problems)
    assert stats["n_templates"] == 8  # cohesive ids 0-3 and compositional ids 100-103
    # Whole-number coincidences (a drops answer equal to the drop factor, the volume...) are
    # expected and harmless; a real leak would be far more frequent than this.
    assert stats["answer_leaked_in_statement"] / len(problems) < 0.05
    # Every branch of the reference implementation is exercised by the sample.
    branches = stats["branches"]
    assert set(branches["family"]) == {"pump", "drops", "dose_day", "volume", "dilution"}
    assert set(branches["level"]) == {"1", "2", "3"}
    assert set(branches["time_style"]) >= {"h", "min", "hmin"}
    assert set(branches["drop_factor"]) >= {"10", "15", "20", "60"}
    assert set(branches["mass_unit_conversion"]) == {"True", "False"}
    assert set(branches["decimal_comma"]) == {"True", "False"}


def test_number_and_time_formatting():
    assert fmt(0.05, comma=True) == "0,05"
    assert fmt(0.05, comma=False) == "0.05"
    assert fmt(5.0, comma=True) == "5"
    assert render_time(240, "h") == "4 h"
    assert render_time(240, "min") == "240 min"
    assert render_time(90, "hmin") == "1 h 30 min"


def test_statement_uses_the_decimal_comma_when_asked():
    problems = GEN.generate(200, "train", seed=2)
    with_comma = [
        p for p in problems if p.params["decimal_comma"] and p.params["family"] == "dose_day"
    ]
    assert with_comma
    for p in with_comma:
        assert fmt(p.params["weight_kg"], comma=True) in p.question


# ----------------------------------------------------------------------------- verifier
def test_answers_are_checkable_with_the_numeric_verifier():
    verifier = NumericVerifier(tolerance=0.01)
    for problem in GEN.generate(20, "test", seed=5):
        assert verifier.is_correct(problem.answer, problem.answer)
        assert not verifier.is_correct("0.00", problem.answer) or problem.answer == "0.00"


@pytest.mark.parametrize("split", ["train", "ood"])
def test_answers_round_trip_through_the_medication_verifier(split):
    """The stored answer, in any accepted spelling, passes; broken ones do not."""
    verifier = MedicationVerifier()
    for p in GEN.generate(100, split):
        unit = ANSWER_UNIT[p.params["family"]]
        kwargs = {"unit": unit, "unit_aliases": UNIT_ALIASES.get(unit, [])}

        def check(answer_text: str) -> bool:
            completion = f"<think>x</think><answer>{answer_text}</answer>"
            return verifier.verify(completion, p.answer, **kwargs).is_correct

        assert check(f"{p.answer} {unit}")
        assert check(f"{p.answer.replace('.', ',')} {unit}")  # Spanish decimal comma
        assert not check(f"{float(p.answer) * 10} {unit}")    # wrong order of magnitude
        assert not check(f"{p.answer}")                       # right number, no unit