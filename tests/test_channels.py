"""Channel names and the fixed channel rule (D-017)."""
from mist.data.channels import normalize, pick_rule_channel, plan_channels


def test_normalize_spellings():
    assert normalize("EEG F4-M1") == "F4-M1"
    assert normalize("F3_M2") == "F3-M2"
    assert normalize("FP1-F3") == "Fp1-F3"
    assert normalize("F3A2") == "F3-A2"
    assert normalize("C4") == "C4"
    for not_eeg in ("ROC-LOC", "LOC / A2", "E1-M2", "EMG1-EMG2", "SAO2", "X1", "24", "A1-C4"):
        assert normalize(not_eeg) is None, not_eeg


def test_rule_order_and_aliases():
    assert pick_rule_channel(["C4-A1", "F3-C3", "F4-C4", "F4-A1"]) == ("F4-C4", "frontal-central")
    assert pick_rule_channel(["Fz-Cz", "F4-C4"]) == ("Fz-Cz", "frontal-central")
    assert pick_rule_channel(["C3-A2", "F3-A2", "F4-A1"]) == ("F4-A1", "frontal-mastoid")
    assert pick_rule_channel(["F3-M2", "F4-M2"]) == ("F3-M2", "frontal-mastoid")      # contralateral first
    assert pick_rule_channel(["C3-A2", "C4-A1", "O2-A1"]) == ("C4-A1", "central-mastoid")
    assert pick_rule_channel(["C3-A1", "O2-A1", "Fp2-F4"]) == (None, None)            # ipsilateral central not in the rule


def test_plan_derives_from_monopolar_leads():
    plan = dict(plan_channels(["ROC", "A1", "C4", "F4", "A2", "F3", "C3-A2"], ("F4-A1", "F3-A2", "C3-A2", "O1-A2")))
    assert plan["C3-A2"] == ("raw", "C3-A2")                 # recorded pair kept as recorded
    assert plan["F4-A1"] == ("diff", "F4", "A1") and plan["F3-A2"] == ("diff", "F3", "A2")
    assert "O1-A2" not in plan
