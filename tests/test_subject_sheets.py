"""Spreadsheet readers for SC-subjects.xls and ST-subjects.xls.

The layouts mirror the real files as inspected in T1 (2026-11-22). Synthetic
.xlsx copies are used because the real sheets are not in the repository.
"""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from mist.data import sleepedf as E  # noqa: E402


def test_sc_sheet_sex_coding(tmp_path):
    df = pd.DataFrame({"subject": [0, 0, 1], "night": [1, 2, 1], "age": [33, 33, 58],
                       "sex (F=1)": [1, 1, 2], "LightsOff": ["00:38:00", "21:57:00", "22:44:00"]})
    df.to_excel(tmp_path / "sc.xlsx", index=False)
    d = E.read_sc_subjects(tmp_path / "sc.xlsx")
    assert d[(0, 2)] == {"age": 33, "sex": "F", "lights_off": "21:57:00"}
    assert d[(1, 1)]["sex"] == "M"


def test_st_sheet_two_headers_and_conditions(tmp_path):
    rows = [["Subject - age - sex", np.nan, np.nan, "Placebo night", np.nan, "Temazepam night", np.nan],
            ["Nr", "Age", "M1/F2", "night nr", "lights off", "night nr", "lights off"],
            [1, 60, 1, 1, "23:01:00", 2, "23:48:00"],
            [2, 35, 2, 2, "23:27:00", 1, "00:00:00"]]
    pd.DataFrame(rows).to_excel(tmp_path / "st.xlsx", index=False, header=False)
    d = E.read_st_subjects(tmp_path / "st.xlsx")
    assert d[(1, 1)] == {"age": 60, "sex": "M", "lights_off": "23:01:00", "condition": "placebo"}
    assert d[(1, 2)]["condition"] == "temazepam"
    assert d[(2, 1)] == {"age": 35, "sex": "F", "lights_off": "00:00:00", "condition": "temazepam"}
    assert len(d) == 4


def test_st_sheet_rejects_unknown_layout(tmp_path):
    pd.DataFrame([["a", "b"], [1, 2]]).to_excel(tmp_path / "bad.xlsx", index=False, header=False)
    with pytest.raises(ValueError):
        E.read_st_subjects(tmp_path / "bad.xlsx")
