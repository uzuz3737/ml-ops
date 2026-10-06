import json

import pytest

pd = pytest.importorskip("pandas")

from mlops_project.data.policy import FEATURES, TARGET  # noqa: E402
from mlops_project.data.tables import cell, read_table  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "value"),
    [("12", 12), (" -5 ", -5), ("1.5", 1.5), ("", None), ("  ", None), ("n/a", "n/a"),
     ("inf", "inf"), (None, None), (float("nan"), None), (7, 7)],
)  # fmt: skip
def test_cell_keeps_what_the_file_says(raw, value):
    assert cell(raw) == value


def test_kaggle_and_generic_headers_map_to_canonical_names(tmp_path):
    kaggle = tmp_path / "kaggle.csv"
    kaggle.write_text("ID,PAY_1,default.payment.next.month\n1,2,0\n", encoding="utf-8")
    assert read_table(kaggle) == [{"ID": 1, "PAY_0": 2, TARGET: 0}]

    generic = tmp_path / "generic.json"
    generic.write_text(json.dumps({"instances": [{"X1": 20000, "X6": -1, "Y": 1}]}))
    assert read_table(generic) == [{FEATURES[0]: 20000, "PAY_0": -1, TARGET: 1}]


def test_uci_spreadsheet_second_header_row_is_used(tmp_path):
    pytest.importorskip("openpyxl")  # in the worker lock, not the CI dev lock
    path = tmp_path / "uci.xlsx"
    rows = [["", "X1", "Y"], ["ID", "LIMIT_BAL", "default payment next month"], [1, 20000, 1]]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    assert read_table(path) == [{"ID": 1, "LIMIT_BAL": 20000, TARGET: 1}]


def test_unparseable_files_raise(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"rows": 3}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_table(bad)
