# Shared set-up for the tests. You don't need to change this file.
# Each test that asks for members gets its own fresh copy of the 600 members in cafe_members.csv, read from this
# exercise's folder, so one test can't change the table another test reads.
from pathlib import Path

import pandas as pd
import pytest

CAFE_MEMBERS = Path(__file__).parent.parent / "cafe_members.csv"


@pytest.fixture
def members():
    """The 600 café members in cafe_members.csv, freshly read for one test."""
    return pd.read_csv(CAFE_MEMBERS)
