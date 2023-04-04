from typing import cast
from unittest import TestCase

from openseries.types import (
    OpenTimeSeriesPropertiesList,
    OpenFramePropertiesList,
    LiteralSeriesProps,
    LiteralFrameProps,
)


class TestTypes(TestCase):
    def test_types_opentimeseriesproplist_validate(self: "TestTypes") -> None:
        subset = cast(LiteralSeriesProps, ["z_score", "kurtosis", "positive_share"])
        lst = OpenTimeSeriesPropertiesList(*subset)
        self.assertIsInstance(lst, OpenTimeSeriesPropertiesList)

        with self.assertRaises(ValueError) as e_boo:
            booset = cast(LiteralSeriesProps, ["z_score", "boo", "positive_share"])
            OpenTimeSeriesPropertiesList(*booset)
        self.assertIsInstance(e_boo.exception, ValueError)
        self.assertIn(member="Invalid string: boo", container=str(e_boo.exception))

        with self.assertRaises(ValueError) as e_booo:
            dupeset = cast(
                LiteralSeriesProps, ["z_score", "skew", "skew", "positive_share"]
            )
            OpenTimeSeriesPropertiesList(*dupeset)
        self.assertIsInstance(e_booo.exception, ValueError)
        self.assertIn(member="Duplicate string: skew", container=str(e_booo.exception))

    def test_types_openframeproplist_validate(self: "TestTypes") -> None:
        subset = cast(LiteralFrameProps, ["z_score", "kurtosis", "positive_share"])
        lst = OpenFramePropertiesList(*subset)
        self.assertIsInstance(lst, OpenFramePropertiesList)

        with self.assertRaises(ValueError) as e_boo:
            booset = cast(LiteralFrameProps, ["z_score", "boo", "positive_share"])
            OpenFramePropertiesList(*booset)
        self.assertIsInstance(e_boo.exception, ValueError)
        self.assertIn(member="Invalid string: boo", container=str(e_boo.exception))

        with self.assertRaises(ValueError) as e_booo:
            dupeset = cast(
                LiteralFrameProps, ["z_score", "skew", "skew", "positive_share"]
            )
            OpenFramePropertiesList(*dupeset)
        self.assertIsInstance(e_booo.exception, ValueError)
        self.assertIn(member="Duplicate string: skew", container=str(e_booo.exception))
