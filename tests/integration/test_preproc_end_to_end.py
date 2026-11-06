import unittest
from pathlib import Path

from vnsentiment.preprocessing.pipeline import PreprocessingPipeline, RawSample
from vnsentiment.preprocessing.variants import VariantMap

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class CompoundJoinerSegmenter:
    def __init__(self, compounds):
        self._compounds = compounds

    def segment(self, text: str) -> str:
        for compound in self._compounds:
            text = text.replace(compound, compound.replace(" ", "_"))
        return text


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_report_example_end_to_end(self):
        variant_map = VariantMap.load(FIXTURES / "variant_map_fixture.json")
        pipeline = PreprocessingPipeline(
            variant_map,
            segmenter=CompoundJoinerSegmenter(["sản phẩm", "chất lượng"]),
            require_approved=False,
        )
        result = pipeline.process(
            RawSample("report-1", "sp này đẹpppp nhưng pin ko trâu", label=2)
        )
        self.assertEqual(
            result.normalized_text, "sản_phẩm này đẹp nhưng pin không trâu"
        )

    def test_multiword_variant_is_replaced_before_segmentation(self):
        variant_map = VariantMap.load(FIXTURES / "variant_map_fixture.json")
        pipeline = PreprocessingPipeline(
            variant_map,
            segmenter=CompoundJoinerSegmenter(["sản phẩm", "chất lượng"]),
            require_approved=False,
        )
        result = pipeline.process(RawSample("report-2", "chất lg ổn"))
        self.assertEqual(result.normalized_text, "chất_lượng ổn")


if __name__ == "__main__":
    unittest.main()
