import unittest
from pathlib import Path

from vnsentiment.preprocessing.pipeline import (
    EmptyTextError,
    IdentitySegmenter,
    PreprocessingPipeline,
    RawSample,
)
from vnsentiment.preprocessing.variants import VariantMap

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class RecordingSegmenter:
    def __init__(self, compounds=("sản phẩm",)):
        self.inputs: list[str] = []
        self._compounds = compounds

    def segment(self, text: str) -> str:
        self.inputs.append(text)
        for compound in self._compounds:
            text = text.replace(compound, compound.replace(" ", "_"))
        return text


class StubSegmenter:
    def __init__(self, output):
        self._output = output

    def segment(self, text: str):
        return self._output


def build_pipeline(segmenter=None, **kwargs):
    variant_map = VariantMap.load(FIXTURES / "variant_map_fixture.json")
    kwargs.setdefault("require_approved", False)
    return PreprocessingPipeline(variant_map, segmenter=segmenter, **kwargs)


class PreprocessingPipelineTests(unittest.TestCase):
    def test_runs_collapse_then_variant_replacement(self):
        pipeline = build_pipeline()
        result = pipeline.process(
            RawSample("s1", "  sp này đẹpppp nhưng pin ko trâu  ", label=2)
        )
        self.assertEqual(
            result.normalized_text, "sản phẩm này đẹp nhưng pin không trâu"
        )

    def test_collapse_runs_before_variant_mapping(self):
        pipeline = build_pipeline()
        result = pipeline.process(RawSample("s1", "spppp"))
        self.assertEqual(result.normalized_text, "sản phẩm")

    def test_unapproved_variant_map_is_rejected_at_runtime(self):
        variant_map = VariantMap.load(FIXTURES / "variant_map_fixture.json")
        with self.assertRaises(ValueError):
            PreprocessingPipeline(variant_map)

    def test_missing_review_status_is_rejected_at_runtime(self):
        variant_map = VariantMap.from_entries({"sp": "sản phẩm"})
        with self.assertRaises(ValueError):
            PreprocessingPipeline(variant_map)

    def test_approved_variant_map_is_accepted(self):
        variant_map = VariantMap.from_entries(
            {"sp": "sản phẩm"}, version="1.0.0", review_status="approved"
        )
        pipeline = PreprocessingPipeline(variant_map)
        result = pipeline.process(RawSample("s1", "sp"))
        self.assertEqual(result.normalized_text, "sản phẩm")

    def test_non_string_segmenter_output_is_rejected(self):
        pipeline = build_pipeline(StubSegmenter(None))
        with self.assertRaises(TypeError):
            pipeline.process(RawSample("s1", "sp"))

    def test_empty_segmenter_output_is_rejected(self):
        pipeline = build_pipeline(StubSegmenter("   \n"))
        with self.assertRaises(EmptyTextError):
            pipeline.process(RawSample("s1", "sp"))

    def test_segmenter_output_is_normalized(self):
        pipeline = build_pipeline(StubSegmenter("  Sản_Phẩm  \n"))
        result = pipeline.process(RawSample("s1", "sp"))
        self.assertEqual(result.normalized_text, "sản_phẩm")

    def test_segmenter_receives_variant_normalized_text(self):
        segmenter = RecordingSegmenter()
        pipeline = build_pipeline(segmenter)
        pipeline.process(RawSample("s1", "sp này đẹpppp"))
        self.assertEqual(segmenter.inputs, ["sản phẩm này đẹp"])

    def test_identity_segmenter_passes_text_through(self):
        self.assertEqual(IdentitySegmenter().segment("sp này"), "sp này")

    def test_empty_text_after_normalization_is_rejected(self):
        pipeline = build_pipeline()
        with self.assertRaises(EmptyTextError):
            pipeline.process(RawSample("s1", "   \n\t "))

    def test_non_string_text_is_rejected(self):
        pipeline = build_pipeline()
        with self.assertRaises(TypeError):
            pipeline.process(RawSample("s1", None))  # type: ignore[arg-type]

    def test_bool_label_is_rejected(self):
        pipeline = build_pipeline()
        with self.assertRaises(ValueError):
            pipeline.process(RawSample("s1", "sp", label=True))

    def test_invalid_label_type_is_rejected(self):
        pipeline = build_pipeline()
        with self.assertRaises(ValueError):
            pipeline.process(RawSample("s1", "sp", label="2"))  # type: ignore[arg-type]

    def test_out_of_range_label_is_rejected(self):
        pipeline = build_pipeline()
        for label in (-1, 3):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    pipeline.process(RawSample("s1", "sp", label=label))

    def test_none_label_is_allowed(self):
        pipeline = build_pipeline()
        result = pipeline.process(RawSample("s1", "sp"))
        self.assertIsNone(result.label)

    def test_metadata_and_versions_are_recorded(self):
        pipeline = build_pipeline(preprocessing_version="test-1.0.0")
        result = pipeline.process(RawSample("sample-7", "sp", label=1))
        self.assertEqual(result.sample_id, "sample-7")
        self.assertEqual(result.label, 1)
        self.assertEqual(result.preprocessing_version, "test-1.0.0")
        self.assertEqual(result.variant_resource_version, "fixture-1.0.0")


if __name__ == "__main__":
    unittest.main()
