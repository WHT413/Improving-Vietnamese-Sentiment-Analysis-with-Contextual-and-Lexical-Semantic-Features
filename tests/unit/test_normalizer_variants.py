import json
import tempfile
import unittest
from pathlib import Path

from vnsentiment.preprocessing.variants import VariantMap, VariantMatcher

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class VariantMapLoadingTests(unittest.TestCase):
    def test_loads_metadata_and_entries(self):
        variant_map = VariantMap.load(FIXTURES / "variant_map_fixture.json")
        self.assertEqual(variant_map.version, "fixture-1.0.0")
        self.assertIn("synthetic fixture", variant_map.source)
        self.assertEqual(len(variant_map.entries), 9)

    def test_keys_and_targets_are_normalized(self):
        variant_map = VariantMap.from_entries({"  SP  ": "  Sản   Phẩm "})
        self.assertEqual(dict(variant_map.entries), {"sp": "sản phẩm"})

    def test_loads_entries_as_list(self):
        payload = json.dumps(
            {"version": "list-form", "entries": [{"key": "SP", "target": "Sản phẩm"}]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list_form.json"
            path.write_text(payload, encoding="utf-8")
            variant_map = VariantMap.load(path)
        self.assertEqual(dict(variant_map.entries), {"sp": "sản phẩm"})

    def test_duplicate_keys_after_normalization_are_rejected(self):
        with self.assertRaises(ValueError):
            VariantMap.load(FIXTURES / "variant_map_duplicates.json")

    def test_raw_duplicate_json_keys_are_rejected(self):
        payload = '{"entries": {"sp": "a", "sp": "b"}}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                VariantMap.load(path)

    def test_empty_key_is_rejected(self):
        with self.assertRaises(ValueError):
            VariantMap.from_entries({"   ": "x"})

    def test_empty_target_is_rejected(self):
        with self.assertRaises(ValueError):
            VariantMap.from_entries({"sp": "   "})

    def test_entry_count_mismatch_is_rejected(self):
        payload = json.dumps({"entry_count": 3, "entries": {"sp": "sản phẩm"}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch.json"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                VariantMap.load(path)


class VariantMatcherTests(unittest.TestCase):
    def setUp(self):
        self.variant_map = VariantMap.load(FIXTURES / "variant_map_fixture.json")
        self.matcher = VariantMatcher(self.variant_map)

    def test_replaces_single_token_with_boundaries(self):
        self.assertEqual(self.matcher.replace("sp này"), "sản phẩm này")

    def test_does_not_replace_inside_longer_word(self):
        self.assertEqual(self.matcher.replace("spa"), "spa")
        self.assertEqual(self.matcher.replace("kool"), "kool")

    def test_prefers_longest_phrase(self):
        self.assertEqual(self.matcher.replace("chất lg"), "chất lượng")

    def test_replaces_every_occurrence(self):
        self.assertEqual(self.matcher.replace("ko, ko!"), "không, không!")

    def test_unknown_variant_is_kept(self):
        self.assertEqual(self.matcher.replace("ok"), "ok")

    def test_single_pass_does_not_rescan_replacements(self):
        self.assertEqual(self.matcher.replace("a b"), "b c")

    def test_punctuation_boundaries_are_replaced(self):
        self.assertEqual(self.matcher.replace("(ko)"), "(không)")
        self.assertEqual(self.matcher.replace("sp,"), "sản phẩm,")

    def test_underscore_blocks_boundary_match(self):
        self.assertEqual(self.matcher.replace("sp_phẩm"), "sp_phẩm")

    def test_single_letter_key_requires_boundary(self):
        self.assertEqual(self.matcher.replace("k"), "không")
        self.assertEqual(self.matcher.replace("không"), "không")
        self.assertEqual(self.matcher.replace("k."), "không.")

    def test_regex_special_characters_are_escaped(self):
        matcher = VariantMatcher(VariantMap.from_entries({"c++": "cpp"}))
        self.assertEqual(matcher.replace("c++"), "cpp")
        self.assertEqual(matcher.replace("c"), "c")

    def test_empty_map_returns_text_unchanged(self):
        matcher = VariantMatcher(VariantMap.from_entries({}))
        self.assertEqual(matcher.replace("sp này"), "sp này")


if __name__ == "__main__":
    unittest.main()
