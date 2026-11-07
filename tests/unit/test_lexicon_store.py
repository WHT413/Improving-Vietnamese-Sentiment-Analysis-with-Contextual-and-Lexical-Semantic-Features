import copy
import json
import unittest
from pathlib import Path

from vnsentiment.lexicon.store import LexiconStore

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"


class LexiconStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.store = LexiconStore(cls.payload)

    def test_loads_every_fixture_entry(self):
        self.assertEqual(len(self.store), 5)
        self.assertEqual(
            self.store.features,
            (
                "polarity",
                "valence",
                "arousal",
                "dominance",
                "joy",
                "sadness",
                "anger",
                "fear",
                "disgust",
                "surprise",
                "trust",
                "anticipation",
            ),
        )

    def test_rescales_polarity_to_unit_interval(self):
        values, mask = self.store.lookup("tốt")
        self.assertAlmostEqual(values[0], 0.9375)
        self.assertEqual(mask[0], 1)
        values, mask = self.store.lookup("tệ")
        self.assertAlmostEqual(values[0], 0.1875)
        self.assertEqual(mask[0], 1)

    def test_keeps_vad_and_emotion_scales_native(self):
        values, mask = self.store.lookup("sản_phẩm")
        self.assertAlmostEqual(values[1], 0.646)
        self.assertAlmostEqual(values[2], 0.417)
        self.assertAlmostEqual(values[3], 0.648)
        self.assertEqual(mask[1:4], (1, 1, 1))
        self.assertEqual(mask[0], 0)
        self.assertEqual(values[0], 0.0)

        values, mask = self.store.lookup("tốt")
        self.assertEqual(values[4], 1.0)
        self.assertEqual(mask[4], 1)
        self.assertEqual(values[5], 0.0)
        self.assertEqual(mask[5], 0)

    def test_true_zero_is_distinct_from_missing(self):
        values, mask = self.store.lookup("này")
        self.assertEqual(values[4:], (0.0,) * 8)
        self.assertEqual(mask[4:], (1,) * 8)
        self.assertEqual(mask[0], 0)
        self.assertEqual(values[0], 0.0)

    def test_oov_is_none(self):
        self.assertIsNone(self.store.lookup("xyz_oov_test"))

    def test_every_present_dimension_is_in_unit_interval(self):
        for key in self.payload["entries"]:
            values, mask = self.store.lookup(key)
            for value, present in zip(values, mask):
                if present:
                    self.assertGreaterEqual(value, 0.0, key)
                    self.assertLessEqual(value, 1.0, key)

    def test_masks_and_values_are_immutable_tuples(self):
        values, mask = self.store.lookup("tốt")
        self.assertIsInstance(values, tuple)
        self.assertIsInstance(mask, tuple)

    def test_rejects_wrong_feature_order(self):
        payload = copy.deepcopy(self.payload)
        payload["features"] = list(reversed(payload["features"]))
        with self.assertRaises(ValueError):
            LexiconStore(payload)

    def test_rejects_wrong_dimension(self):
        for field in ("values", "mask"):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                payload["entries"]["tốt"][field] = payload["entries"]["tốt"][field][:11]
                with self.assertRaises(ValueError):
                    LexiconStore(payload)

    def test_rejects_nonzero_value_behind_missing_mask(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"]["tốt"]["values"][5] = 0.5
        with self.assertRaises(ValueError):
            LexiconStore(payload)

    def test_rejects_polarity_outside_native_range(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"]["tốt"]["values"][0] = 1.5
        with self.assertRaises(ValueError):
            LexiconStore(payload)

    def test_rejects_vad_outside_unit_interval(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"]["tốt"]["values"][1] = 1.5
        with self.assertRaises(ValueError):
            LexiconStore(payload)

    def test_rejects_non_binary_emotion(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"]["tốt"]["values"][4] = 0.5
        with self.assertRaises(ValueError):
            LexiconStore(payload)

    def test_rejects_invalid_mask(self):
        payload = copy.deepcopy(self.payload)
        payload["entries"]["tốt"]["mask"][0] = 2
        with self.assertRaises(ValueError):
            LexiconStore(payload)


if __name__ == "__main__":
    unittest.main()