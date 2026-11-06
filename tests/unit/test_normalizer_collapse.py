import unicodedata
import unittest

from vnsentiment.preprocessing.collapse import (
    collapse_elongations,
    normalize_basic,
    normalize_text,
)


class NormalizeBasicTests(unittest.TestCase):
    def test_nfc_lowercase_and_whitespace(self):
        self.assertEqual(normalize_basic("  SP   NÀY\tCÓ   PIN  "), "sp này có pin")

    def test_nfd_input_is_composed(self):
        raw = unicodedata.normalize("NFD", "TỆỆỆ")
        self.assertEqual(normalize_basic(raw), "tệệệ")

    def test_whitespace_only_becomes_empty(self):
        self.assertEqual(normalize_basic("   \n\t "), "")


class CollapseElongationTests(unittest.TestCase):
    def test_collapses_four_repeated_letters(self):
        self.assertEqual(collapse_elongations("đẹpppp"), "đẹp")

    def test_collapses_three_repeated_letters(self):
        self.assertEqual(collapse_elongations("tệệệ"), "tệ")
        self.assertEqual(collapse_elongations("coool"), "col")

    def test_preserves_two_repeated_letters(self):
        for word in ("cool", "good", "see", "room", "sooc"):
            with self.subTest(word=word):
                self.assertEqual(collapse_elongations(word), word)

    def test_preserves_digit_runs(self):
        self.assertEqual(collapse_elongations("1000đ"), "1000đ")
        self.assertEqual(collapse_elongations("111"), "111")

    def test_preserves_punctuation_runs(self):
        self.assertEqual(collapse_elongations("!!! ??? ..."), "!!! ??? ...")

    def test_preserves_underscore(self):
        self.assertEqual(collapse_elongations("sản_phẩm"), "sản_phẩm")

    def test_collapses_letter_run_next_to_digits(self):
        self.assertEqual(collapse_elongations("aaa111"), "a111")

    def test_collapses_uppercase_run(self):
        self.assertEqual(collapse_elongations("ĐẸPPP".lower()), "đẹp")


class NormalizeTextTests(unittest.TestCase):
    def test_report_example(self):
        self.assertEqual(
            normalize_text("  sp này đẹpppp nhưng pin ko trâu "),
            "sp này đẹp nhưng pin ko trâu",
        )

    def test_nfd_tệệệ_is_composed_then_collapsed(self):
        raw = unicodedata.normalize("NFD", "tệệệ")
        self.assertEqual(normalize_text(raw), "tệ")

    def test_combined_punctuation_and_digits_kept(self):
        self.assertEqual(normalize_text("SP đẹpppp!!! 1000đ"), "sp đẹp!!! 1000đ")

    def test_empty_string(self):
        self.assertEqual(normalize_text(""), "")


if __name__ == "__main__":
    unittest.main()
