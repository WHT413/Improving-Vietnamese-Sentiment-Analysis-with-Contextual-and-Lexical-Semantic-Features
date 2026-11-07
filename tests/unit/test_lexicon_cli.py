import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from vnsentiment.lexicon.cli import LexiconCli
from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.projection import LexiconProjection
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"
PIECES = {
    "tốt": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
}


class LexiconCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore.load(FIXTURE)

    def _cli(self, pieces=None, hidden_dim=16):
        tokenizer = FakeTokenizer(pieces or PIECES)
        branch = LexiconBranch(
            self.store, LexiconProjection(hidden_dim=hidden_dim)
        )
        return LexiconCli(self.store, branch, tokenizer), tokenizer

    def _report(self, cli, text):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli.print_report(text)
        return buffer.getvalue()

    def test_loads_fixture_store_and_tokenizer(self):
        cli = LexiconCli.load(FIXTURE, tokenizer=FakeTokenizer(PIECES))
        self.assertEqual(len(cli.store), 5)
        self.assertEqual(len(cli.store.features), 12)

    def test_prints_per_token_lookup_and_profile(self):
        cli, _ = self._cli()
        output = self._report(cli, "tốt này không tệ")

        self.assertIn("5 entries", output)
        self.assertIn("12 dims", output)
        self.assertIn("projection=16", output)
        self.assertIn("tốt | tốt | 0.9375", output)
        self.assertIn("tệ | tệ | 0.1875", output)
        self.assertIn("profile P/VAD", output)
        self.assertIn("polarity'=0.5625[n=2]", output)
        self.assertIn("L=1x8x12", output)
        self.assertIn("s=1x8x1", output)
        self.assertIn("L~=1x8x16", output)
        self.assertIn("không phải dự đoán nhãn", output)

    def test_prints_oov_rows_and_zero_mask(self):
        tokenizer = FakeTokenizer({"xyz_oov_test": [99]})
        cli, _ = self._cli({"xyz_oov_test": [99]})
        output = self._report(cli, "xyz_oov_test")

        self.assertIn("xyz_oov_test | xyz_oov_test | -", output)
        self.assertIn("OOV", output)
        self.assertIn("mask=0/12", output)
        self.assertIn("s=0", output)

    def test_profile_uses_true_zero_annotations(self):
        cli, _ = self._cli()
        output = self._report(cli, "này")
        self.assertIn("joy=0.0000[n=1]", output)


if __name__ == "__main__":
    unittest.main()