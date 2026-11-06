import os
import unittest

from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter


@unittest.skipUnless(
    os.environ.get("RUN_VNCORENLP_TEST") == "1",
    "set RUN_VNCORENLP_TEST=1 to run the real Java/PyJNIus smoke test",
)
class VnCoreNLPSmokeTests(unittest.TestCase):
    def test_real_segmentation(self):
        segmenter = VnCoreNLPSegmenter()
        self.assertEqual(
            segmenter.segment("sản phẩm này không tệ"),
            "sản_phẩm này không tệ",
        )
        self.assertEqual(
            segmenter.segment("slide giáo trình đầy đủ ."),
            "slide giáo_trình đầy_đủ .",
        )


if __name__ == "__main__":
    unittest.main()
