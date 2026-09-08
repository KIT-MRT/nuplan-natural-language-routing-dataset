import tempfile
import unittest
from pathlib import Path

from route_description_generation.cli.export_tokens import (
    export_tokens,
    token_from_npz_filename,
)


class TestExportTokens(unittest.TestCase):
    def test_extract_token_from_filename(self):
        self.assertEqual(
            token_from_npz_filename("sg-one-north_168c4e592f1b5984.npz"),
            "168c4e592f1b5984",
        )
        self.assertIsNone(token_from_npz_filename("invalid.txt"))
        self.assertIsNone(token_from_npz_filename("missingdelimiter.npz"))

    def test_export_tokens_writes_expected_lines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "sg-one-north_aaa111.npz").touch()
            (root / "us-ma-boston_bbb222.npz").touch()
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            output = root / "out" / "train_tokens.txt"
            exported = export_tokens(
                input_dir=root,
                output_file=output,
                flush_every=1,
                show_progress=False,
            )

            self.assertEqual(exported, 2)
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(set(lines), {"aaa111", "bbb222"})


if __name__ == "__main__":
    unittest.main()
