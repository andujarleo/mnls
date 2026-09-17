"""Offline preservation checks; run with the Python standard library."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "verify_reference.py"


class ReferenceChecks(unittest.TestCase):
    def test_repository_reference(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("4.27", result.stdout)
        self.assertIn("4.87", result.stdout)

    def test_detects_changed_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "docs" / "archive"
            archive.mkdir(parents=True)
            expected = hashlib.sha256(b"original").hexdigest()
            (archive / "reference-manifest.json").write_text(json.dumps({
                "format": 1, "files": {"changed.txt": expected, "missing.txt": expected}
            }))
            (root / "changed.txt").write_text("modified")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CHANGED: changed.txt", result.stdout)
            self.assertIn("MISSING: missing.txt", result.stdout)


if __name__ == "__main__":
    unittest.main()
