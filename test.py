"""Run the canonical synthetic regression suite."""
import unittest
if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests", pattern="test_*.py"))
    raise SystemExit(not result.wasSuccessful())
