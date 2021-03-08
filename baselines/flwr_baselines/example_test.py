"""Placeholder to ensure test setup is working correctly.

Should be removed with first real code
"""

import unittest


class TestStringMethods(unittest.TestCase):
    """Test string methods."""

    def test_upper(self):
        """Testing upper method."""
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
