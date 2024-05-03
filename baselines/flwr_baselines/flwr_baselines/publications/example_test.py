"""Placeholder to ensure test setup is working correctly.

Should be removed when first real code has tests.
"""

import unittest


class TestStringMethods(unittest.TestCase):
    """Test string methods."""

    def test_upper(self) -> None:
        """Testing upper method."""
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()
