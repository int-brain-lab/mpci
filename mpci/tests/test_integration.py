import unittest
from pathlib import Path

from mpci.tests import IntegrationTestCase


class TestIntegration(IntegrationTestCase):
    """Integration tests that require S3 data."""

    def test_integration(self):
        """Example integration test that uses S3 data."""
        # Access the integration data directory
        self.assertTrue(self.DATA_DIR is not None, "Integration data directory is not set.")
        data_path = Path(self.DATA_DIR)
        self.assertTrue(data_path.exists(), "Integration data directory does not exist.")


if __name__ == "__main__":
    unittest.main()
