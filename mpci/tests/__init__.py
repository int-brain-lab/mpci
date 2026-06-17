import os
import unittest

INTEGRATION_DATA_DIR = os.environ.get("INTEGRATION_DATA_DIR")


@unittest.skipUnless(
    INTEGRATION_DATA_DIR and os.path.isdir(INTEGRATION_DATA_DIR),
    "Integration data not available (set INTEGRATION_DATA_DIR to enable).",
)
class IntegrationTestCase(unittest.TestCase):
    """Base class for tests that require S3 integration data.

    Subclass this for any test needing integration data. When
    INTEGRATION_DATA_DIR is unset or missing (e.g. an outside contributor
    running plain `python -m unittest discover`), these tests auto-skip,
    so the unit suite still runs cleanly.
    """

    DATA_DIR = INTEGRATION_DATA_DIR
