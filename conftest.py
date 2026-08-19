"""Pytest configuration for the mpci test suite."""

import os
import unittest

from ibllib.tests.base import INTEGRATION_DATA_DIR, IntegrationTest

# ibllib decorates IntegrationTest with skipUnless(INTEGRATION_DATA_DIR), but its __init__
# raises FileNotFoundError when the data root holds no 'Subjects_init' folder. That raise
# happens while the test cases are instantiated during collection, i.e. before the skip is
# ever evaluated, and aborts discovery of the entire suite. Dropping the check lets the skip
# do its job: the integration tests report as skipped and the unit tests still run.
if not (INTEGRATION_DATA_DIR and os.path.isdir(INTEGRATION_DATA_DIR)):
    IntegrationTest.__init__ = unittest.TestCase.__init__
