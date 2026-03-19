"""
shared test fixtures for the tests/ directory.

ensures src/ is on the python path for all secondary tests.
all comments are lowercase.
"""

import os
import sys

# ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
