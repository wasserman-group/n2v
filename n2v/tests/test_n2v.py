"""
Unit and regression test for the n2v package.
"""

# Import package, test suite, and other packages as needed
import n2v
import pytest
import sys

def test_n2v_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "n2v" in sys.modules
