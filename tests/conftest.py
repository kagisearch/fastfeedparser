import pytest

from fastfeedparser import main


@pytest.fixture
def lxml_path_only(monkeypatch):
    """Parse on the lxml path even when the native core is installed.

    For tests of that path's own machinery (its parsers, CDATA lifting, the
    trees it keeps in flight), which the native core never exercises.
    """
    monkeypatch.setattr(main, "_core", None)
