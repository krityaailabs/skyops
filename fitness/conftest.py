"""Turn an unfinished lab into a clear failure message instead of a traceback."""
import pytest

from skyops.model.errors import LabNotDone


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    exc = outcome.excinfo
    if exc is not None and issubclass(exc[0], LabNotDone):
        outcome.force_exception(pytest.fail.Exception(f"LAB NOT DONE: {exc[1]}", pytrace=False))
