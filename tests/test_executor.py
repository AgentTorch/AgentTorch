import pytest
from contextlib import contextmanager

from fixtures.executor import executor


@contextmanager
def not_raises():
    try:
        yield
    except Exception as exception:
        raise pytest.fail(f"Raised {exception}.")


def test_executor(executor):
    with not_raises():
        executor.execute()
