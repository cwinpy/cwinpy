"""
Add _called_from_test if running code within as test, see
https://docs.pytest.org/en/latest/example/simple.html#detect-if-running-from-within-a-pytest-run
"""


def pytest_configure(config):
    import cwinpy

    cwinpy._called_from_test = True


def pytest_unconfigure(config):
    import cwinpy

    del cwinpy._called_from_test
