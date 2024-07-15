"""This test module contains all tests for the configuration of benchmark games."""

from shapiq.benchmark import print_benchmark_configurations


def test_print_config():
    print_benchmark_configurations()
    assert True
