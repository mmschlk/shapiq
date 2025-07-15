"""A test module which contains tests for deprecations.

This module is used to ensure that deprecated features are properly flagged and that after
they should be removed, that they indeed are no longer present in the codebase.

Usage:
    - Adding Deprecations: If you add a new deprecation, you should add a test here to ensure that
        it is raised properly. Ensure that in each deprecation message, you include the version
        of future removal, so that users can easily identify when the feature will be removed.
        The test should check that the deprecation warning includes a version.
    - Removing Deprecations: If you remove a deprecated feature, ensure that the test for that
        feature is also removed. This helps keep the test suite clean and focused on current
        features.
"""
