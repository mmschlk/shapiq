# Creating a Release

> Requires push access to `mmschlk/shapiq`.

## Version format

Tags must match `v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v1.2.3`). Tags containing `-` are skipped by the publish workflow.

The version is derived automatically from the tag via `setuptools_scm`, which means that no manual edits to `pyproject.toml` are needed.

## Steps

1. **Create and push an annotated tag** on the commit you want to release (any branch):
   ```sh
   git tag -a v1.2.3 -m "Release v1.2.3"
   git push origin v1.2.3
   ```

2. **Publish the GitHub Release**: Releases → Draft a new release → select the tag → add title + changelog → **Publish release**.

## What happens automatically

The `python-publish.yml` workflow triggers, builds wheels for Linux / Windows / macOS (Intel + ARM) and an sdist via `build.yml`, then publishes to PyPI using trusted publishing (OIDC).

## Verification

Check the [Actions tab](https://github.com/mmschlk/shapiq/actions) and the [shapiq PyPI page](https://pypi.org/project/shapiq/).

## If publish fails

Re-run the failed workflow from the Actions UI — no need to delete or recreate the release.
