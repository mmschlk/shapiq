# ✌️ Contributing Guidelines

This document outlines the guidelines for contributing to the project. It should enable contributors
to understand the process for applying changes to the project and how to interact with the community.
For the code of conduct, please refer to the [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## 🗺️ What to Work On
First, we welcome contributions from everyone in every form. If you feel that something is missing
or could be improved, feel free to change it. However, to streamline the process of contributing
higher-tier changes or features to the project, we maintain an open
[roadmap](https://github.com/users/mmschlk/projects/4/views/4). There, we collect ideas and features
that we want to add to the project. If you want to work on something, please check the roadmap first
to see if the feature is already planned or if there is a similar feature that you could contribute
to.

### 🙏 Discussions
If you have an idea for a new feature or a change, we encourage everyone to open a discussion in the
[Discussions](https://github.com/mmschlk/shapiq/discussions/new/choose) section.
We encourage you to open a discussion so that we can align on the work to be done. It's generally a
good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.


## 📝 Typical Setup: Fork, Clone, and Pull

The typical workflow for contributing to `shapiq` is:

1. Fork the `main` branch from the [GitHub repository](https://github.com/mmschlk/shapiq/).
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `main` branch.

## 📦 Development Setup

Start by cloning the repository:

```sh
git clone https://github.com/mmschlk/shapiq/
```

Next you need a python environment with a supported version of python. We recommend using
[pyenv](https://github.com/pyenv/pyenv-installer). Once you have pyenv, you can install the latest
Python version `shapiq` supports:

```sh
pyenv install 3.9
```

Then, create a virtual environment and install the development dependencies:

```sh
cd shapiq
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Finally, install the [pre-commit](https://pre-commit.com/) push hooks. This will run some code
quality checks every time you push to GitHub.

```sh
pre-commit install --hook-type pre-push
```

If you want, you can optionally run `pre-commit` at any time as so:

```sh
pre-commit run --all-files
```

## 📝 Commit Messages

We do not enforce a strict commit message format, but we encourage you to follow good practices.
We recommend to use action-words to automatically close issues or pull requests (example: `closes #123`).
For example, start the commit message with a verb in the imperative mood, and keep the message short
and concise. For example:

```
add feature-xyz and closes #123
```

## 🛠️ Making Changes

Now, you're ready to make changes to the code. We recommend that you check out `shapiq`'s source
code for inspiration before getting started. How you make changes is, of course, up to you. However,
we can give you some tips on how to document and test your changes.

### 📖 Documenting Changes
If you are adding a new class of function, you will need to add a docstring to the class or
function. With `shapiq`, we use the [Google Style Convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
Please add a docstring in the same style.

To build the documentation on your end and to check if your changes are documented correctly, you
need to install the documentation dependencies:

```sh
pip install -e .[docs]
```

Then, you can build the documentation from the root of the repository with:

```sh
sphinx-build docs/source docs/build
```

This will render the documentation in the `docs/build` directory. You can open the `index.html` file
in your browser to see the rendered documentation.

### 🎯 Testing Changes

We use `pytest` for running unit tests and coverage. In the near future we will add `mypy` to the
static type checking.

#### Unit Tests

Unit tests **absolutely need to pass**. Write unit tests for your changes. If you are adding a new
feature, you need to add tests for the new feature. If you are fixing a bug it is a good idea to add
a test that shows the bug and that your fix works.
Unit tests are located in the `tests` directory. To run the tests, you can use the following command:

```sh
pytest
```

#### Coverage

With `shapiq`, we aim to have a high test coverage (95% -100%). We aim that every pull request does
not decrease the test coverage.
We use `pytest-cov` to measure the test coverage. To run the tests with coverage, you can use the
following command:

```sh
pytest --cov=shapiq
```

#### Static Type Checking and Code Quality

Currently, we do not have static type checking in place. We use `pre-commit` to run some code quality
checks. These checks **absolutely need to pass**. You can run the checks with the following command:

```sh
pre-commit run --all-files
```

In the near future we aim to use `mypy` for type checking. Once added this will also be part of the
pre-commit pipeline and hence **absolutely need to pass**.

If you want, you can run `mypy` with the following command:

```sh
mypy shapiq
```
