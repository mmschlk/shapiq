# ✌️ Contributing Guidelines

This document outlines how to easily contribute to the project.
For the code of conduct, please refer to the [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## 🗺️ What to Work On
First, we welcome contributions **from everyone** in **every form**. If you feel that something is
missing or could be improved, feel free to change it. However, to streamline the process of
contributing higher-tier changes or features to the project, we maintain an open
[roadmap](https://github.com/users/mmschlk/projects/4/views/4) which is just storing all the ideas
and problems found on shapiq's [issues page](https://github.com/mmschlk/shapiq/issues). If you want
to work on something, check out the roadmap or issues first to see if the feature is already
planned or if there is a similar feature that you could contribute to.

Here are some examples of what we are very happy to receive contributions for:

### Approximators
We are always looking for new approximators to add to `shapiq`. Approximators are always extending
the base class `Approximator` and implementing `approximate` method. Make sure to create unit tests
for the new approximator.

### Explainers
If you want to add a new explainer, you can extend the base class `Explainer` and implement the
`explain_function` method. Make sure to create unit tests for the new explainer. Note that
explainers are quite elaborate, so it is a very good idea to open a discussion before starting to
work on a new explainer.

### Model Support
You like a particular machine learning model and it is not yet supported by `shapiq`? Maybe you can
add support in the [transformation code](https://github.com/mmschlk/shapiq/blob/56e1ea4a41d185b8364ca8e6370a01646dd792c6/shapiq/explainer/utils.py#L1) or [tree/validation](https://github.com/mmschlk/shapiq/blob/56e1ea4a41d185b8364ca8e6370a01646dd792c6/shapiq/explainer/tree/validation.py).
Make sure to add tests for the new model as part of the unit tests (you can find the tests of the
other model types).

### Visualizations
If you have a nice idea to visualize Shapley values or Shapley interaction values, you can add a new
visualization to the `shapiq.plot` package. Make sure that plots are also available through the
`InteractionValues` object like the other plots (e.g. `InteractionValues.plot_force`). Make sure to
add tests for the new visualization.

### 🙏 Discussions and Issues
If you have an idea for a new feature or a change, we encourage everyone to open a discussion in the
[Discussions](https://github.com/mmschlk/shapiq/discussions/new/choose) section or open an [issues](https://github.com/mmschlk/shapiq/issues).
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
[uv](https://docs.astral.sh/uv/getting-started/installation), which works extremely fast and helps
to stay up-to-date with the latest Python versions. With `uv`, you can set up your development
environment with the following command:

```sh
uv sync --all-extras --dev
```

Now you are all set up and ready to go.


> 📝 **Note**: `shapiq` uses the `requests` library, you might need to install the
> certificates on your MacOS system ([more information](https://stackoverflow.com/a/53310545)).
```sh
/Applications/Python\ 3.x/Install\ Certificates.command
```

### 🛠️ Pre-Commit Hooks
To ensure that the code quality is maintained, we use `pre-commit` hooks. You need to install the
[pre-commit](https://pre-commit.com/)  hooks. This will run some code quality checks every time
you push to GitHub. You can view the checks in the `.pre-commit-config.yaml` file and the setup in
the `pyproject.toml` file.

```sh
uv run pre-commit install
```

If you want, you can optionally run `pre-commit` at any time as so:

```sh
uv run pre-commit run --all-files
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
uv sync --all-extras --dev --docs
```

Then, you can build the documentation from the root of the repository with:

```sh
sphinx-build docs/source docs/build
```

This will render the documentation in the `docs/build` directory. You can open the `index.html` file
in your browser to see the rendered documentation.

### 🎯 Testing Changes

We use `pytest` for running unit tests and coverage.

#### Unit Tests

Unit tests **absolutely need to pass**. Write unit tests for your changes. If you are adding a new
feature, you need to add tests for the new feature. If you are fixing a bug it is a good idea to add
a test that shows the bug and that your fix works.
Unit tests are located in the `tests` directory. To run the tests with `pytest`, you can use the
following command and replace `n_jobs` with the number of jobs you want to run in parallel:

```sh
uv run pytest -n n_jobs
```

#### Coverage

With `shapiq`, we aim to have a high test coverage (95% -100%). We aim that every pull request does
not decrease the test coverage.
We use `pytest-cov` to measure the test coverage. To run the tests with coverage, you can use the
following command:

```sh
uv run pytest --cov=shapiq
```

#### Static Type Checking and Code Quality

Currently, we do not have static type checking in place. We use `pre-commit` to run some code quality
checks. These checks **absolutely need to pass**. You can run the checks with the following command:

```sh
uv run pre-commit run --all-files
```
