# Contributing to AgentTune

Thank you for your interest in contributing to AgentTune! We welcome contributions from the community.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

This repo is the AgentTune repository
[`Lexsi-Labs/AgentTune`](https://github.com/Lexsi-Labs/AgentTune); contributions here target
this repo.

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/AgentTune.git
   cd AgentTune
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Lexsi-Labs/AgentTune.git
   ```

## How to Contribute

### Reporting Bugs

- Use the [bug report template](https://github.com/Lexsi-Labs/AgentTune/blob/main/.github/ISSUE_TEMPLATE/bug_report.md) on the GitHub Issues page
- Search existing issues first to avoid duplicates
- Provide detailed information:
  - Operating system and version
  - Python version
  - AgentTune version
  - Complete error messages and stack traces
  - Minimal reproducible example

### Suggesting Features

- Use the [feature request template](https://github.com/Lexsi-Labs/AgentTune/blob/main/.github/ISSUE_TEMPLATE/feature_request.md) on the GitHub Issues page
- Clearly describe the feature and its benefits
- Provide examples of how it would be used

### Improving Documentation

- Documentation improvements are always welcome!
- Fix typos, clarify explanations, add examples
- Update outdated information
- Add new guides or tutorials

## Development Setup

### Prerequisites

- Python 3.12 or higher (see `requires-python` in `pyproject.toml`)
- Git
- pip and setuptools

### Installation

```bash
# Install in development mode
pip install -e .
```

`pyproject.toml` also defines optional extras: `service`, `docs`, `flash-attn`, and `dev`
(`pip install -e '.[docs]'` to build the documentation site). Add whichever the area you're
touching needs — the RAG package, extra eval metrics, and the PostgreSQL destination are all
part of the base install.

### Development Dependencies

```bash
pip install -e '.[dev]'
```

This installs `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-xdist`, `pytest-mock`,
`pytest-timeout`, `hypothesis`, `black`, `isort`, `ruff`, `mypy`, `bandit`, and
`pre-commit` (matching the
`[tool.pytest.ini_options]`/`[tool.black]`/`[tool.ruff]`/`[tool.mypy]`/`[tool.bandit]`
config in `pyproject.toml`). The base install does not include the test tooling — the
default `pytest` addopts (`--cov=...`) need this extra.

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all public modules, functions, classes, and methods

### Code Formatting

We use `black` for code formatting:

```bash
black src/agenttune tests/
```

### Linting

`ruff` covers linting and import sorting (`pyproject.toml`'s `[tool.ruff]` config):

```bash
ruff check src/agenttune tests/

# Run mypy for type checking
mypy src/agenttune
```

## Testing

Every push and pull request runs [`.github/workflows/ci.yml`](https://github.com/Lexsi-Labs/AgentTune/blob/main/.github/workflows/ci.yml):
an import smoke test plus the fast test subset
(`-m "not slow and not gpu and not wandb and not openenv and not qwen_e2e"`), with an
advisory `ruff` / `black` lint job. Run `pytest` locally before opening a PR.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/agenttune tests/

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v tests/
```

### Writing Tests

- Write tests for all new functionality
- Maintain or improve code coverage
- Place tests in the `tests/` directory
- Follow naming convention: `test_*.py`
- Use descriptive test names that explain what is being tested

### Test Structure

```python
def test_feature_description():
    """Test that feature works as expected."""
    # Arrange
    input_data = setup_test_data()
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Write clear, concise commit messages
   - Keep commits atomic and focused
   - Add tests for new functionality

4. **Run tests and linting**:
   ```bash
   pytest tests/
   black src/agenttune tests/
   ruff check src/agenttune tests/
   ```

5. **Update documentation**:
   - Update README.md if needed
   - Update relevant documentation in `docs/`
   - Add docstrings to new functions/classes

### Submitting the Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference related issues (e.g., "Fixes #123")
   - Provide context and motivation
   - List changes made
   - Include screenshots for UI changes

3. **PR Review Process**:
   - Address reviewer feedback promptly
   - Push additional commits to the same branch
   - Keep discussions respectful and constructive
   - Be patient - reviews may take time

### PR Requirements

- All tests pass
- Code follows style guidelines
- Documentation is updated
- Commit messages are clear
- No merge conflicts
- PR description is complete

## License Agreement

By contributing to AgentTune, you agree that:

1. **Your contributions** will be licensed under the Lexsi Labs Source Available License (LSAL) v1.1
2. **You have the right** to submit the contribution
3. **You grant** Lexsi Labs a perpetual, worldwide, royalty-free license to use, modify, distribute, and license your contributions under any terms, including commercial ones
4. **You understand** that your contributions may be used in commercial products or services offered by Lexsi Labs

This is required for contributions to be accepted into AgentTune.

## Contribution Guidelines

### Good First Issues

Look for issues labeled `good first issue` - these are suitable for newcomers.

### Areas We Need Help

- Bug fixes
- Documentation improvements
- New algorithm implementations
- Test coverage improvements
- UI/UX improvements for examples
- Translations and localization
- Performance optimizations

### What Makes a Good Contribution

- **Clear purpose**: Solves a specific problem or adds clear value
- **Well tested**: Includes appropriate tests
- **Well documented**: Code is clear and documented
- **Follows standards**: Adheres to project conventions
- **Backward compatible**: Doesn't break existing functionality (unless discussed)

## Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Usage questions and general discussion
- **Email**: [support@lexsi.ai](mailto:support@lexsi.ai) for private matters

## Recognition

Contributors will be:
- Acknowledged in release notes
- Given credit in the project

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Open a new discussion
4. Email us at support@lexsi.ai

Thank you for contributing to AgentTune!
