# Publishing yaac to PyPI

This document describes how to publish the `yaac` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **TestPyPI Account** (optional, for testing): Create an account at https://test.pypi.org/account/register/
3. **Build Tools**: Install `uv` (recommended) or `build` and `twine`:
   ```bash
   # Using uv (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or using pip
   pip install build twine
   ```

## Pre-Publishing Checklist

- [x] Build system configured in `pyproject.toml`
- [x] Package metadata (version, description, classifiers) added
- [x] Package exports defined in `__init__.py` files
- [x] `.gitignore` excludes build artifacts (`build/`, `dist/`, `*.egg-info/`)
- [x] LICENSE file present
- [x] README.md present and informative
- [ ] Tests pass
- [ ] Version number updated (if needed)

## Building the Package

1. **Clean previous builds** (if any):
   ```bash
   rm -rf build/ dist/ *.egg-info/
   ```

2. **Build the package**:
   ```bash
   # Recommended: Using uv (handles environment automatically)
   uv build
   
   # Alternative: Using python -m build
   python -m build
   ```

   This creates source distribution (`.tar.gz`) and wheel (`.whl`) files in `dist/`.

3. **Verify the build**:
   ```bash
   # Check the built files
   ls -lh dist/
   
   # Verify the package contents
   tar -tzf dist/yaac-*.tar.gz | head -20
   ```

## Testing on TestPyPI (Optional)

**Note**: TestPyPI doesn't allow deleting or overwriting existing versions. If you need to re-test the same version, use a post/dev suffix (e.g., `0.1.4.post1`) or skip TestPyPI and test directly on production PyPI (packages can be deleted within 30 days).

1. **Upload to TestPyPI**:
   ```bash
   # Using helper script (recommended)
   ./publish/upload_to_testpypi.sh
   
   # Or manual upload
   python -m twine upload --repository testpypi dist/*
   ```

2. **Test installation from TestPyPI**:
   ```bash
   # Create temporary test environment
   uv venv test_yaac_env
   source test_yaac_env/bin/activate
   
   # Install from TestPyPI
   uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ yaac
   ```

3. **Verify the installation**:
   ```bash
   python -c "import yaac; print(yaac.__version__)"
   python -c "from yaac.common.model_loader import load_model_from_checkpoint; print('Imports work!')"
   ```
   
   When done testing, deactivate and remove the test environment:
   ```bash
   deactivate
   rm -rf test_yaac_env
   ```

## Publishing to Production PyPI

1. **Upload to PyPI**:
   ```bash
   # Using helper script (recommended)
   ./publish/upload_to_pypi.sh
   
   # Or manual upload
   python -m twine upload dist/*
   ```

2. **Verify the publication**:
   ```bash
   # Check package page
   # https://pypi.org/project/yaac/
   
   # Test installation in fresh environment
   uv venv test_yaac_prod
   source test_yaac_prod/bin/activate
   uv pip install yaac
   python -c "import yaac; print(yaac.__version__)"
   python -c "from yaac.common.model_loader import load_model_from_checkpoint; print('Imports work!')"
   deactivate
   rm -rf test_yaac_prod
   ```

## Updating the Package

When you need to publish a new version:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # or "0.2.0", etc.
   ```

2. **Update `__version__`** in `yaac/__init__.py`:
   ```python
   __version__ = "0.1.1"
   ```

3. **Follow the build and upload steps above**

## API Token Setup

1. Create API tokens at:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - Production: https://pypi.org/manage/account/token/
2. Create `.env` file in project root:
   ```bash
   TESTPYPI_TOKEN=pypi-your-testpypi-token-here
   PYPI_TOKEN=pypi-your-pypi-token-here
   ```
3. Helper scripts (`./publish/upload_to_*.sh`) automatically use tokens from `.env`

## Troubleshooting

### "ensurepip is not available" error

If `python -m build` fails with this error:
- **Quick fix**: Use `uv build` instead (recommended)
- **Alternative**: Install `python3-venv` for your Python version (e.g., `sudo apt install python3.12-venv`)

### "400 Bad Request" error

Common causes:
- **Version already exists**: Use post/dev suffix (e.g., `0.1.4.post1`) or skip TestPyPI
- **Missing files**: Ensure both `.tar.gz` and `.whl` are in `dist/`
- **Invalid metadata**: Validate with `twine check dist/*`
- **Token permissions**: Ensure token has "Upload packages" scope

Get detailed error: `python -m twine upload --repository testpypi dist/* --verbose`

## Package Structure

Main exports:
- `yaac.common.model_loader.load_model_from_checkpoint`
- `yaac.common.trainable_model.TrainableModel`
- `yaac.models.sic.SIC`, `yaac.models.sic.make_model`
