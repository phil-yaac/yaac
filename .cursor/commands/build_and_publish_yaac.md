# Build and Publish yaac Package

## Instruction

You are helping to build and publish the `yaac` package to PyPI. **Follow the steps in `yaac/publish/PUBLISHING.md` step-by-step**. Do not skip steps or proceed if you encounter errors.

## Key Principles

1. **Reference the documentation**: The full publishing guide is in `yaac/publish/PUBLISHING.md`. Read it first and follow it. Do not duplicate information from that doc here.

2. **Go step-by-step**: Complete each step fully before moving to the next. Verify each step succeeded before proceeding.

3. **Stop on errors**: If you encounter any error, warning, or unexpected output:
   - **STOP immediately**
   - Report the error clearly to the user
   - Ask the user how they want to proceed
   - Do NOT attempt to fix errors without user approval (unless it's a trivial issue you're certain about)

4. **Verify at each stage**: After each major step (build, upload, install), verify it worked correctly before continuing.

## Workflow

### 1. Pre-flight Checks

Before starting:
- Read `yaac/publish/PUBLISHING.md` to understand the full process
- Check current version in `pyproject.toml` and `yaac/__init__.py` (they should match)
- Verify you're in the correct directory (`/home/phil/code/yaac`)
- Check if build tools are available (`uv` or `python -m build`)

### 2. Building the Package

Follow the "Building the Package" section in PUBLISHING.md:
- Clean previous builds
- Build using `uv build` (recommended) or `python -m build`
- Verify the build created both `.tar.gz` and `.whl` files in `dist/`
- **If build fails**: Stop and ask the user. Check the troubleshooting section in PUBLISHING.md.

### 3. Testing on TestPyPI (Optional)

If the user wants to test on TestPyPI:
- Follow the "Testing on TestPyPI" section in PUBLISHING.md
- Upload to TestPyPI
- Create a fresh test environment and install from TestPyPI
- Verify the installation works
- **If upload fails with "400 Bad Request"**: Check if version already exists. Stop and ask user if they want to use a post/dev suffix or skip TestPyPI.

### 4. Publishing to Production PyPI

Follow the "Publishing to Production PyPI" section in PUBLISHING.md:
- Upload to production PyPI
- Verify the publication (check package page, test installation)
- **If upload fails**: Stop and ask the user. Check the troubleshooting section.

## Error Handling

When you encounter an error:
1. **Stop immediately** - do not proceed
2. **Read the error message carefully** - check if it's covered in PUBLISHING.md troubleshooting
3. **Report to user** - explain what happened and what step failed
4. **Suggest solutions** - reference PUBLISHING.md troubleshooting or ask user for guidance
5. **Wait for user approval** - do not attempt fixes without user confirmation

## Updating Documentation

Only update `yaac/publish/PUBLISHING.md` or this command file if:
- You discover a **critical issue** that would affect future runs
- You find a **better approach** that significantly improves the process
- The documentation is **missing essential information** that caused a failure

Do NOT update documentation for:
- One-off issues specific to the current run
- Minor clarifications that don't affect the process
- Personal preferences or style changes

If you do update documentation, explain clearly what changed and why.

## User Communication

- **Be clear about what you're doing**: State which step you're on
- **Report progress**: Confirm when steps complete successfully
- **Ask before proceeding**: If a step requires user input (like confirming upload), ask first
- **Summarize at the end**: After successful publication, summarize what was done

## Example Flow

```
1. "Reading PUBLISHING.md to understand the process..."
2. "Checking current version in pyproject.toml and __init__.py..."
3. "Cleaning previous builds..."
4. "Building package with uv build..."
5. "Build successful! Created dist/yaac-X.X.X.tar.gz and dist/yaac-X.X.X-py3-none-any.whl"
6. "Would you like to test on TestPyPI first, or proceed directly to production PyPI?"
7. [After user choice, proceed with upload and verification]
8. "Publication complete! Package is available at https://pypi.org/project/yaac/"
```
