# Contributing to TCDOC_EXTRACT

Thank you for your interest in contributing to TCDOC_EXTRACT! This document provides guidelines for contributing to this research toolkit.

## Project Context

TCDOC_EXTRACT is a research toolkit accompanying a published academic paper. The primary purpose is reproducibility of research findings. However, we welcome contributions that:

- Improve code quality and performance
- Fix bugs or issues
- Enhance documentation
- Add support for additional data formats
- Extend functionality in useful ways

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/TCDOC_EXTRACT.git
cd TCDOC_EXTRACT
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## Development Guidelines

### Code Style

- **Python Version**: Support Python 3.8+
- **Formatting**: Use Black for code formatting
  ```bash
  black *.py
  ```
- **Linting**: Check with flake8
  ```bash
  flake8 *.py --max-line-length=100
  ```
- **Imports**: Organize imports logically (standard library, third-party, local)
- **Docstrings**: Use clear docstrings for functions and classes

### Example Style

```python
def process_action_text(text: str, mode: str = 'lemmatization') -> list:
    """
    Process action text using NLP preprocessing.

    Parameters:
        text (str): Input text to process
        mode (str): Processing mode ('lemmatization' or 'basic')

    Returns:
        list: List of processed tokens
    """
    # Implementation here
    pass
```

### Testing

While the original codebase does not include comprehensive tests, we encourage adding tests for new features:

```python
# test_your_feature.py
import pytest
from your_module import your_function

def test_your_function():
    result = your_function("test input")
    assert result == expected_output
```

Run tests with:
```bash
pytest
```

## Contribution Areas

### Priority Areas

1. **Performance Optimization**
   - Reduce API calls
   - Optimize TF-IDF computations
   - Parallelize independent operations

2. **Format Support**
   - Additional document formats (.docx improvements, .odt)
   - Better table extraction algorithms
   - HTML document support

3. **Documentation**
   - Usage examples
   - Troubleshooting guides
   - Video tutorials

4. **Robustness**
   - Better error handling
   - Input validation
   - Edge case handling

### Feature Requests

Before implementing a major feature:
1. Open an issue to discuss the proposed change
2. Wait for maintainer feedback
3. Proceed with implementation once approved

## Pull Request Process

### 1. Make Your Changes

- Keep changes focused and atomic
- Write clear commit messages
- Update documentation as needed

### 2. Test Your Changes

```bash
# Run existing scripts to ensure nothing breaks
python predefined_categories6.py  # (with test data)

# Run any tests you've added
pytest
```

### 3. Update Documentation

If your changes affect:
- **User-facing behavior**: Update README.md
- **Installation**: Update INSTALL.md
- **API**: Update API_REFERENCE.md
- **Configuration**: Document in relevant files

### 4. Submit Pull Request

```bash
git add .
git commit -m "Brief description of changes"
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- **Title**: Clear, concise description
- **Description**:
  - What changes were made
  - Why they were made
  - Any testing performed
  - Related issue numbers (if applicable)

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
Describe testing performed:
- [ ] Tested with sample data
- [ ] Unit tests added
- [ ] Existing functionality unchanged

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented if necessary)
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

### Good Examples:
```
Add support for .odt document extraction
Fix keyword matching bug in predefined_categories6.py
Update README with installation troubleshooting
Optimize TF-IDF calculation in rag_alignment_workflow
```

### Bad Examples:
```
Fixed stuff
Update
Changes
WIP
```

### Commit Message Format:
```
<type>: <subject>

<optional body>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## Reporting Issues

### Bug Reports

When reporting bugs, include:
1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - Python version
   - Operating system
   - Relevant package versions
6. **Error Messages**: Full error traceback

### Feature Requests

When requesting features, include:
1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Additional Context**: Any other relevant information

## Code Review Process

Pull requests will be reviewed for:
- **Correctness**: Does it work as intended?
- **Code Quality**: Is it readable and maintainable?
- **Documentation**: Is it properly documented?
- **Testing**: Are changes adequately tested?
- **Compatibility**: Does it maintain backward compatibility?

Reviewers may request changes before merging.

## Research Integrity

Since this is a research toolkit:
- **Do not** modify core algorithms without discussion
- **Do not** change default parameters that affect reproducibility
- **Do** add new algorithms/methods as alternatives
- **Do** clearly document any methodological changes

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0, the same license as the project.

## Questions?

If you have questions:
1. Check existing documentation (README, INSTALL, API_REFERENCE)
2. Search existing issues on GitHub
3. Open a new issue with your question

## Recognition

Contributors will be acknowledged in:
- GitHub contributors list
- CONTRIBUTORS.md file (if created)
- Release notes for significant contributions

Thank you for helping improve TCDOC_EXTRACT!

---

**Maintained by**: Dr. Samuel J Jackson
**Last Updated**: December 2025
