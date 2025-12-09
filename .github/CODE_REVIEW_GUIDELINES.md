# Code Review Guidelines

This document outlines guidelines for conducting effective code reviews in this project.

## Review Objectives

Code reviews should ensure:
1. **Correctness**: Code works as intended
2. **Quality**: Code follows best practices
3. **Maintainability**: Code is readable and maintainable
4. **Consistency**: Code follows project conventions
5. **Documentation**: Code is well-documented

## Review Checklist

### Code Quality

- [ ] Code follows Python style guidelines (PEP 8)
- [ ] Functions have clear, concise docstrings
- [ ] Variable names are descriptive
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] No hardcoded values (use config/constants)
- [ ] Error handling is appropriate
- [ ] Logging is standardized (no print statements)

### Functionality

- [ ] Code implements the intended feature correctly
- [ ] Edge cases are handled
- [ ] Error cases are handled gracefully
- [ ] No obvious bugs or logic errors
- [ ] Performance considerations addressed

### Testing

- [ ] Unit tests are present and passing
- [ ] Tests cover edge cases
- [ ] Integration tests pass (if applicable)
- [ ] DVC pipeline runs successfully
- [ ] Manual testing completed (if needed)

### Documentation

- [ ] Docstrings are present and complete
- [ ] README updated (if needed)
- [ ] Comments explain "why" not "what"
- [ ] PR description is clear and complete

### Data & Models

- [ ] Data files tracked with DVC (if new)
- [ ] Model artifacts properly versioned
- [ ] Feature engineering is documented
- [ ] Data preprocessing steps are clear

### DVC Pipeline

- [ ] `dvc.yaml` is updated correctly
- [ ] Dependencies are specified
- [ ] Outputs are tracked
- [ ] Parameters are documented
- [ ] Pipeline is reproducible

## Review Process

### 1. Initial Review

- Read the PR description
- Understand the context and goals
- Review the code changes
- Check CI/CD status

### 2. Detailed Review

- Review each file changed
- Check for issues using the checklist
- Test locally (if needed)
- Verify DVC pipeline

### 3. Provide Feedback

- Be constructive and specific
- Explain the "why" behind suggestions
- Suggest improvements, not just problems
- Acknowledge good practices

### 4. Approve or Request Changes

- **Approve**: Code is ready to merge
- **Request Changes**: Issues need to be addressed

## Review Comments

### Good Review Comments ✅

```python
# Good: Specific and constructive
"Consider using logger.info() instead of print() here for consistency with the rest of the codebase."

# Good: Explains why
"This function could benefit from a docstring explaining the assumptions about the input data format."

# Good: Suggests improvement
"Could we add error handling here? What happens if df is empty?"
```

### Poor Review Comments ❌

```python
# Poor: Vague
"This looks wrong."

# Poor: Not constructive
"Don't do this."

# Poor: Personal preference without reason
"I don't like this approach."
```

## Common Issues to Look For

### Logging

- ❌ `print()` statements
- ✅ `logger.info()`, `logger.warning()`, `logger.error()`

### Documentation

- ❌ Missing docstrings
- ❌ Incomplete docstrings (no Args/Returns/Assumptions)
- ✅ Complete docstrings with all sections

### Error Handling

- ❌ Silent failures
- ❌ Generic exception catching
- ✅ Specific exception handling with logging

### Code Organization

- ❌ Long functions (>50 lines)
- ❌ Duplicated code
- ✅ Modular, reusable functions

### Data Processing

- ❌ Missing data validation
- ❌ No handling of edge cases
- ✅ Explicit data cleaning steps

## Review Response Time

- **Initial Review**: Within 24 hours
- **Re-review**: Within 12 hours after changes
- **Urgent PRs**: As soon as possible

## Approval Criteria

A PR should be approved when:
1. All checklist items are satisfied
2. Code quality is acceptable
3. Tests are passing
4. Documentation is complete
5. No blocking issues remain

## Disagreements

If there's disagreement:
1. Discuss in PR comments
2. Reference project guidelines
3. Seek consensus
4. Escalate if needed (project maintainer)

## Examples

### Example 1: Good PR Review

**Reviewer**: "The hypothesis testing implementation looks solid. A few suggestions:
1. Consider adding a docstring to `test_province_risk_differences()` explaining the statistical assumptions
2. The logging is well-standardized - good work!
3. One edge case: what happens if all provinces have the same loss ratio? Consider adding a check.

Overall, this is ready to merge after addressing the docstring."

### Example 2: Needs Work

**Reviewer**: "I see several issues:
1. There are `print()` statements on lines 45, 67 - these should use logger
2. Missing docstrings for `calculate_margin()` function
3. The error handling could be more specific - catching generic `Exception` is too broad
4. DVC pipeline validation failed - check `dvc.yaml` syntax

Please address these before re-requesting review."

## Questions?

If you're unsure about something:
1. Ask clarifying questions in PR comments
2. Reference these guidelines
3. Check similar code in the codebase
4. Consult with the team

