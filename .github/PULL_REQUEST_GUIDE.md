# Pull Request Workflow Guide

This guide outlines best practices for creating, reviewing, and merging Pull Requests (PRs) in this project.

## Overview

All feature work and significant changes should be done in task branches and merged into `main` via Pull Requests. This ensures:
- **Code Review**: Changes are reviewed before merging
- **Documentation**: PRs document what changed and why
- **History**: Clear change history with context
- **Quality**: Issues caught before merging to main

## Creating a Pull Request

### 1. Work on a Task Branch

```bash
# Create and switch to a new task branch
git checkout -b task-X

# Make your changes and commit
git add .
git commit -m "task-X: descriptive commit message"

# Push branch to remote
git push origin task-X
```

### 2. Create PR on GitHub

1. Go to your repository on GitHub
2. Click "Pull Requests" tab
3. Click "New Pull Request"
4. Select:
   - **Base branch**: `main`
   - **Compare branch**: `task-X`
5. Fill out the PR template with:
   - Clear description
   - Changes made
   - Testing performed
   - Business impact

### 3. PR Title Format

Use descriptive titles:
- ✅ `task-3: Implement hypothesis testing with statistical validation`
- ✅ `task-4: Add advanced modeling with SHAP interpretability`
- ❌ `fix`
- ❌ `updates`

### 4. PR Description

Use the PR template and include:
- **What**: What changes were made
- **Why**: Why these changes were needed
- **How**: How the changes were implemented
- **Testing**: What testing was done
- **Impact**: Business/technical impact

## Review Process

### For Authors

1. **Self-Review First**
   - Review your own code before requesting review
   - Check for:
     - Code quality and style
     - Documentation completeness
     - Test coverage
     - Logging standardization

2. **Request Review**
   - Assign reviewers (if applicable)
   - Add labels (e.g., `task-3`, `enhancement`, `documentation`)
   - Link related issues/tasks

3. **Respond to Feedback**
   - Address review comments promptly
   - Make requested changes
   - Update PR description if needed
   - Re-request review after changes

### For Reviewers

1. **Review Checklist**
   - [ ] Code follows project conventions
   - [ ] Tests are adequate
   - [ ] Documentation is updated
   - [ ] No breaking changes (or documented)
   - [ ] DVC pipeline still works
   - [ ] Logging is standardized

2. **Provide Constructive Feedback**
   - Be specific about issues
   - Suggest improvements
   - Approve when satisfied

3. **Approve or Request Changes**
   - Approve: Ready to merge
   - Request changes: Needs fixes before merging

## Merging PRs

### Merge Methods

1. **Squash and Merge** (Recommended for task branches)
   - Combines all commits into one
   - Cleaner history
   - Use for: Feature branches, task branches

2. **Merge Commit**
   - Preserves all commit history
   - Use for: Long-running branches, complex features

3. **Rebase and Merge**
   - Linear history
   - Use for: Small changes, hotfixes

### Merge Process

1. **Pre-Merge Checks**
   - [ ] All CI/CD checks pass
   - [ ] At least one approval (if required)
   - [ ] No merge conflicts
   - [ ] PR description is complete

2. **Merge**
   - Click "Merge Pull Request"
   - Choose merge method (usually "Squash and Merge")
   - Confirm merge

3. **Post-Merge**
   - Delete branch (local and remote)
   - Update project board
   - Notify stakeholders if needed

## Branch Management

### Branch Naming

- Task branches: `task-1`, `task-2`, `task-3`, etc.
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Hotfixes: `hotfix/description`

### Branch Lifecycle

1. **Create**: `git checkout -b task-X`
2. **Develop**: Make commits with descriptive messages
3. **Push**: `git push origin task-X`
4. **PR**: Create Pull Request
5. **Review**: Address feedback
6. **Merge**: Merge via PR
7. **Cleanup**: Delete branch after merge

## Example Workflow

### Complete Example: Task 3 PR

```bash
# 1. Create branch
git checkout -b task-3

# 2. Make changes and commit
git add src/eda/hypothesis_testing.py
git commit -m "task-3: implement hypothesis testing framework"

git add src/eda/run_eda.py
git commit -m "task-3: add statistical tests to EDA pipeline"

# 3. Push to remote
git push origin task-3

# 4. Create PR on GitHub (use template)
# Title: "task-3: Implement hypothesis testing with statistical validation"
# Description: Fill out PR template

# 5. Address review feedback
git add .
git commit -m "task-3: address review feedback - fix encoding issues"
git push origin task-3

# 6. After approval, merge via GitHub UI
# 7. Cleanup
git checkout main
git pull origin main
git branch -d task-3
git push origin --delete task-3
```

## PR Best Practices

### Do's ✅

- Create focused PRs (one task/feature per PR)
- Write clear, descriptive commit messages
- Fill out PR template completely
- Link related issues/tasks
- Request review from appropriate reviewers
- Respond to feedback promptly
- Keep PRs small and manageable
- Test before requesting review

### Don'ts ❌

- Don't merge your own PRs without review (if required)
- Don't force push to shared branches
- Don't skip the PR template
- Don't create PRs with incomplete work
- Don't ignore review feedback
- Don't merge with failing CI/CD checks

## CI/CD Integration

PRs automatically trigger:
- Code linting (flake8, black)
- Unit tests (pytest)
- DVC pipeline validation (`dvc dag`)
- Documentation checks

All checks must pass before merging.

## Documentation

PRs should update:
- Code docstrings (if functions changed)
- README.md (if setup/usage changed)
- DVC pipeline docs (if pipeline changed)
- Task completion docs (if task completed)

## Questions?

If you have questions about the PR process:
1. Check this guide
2. Review existing PRs for examples
3. Ask in team discussions
4. Refer to GitHub documentation

