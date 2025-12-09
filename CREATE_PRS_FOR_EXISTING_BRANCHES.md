# Creating Pull Requests for Existing Branches

This guide shows how to create Pull Requests for your existing task branches (task-1, task-3, task-4) that haven't been merged via PR yet.

## Current Branch Status

You have the following branches that can be merged via PR:
- `task-1` - Initial development branch
- `task-3` - Hypothesis testing implementation
- `task-4` - Advanced modeling implementation

## Step-by-Step: Create PR for Existing Branch

### Option 1: Using GitHub Web Interface (Recommended)

1. **Go to your repository on GitHub**
   - Navigate to: `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`

2. **Click "Pull Requests" tab**
   - You'll see a list of existing PRs (if any)

3. **Click "New Pull Request"**

4. **Select branches:**
   - **Base branch**: `main` (the branch you want to merge INTO)
   - **Compare branch**: `task-3` (the branch you want to merge FROM)
   - GitHub will show you the diff

5. **Fill out PR template:**
   - The PR template will automatically appear
   - Fill out all relevant sections:
     - **Description**: What this task accomplished
     - **Changes Made**: List key changes
     - **Testing**: What testing was done
     - **Business Impact**: Impact of these changes

6. **Example PR Description for task-3:**
   ```
   ## Description
   Implements comprehensive hypothesis testing framework (Task 3) with statistical validation of risk drivers.
   
   ## Changes Made
   - Implemented hypothesis testing module (`src/eda/hypothesis_testing.py`)
   - Added statistical tests for province, zipcode, and gender risk differences
   - Integrated hypothesis testing into EDA pipeline
   - Generated business recommendations based on test results
   - Fixed data normalization and encoding issues
   
   ## Testing
   - ✅ Unit tests pass
   - ✅ DVC pipeline runs successfully
   - ✅ Hypothesis tests execute correctly
   - ✅ Reports generated successfully
   
   ## Business Impact
   - Identifies statistically significant risk differences across regions
   - Provides data-driven recommendations for premium adjustments
   - Enables evidence-based pricing decisions
   ```

7. **Add labels** (optional):
   - `task-3`
   - `enhancement`
   - `documentation`

8. **Request review** (if applicable):
   - Assign reviewers if you have collaborators
   - Or mark as ready for review

9. **Click "Create Pull Request"**

### Option 2: Using GitHub CLI (if installed)

```bash
# Install GitHub CLI if not installed
# https://cli.github.com/

# Create PR from command line
gh pr create \
  --base main \
  --head task-3 \
  --title "task-3: Implement hypothesis testing with statistical validation" \
  --body-file .github/pull_request_template.md
```

## PR Creation Checklist

Before creating the PR, ensure:

- [ ] Branch is up to date with latest changes
- [ ] All commits have descriptive messages
- [ ] Code follows project style guidelines
- [ ] Tests are passing
- [ ] Documentation is updated
- [ ] DVC pipeline runs successfully
- [ ] No merge conflicts with main

## After Creating PR

1. **Monitor CI/CD Status:**
   - Check that all CI/CD checks pass
   - Fix any failing checks

2. **Respond to Reviews:**
   - Address review comments
   - Make requested changes
   - Re-request review after changes

3. **Merge When Ready:**
   - After approval, merge via GitHub UI
   - Use "Squash and Merge" for task branches
   - This creates a single commit on main

4. **Cleanup:**
   ```bash
   # After merge, delete local branch
   git checkout main
   git pull origin main
   git branch -d task-3
   
   # Delete remote branch (or use GitHub UI)
   git push origin --delete task-3
   ```

## Example: Creating PR for task-3

### 1. Verify branch status:
```bash
git checkout task-3
git log --oneline -5
# Should show commits like:
# 2525a5a task-3: add completion status document
# 2db1ab1 task-3: fix hypothesis testing...
```

### 2. Ensure branch is pushed:
```bash
git push origin task-3
```

### 3. Create PR on GitHub:
- Base: `main`
- Compare: `task-3`
- Title: `task-3: Implement hypothesis testing with statistical validation`
- Description: Fill out PR template

### 4. Wait for CI/CD checks:
- All checks should pass
- Fix any issues if they fail

### 5. Request review (if applicable)

### 6. Merge after approval:
- Use "Squash and Merge"
- Confirm merge

### 7. Cleanup:
```bash
git checkout main
git pull origin main
git branch -d task-3
```

## PR Best Practices

### Do's ✅
- Fill out PR template completely
- Write clear, descriptive titles
- Link related issues/tasks
- Request review before merging
- Respond to feedback promptly
- Keep PR focused (one task per PR)

### Don'ts ❌
- Don't skip the PR template
- Don't merge your own PRs without review (if required)
- Don't create PRs with incomplete work
- Don't ignore CI/CD failures
- Don't merge with merge conflicts

## Troubleshooting

### Merge Conflicts
If there are merge conflicts:
```bash
git checkout task-3
git fetch origin
git merge origin/main
# Resolve conflicts
git add .
git commit -m "task-3: resolve merge conflicts with main"
git push origin task-3
```

### CI/CD Failures
- Check the error messages
- Fix issues locally
- Push fixes to branch
- CI/CD will re-run automatically

### PR Not Showing Changes
- Ensure branch is pushed: `git push origin task-3`
- Check that you're comparing the right branches
- Verify commits exist on remote branch

## Next Steps

1. **Create PRs for all existing branches:**
   - task-1 → main
   - task-3 → main
   - task-4 → main

2. **Use PR workflow for future work:**
   - Always create PRs for new task branches
   - Follow the PR guide for best practices

3. **Review and merge:**
   - Review PRs using code review guidelines
   - Merge when approved

## Resources

- [Pull Request Guide](.github/PULL_REQUEST_GUIDE.md) - Complete PR workflow
- [Code Review Guidelines](.github/CODE_REVIEW_GUIDELINES.md) - Review best practices
- [PR Template](.github/pull_request_template.md) - PR template

