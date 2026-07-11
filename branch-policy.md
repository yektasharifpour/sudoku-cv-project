# Team Workflow & Branch Policy

---

## Branches

### `main`

- Stable and releasable at all times
- Protected
- No direct pushes
- Changes only enter through Pull Requests

### `develop`

- Integration branch for ongoing work
- Protected
- Used for combining completed feature work before release

### Working branches

Create short-lived branches from `develop` using one of these prefixes:

- `phasex/feature/<name>`
- `documents/phasex`
- `phasex/bugfix/<name>`
- `phasex/refactor/<name>`
- `phasex/docs/<name>`

Examples:

```text
phase1/feature/grid-extraction
phase1/refactor/grid-extraction
phase1/bugFix/render-loop
phase1/docs/...

```

---

## Workflow

### Normal development

text
feature/\* -> develop -> main

1. Create a branch from `develop`
2. Make your changes
3. Push your branch
4. Open a Pull Request into `develop`
5. After review and checks, merge into `develop`
6. When ready for release, open a Pull Request from `develop` into `main`

### Hotfixes

text
hotfix/\* -> main

1. Create a `hotfix/*` branch from `main`
2. Implement the fix
3. Open a Pull Request into `main`
4. After merge, back-merge the fix into `develop`

---

## Pull Request Guidelines

Each Pull Request should:

- Focus on one logical change
- Have a clear and descriptive title
- Include a short summary of what changed
- Reference the related issue if applicable
- Pass all required checks before merge
- Resolve all review comments before merge

Example PR titles:

text
feat(player): add jump handling
fix(enemy): correct patrol direction logic
refactor(renderer): separate Vulkan init stage

---

## Merge Strategy

Preferred merge strategy:

- Squash merge into `develop`
- Squash merge or rebase merge into `main`

This keeps history clean and easier to read.

---

## Review Policy

### For PRs into `develop`

- At least 1 approval is required

### For PRs into `main`

- At least 2 approvals are required

Reviewers should check:

- correctness
- readability
- consistency with project architecture
- test/build impact
- possible regressions

---
