# Jupyter Book Reporting Layer

This folder is a GitHub-synced reporting interface for the project. The production repository remains the source of truth; these MyST Markdown pages summarize and inspect tracked files only.

The book also serves as a project evolution record: later pages expose tracked scripts, pipeline-level code organization, and recent GitHub commit evolution for project review.

## What Is Included

The pages use files tracked in GitHub, such as README files, documentation, scripts, configs, and data manifests. Ignored local data, local outputs, private paths, model artifacts, and under-construction local experimental work are not included unless they are later committed to the repository.

## Open On GitHub

Each chapter is stored as a plain MyST Markdown `.md` page. GitHub renders the text source cleanly, while GitHub Actions builds the full Jupyter Book site for richer tables, code displays, and executed repository summaries.

## Build Locally

From the repository root:

```powershell
python -m pip install -r requirements-notebooks.txt
jupyter-book build notebooks
```

The local static site will be written to:

```text
notebooks/_build/html/
```

Preview locally by opening:

```text
notebooks/_build/html/index.html
```

## GitHub Actions And Pages

The workflow at `.github/workflows/build-jupyter-book.yml` builds the Jupyter Book after each push to `main`. It uses only the notebook/reporting dependency file and does not run project pipelines, download data, validate external URLs, access Zenodo, train models, or inspect ignored local outputs.

To publish with GitHub Pages:

1. Open the repository on GitHub.
2. Go to **Settings** -> **Pages**.
3. Set **Source** to **GitHub Actions**.
4. Push to `main` or manually run the workflow.

If Pages is not enabled, the workflow can still build the site artifact for review in the Actions run.

## Boundary

These pages are a reporting and navigation layer. They do not migrate production code into the book, rewrite scientific logic, or replace the command-line workflows.

Some pages include executable MyST code cells for lightweight repository inspection. Routine setup and helper cells are hidden in the rendered Jupyter Book so the site reads like a project report. Technically interested readers can still inspect the plain Markdown source directly on GitHub.
