# Jhony H. Giraldo — academic website, version 2

This is a separate Quarto-based renewal of the current website. It does not alter or deploy the existing `jhonygiraldo.github.io` repository.

## Local preview

Install [Quarto](https://quarto.org/docs/get-started/) and run:

```bash
quarto preview
```

The site is generated in `_site/`. LaTeX equations are rendered with MathJax.

## Routine updates

- News: add one object to `_data/news.json`.
- Team members: edit `_data/people.json`.
- Publications: edit `_bibliography/papers.bib`; use unique citation keys and separate authors with `and`.
- Courses: edit `_data/courses.json` and add lecture `.qmd` files beneath `teaching/`.
- CV: replace `assets/pdf/Jhony-Giraldo-CV.pdf` with the new PDF, keeping the filename stable.

The pre-render script turns structured data into reusable page fragments. Validate an update with:

```bash
python3 scripts/render_content.py --check
quarto render
```

The GitHub Actions workflow performs both steps before publication.

