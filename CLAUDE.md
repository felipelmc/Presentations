# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Felipe Lamarca's talks and workshops, one folder per presentation, published as a gallery at <https://felipelamarca.com/Presentations/>. The repository is a Quarto `website` project: CI renders the revealjs decks, generates their PDFs and writes the gallery page. The Beamer talks are kept exactly as presented and published as their committed PDFs.

## Two generations of decks

- **revealjs (2026 onward):** `AgentesIA-FMMAPE-2026/slides.qmd`, `Intro-to-ClaudeCode-CERES-2026/slides.qmd`, `Intro-to-ClaudeCode-LABIIA-2026/dia{1,2}.qmd` (course style) and `Quantos-Votos-ANPOCS-2026/slides.qmd` (academic style, with OJS charts).
  - They use the Plenário Slides extension (`_extensions/felipelmc/plenario/`, from [felipelmc/Slides-Template](https://github.com/felipelmc/Slides-Template)) with `format: plenario-revealjs`.
  - `estilo: curso` is for courses and workshops; `estilo: academico` is for papers.
  - Components, front matter and the PDF behaviour are documented in the Slides-Template README.
- **Beamer (2025):** `BIEN-2025`, `GT-Jornada-Discente-IESP-2025`, `ML-FMMAPE-2025`, `Minicurso-DL-Jornada-Discente-IESP-2025`, `SICSS-2025`.
  - These are frozen: their `.qmd` never match the render globs, and their committed PDFs are copied to the site as resources.
  - Rebuilding one would need LaTeX locally and a folder-level `_quarto.yml` with `type: default`.

## Rendering

From the repository root:

```bash
quarto preview AgentesIA-FMMAPE-2026/slides.qmd   # one deck, no PDF
quarto render                                     # whole site in _site/: decks, PDFs (decktape), gallery
PLENARIO_PDF=0 quarto render                      # skip PDFs
python3 _extensions/felipelmc/plenario/tools/check.py pre
python3 _extensions/felipelmc/plenario/tools/check.py post _site render.log
```

- PDFs need Node (for `npx decktape`) and Chrome. They are cached in `.quarto/plenario-pdf/`.
- Covers for the gallery come from `pdftoppm` (poppler).
- Nothing under `_site/`, no `*_files/` and no rendered `.html` or revealjs `.pdf` is committed; CI builds them.

## Adding a presentation

```bash
bash _extensions/felipelmc/plenario/tools/nova-palestra.sh Pasta-Evento-AAAA academico "Título" AAAA-MM-DD "Evento"
```

The script creates `Pasta-Evento-AAAA/slides.qmd` and appends an entry to `palestras.yml`, which drives the gallery (title, date, event, section, links). In CI the render fails if a rendered deck is missing from `palestras.yml`.

## Rules

- No inline HTML, no `style=`, no hard-coded `background-color` in decks. Use the components (`.divisoria`, `.escuro`, `.encerramento`, `::: agenda`, `:::: cards`, `.tela`, `:::: janela`, …). `check.py pre` enforces this.
- Never put `---` right before a `#` heading: it creates a blank slide.
- Event logos (`logo-mape.svg`, `logo-ceres.svg`, `logo-labiia.png`) are committed; the title-slide badge needs them on the published site.
- Emoji in the workshop decks are content, not decoration: keep them.
- Update the extension only through a tag: `quarto update extension felipelmc/Slides-Template@vX.Y.Z --no-prompt`. Never `quarto add ../Slides-Template`: the local path drops the `felipelmc/` folder.

## Language

The presenter writes primarily in Portuguese (pt-BR); slide content, file names and commit messages are mostly in Portuguese. `SICSS-2025` is in English.
