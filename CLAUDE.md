# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Felipe Lamarca's talks and workshops, one folder per presentation, published as a gallery at <https://felipelamarca.com/Presentations/>. The repository is a Quarto `website` project: CI renders the revealjs decks, generates their PDFs and writes the gallery page. The Beamer talks are kept exactly as presented and published as their committed PDFs.

## Two generations of decks

- **revealjs (2026 onward):** `AgentesIA-FMMAPE-2026/slides.qmd`, `Intro-to-ClaudeCode-CERES-2026/slides.qmd`, `Intro-to-ClaudeCode-LABIIA-2026/dia{1,2}.qmd` (course style) and `Quantos-Votos-ANPOCS-2026/slides.qmd` (academic style, with OJS charts).
  - They use the Plenário Slides extension (`_extensions/felipelmc/plenario/`, from [felipelmc/Slides-Template](https://github.com/felipelmc/Slides-Template)) with `format: plenario-revealjs`.
  - `estilo: curso` is for courses and workshops; `estilo: academico` is for papers.
  - Components, front matter and the PDF behaviour are documented in the Slides-Template README.
  - Deck-specific notes (title badge, assets, what changed and why) live in each folder's own `CLAUDE.md`, or `README.md` for ANPOCS. Read it before editing that deck.
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

- The project runs two post-render scripts from the extension, in order: `tools/pdf.py` (decktape PDFs) and then `tools/galeria.py` (writes `_site/index.html` from `palestras.yml`).
- PDFs need Node (for `npx decktape`) and Chrome. They are cached in `.quarto/plenario-pdf/`, keyed by a hash of each deck's HTML, CSS and images.
- `gerar-pdf` is on by default for `estilo: academico` and off for `estilo: curso`, which is why the course decks set `gerar-pdf: true`. If `palestras.yml` lists a `pdf:` that the deck doesn't produce, the gallery reports a missing file.
- The gallery needs PyYAML (`pip install pyyaml`). Its covers come from `pdftoppm` (poppler), using the first page of each PDF.
- OJS charts don't load over `file://`. View a deck with them through `quarto preview` or a local server; `pdf.py` serves `_site` over HTTP for the same reason.
- CI (`.github/workflows/publish.yml`) pins Quarto 1.9.35 and runs `check.py pre`, `quarto render` and `check.py post` before deploying to GitHub Pages. It can be run manually without PDFs (`sem_pdf`).
- Nothing under `_site/`, no `*_files/` and no rendered `.html` or revealjs `.pdf` is committed; CI builds them.

## Adding a presentation

```bash
bash _extensions/felipelmc/plenario/tools/nova-palestra.sh Pasta-Evento-AAAA academico "Título" AAAA-MM-DD "Evento"
```

The script creates `Pasta-Evento-AAAA/slides.qmd` and appends an entry to `palestras.yml`, which drives the gallery (title, date, event, section, links). Then:

- Keep deck files named `slides.qmd` or `dia*.qmd`. Those are the render globs in `_quarto.yml`, and any other name is skipped without a warning.
- For a `curso` deck, uncomment `pdf:` in the new `palestras.yml` entry only after adding `gerar-pdf: true` to the deck.
- Add a row to the table in `README.md` by hand; the script doesn't touch it.
- In CI, a full render fails if a rendered deck is missing from `palestras.yml` or a file it lists doesn't exist. Locally these are only `galeria: ⚠` warnings in the render output, so check for them.

## Rules

- No inline HTML, no `style=`, no hard-coded `background-color` in decks. Use the components (`.divisoria`, `.escuro`, `.encerramento`, `::: agenda`, `:::: cards`, `.tela`, `:::: janela`, …). `check.py pre` enforces this.
- Never put `---` right before a `#` heading: it creates a blank slide.
- Event logos (`logo-mape.svg`, `logo-ceres.svg`, `logo-labiia.png`) are committed; the title-slide badge needs them on the published site.
- Emoji in the workshop decks are content, not decoration: keep them.
- Update the extension only through a tag: `quarto update extension felipelmc/Slides-Template@vX.Y.Z --no-prompt`. Never `quarto add ../Slides-Template`: the local path drops the `felipelmc/` folder. The next update overwrites any edit made inside `_extensions/felipelmc/plenario/`, so fix the extension upstream in Slides-Template.
- webR is not committed (`_extensions/coatless/` is gitignored). A deck that uses `{webr-r}` needs `quarto add coatless/quarto-webr@0.4.3 --no-prompt` locally, `engine: markdown` in its front matter and the install step uncommented in the CI workflow. If the deck has no PDF, give it a `capa:` image in `palestras.yml`.

## Language

The presenter writes primarily in Portuguese (pt-BR); slide content, file names and commit messages are mostly in Portuguese. `SICSS-2025` is in English.
