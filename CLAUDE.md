# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is an archive of academic talks and workshops given by Felipe Lamarca (IESP-UERJ). Each top-level directory is one self-contained presentation — slide source, bibliography, images, and any accompanying demo code/notebooks — with no shared build system or dependencies between folders. Adding a new talk means adding a new sibling directory; see `README.md` for the running list of presentations with dates and links.

## Rendering slides

All presentations are authored in Quarto (`.qmd`). Render from inside the presentation's own directory (relative image/bibliography paths depend on it):

```bash
cd <presentation-dir>
quarto render <file>.qmd
```

Two output formats are used depending on the talk:

- **`format: beamer`** (PDF, via `pdf-engine: lualatex`) — the research talks: `BIEN-2025`, `GT-Jornada-Discente-IESP-2025`, `ML-FMMAPE-2025`, `Minicurso-DL-Jornada-Discente-IESP-2025`, `SICSS-2025`. These commonly set `theme: Rochester`, a matching `colortheme`, `fontsize: 10pt`, and `\usepackage{emoji}` with `from: markdown+emoji` so emoji render in the PDF. Citations use a `.bib` file with `@cite-key` references (`BIEN-2025` additionally uses an ABNT numeric `.csl`).
- **`format: revealjs`** (HTML) — the Claude Code workshop decks: `Intro-to-ClaudeCode-CERES-2026`, `Intro-to-ClaudeCode-LABIIA-2026`, `AgentesIA-FMMAPE-2026`. These support:

  ```bash
  open <file>.html
  npx decktape reveal <file>.html <file>.pdf   # export to static PDF, requires Node.js
  ```

No `.qmd` file in this repo executes R/Python code chunks (the only fenced chunks are `{mermaid}` diagrams); any accompanying analysis code lives as a separate script/notebook alongside the deck rather than being knitted into the slides. Rendered outputs (`.pdf`/`.html`) are committed next to the source so viewers don't need Quarto installed.

## Structure per presentation directory

- One or more `.qmd` source files — multi-day workshops split them into `dia1.qmd`, `dia2.qmd`, etc., each rendering to its own HTML/PDF.
- `img/` for slide images.
- `referencias.bib` / `bibliography.bib` (+ optional `.csl`) when the talk uses citations.
- Non-slide analysis code lives beside the deck rather than in a separate repo, e.g. `code-and-data/script.R` (`ML-FMMAPE-2025`), `notebooks/*.ipynb` (`Minicurso-DL-Jornada-Discente-IESP-2025`), a standalone `.ipynb` (`SICSS-2025`).

## Claude Code workshop decks

`Intro-to-ClaudeCode-CERES-2026`, `Intro-to-ClaudeCode-LABIIA-2026`, and `AgentesIA-FMMAPE-2026` are revealjs decks about Claude Code itself, each for a different audience/venue but sharing most of their content — each has its own `CLAUDE.md` with rendering commands and notes on what's specific to that version; read that file when working inside any of them. They share the same setup: `custom.scss` for theming (Inter + JetBrains Mono loaded from Google Fonts), a title-slide badge/logo injected via an `include-in-header` `<script>` block in the YAML, and a gitignored institutional logo file + `.claude/` (local-only, not committed).

## Language

The presenter writes primarily in Portuguese (pt-BR); most slide content, file names, and commit messages are in Portuguese. Some materials (`SICSS-2025`) are in English for an international audience.
