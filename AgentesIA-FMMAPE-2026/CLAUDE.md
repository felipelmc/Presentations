# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository context

This directory is part of a broader `Presentations` repository maintained by Felipe Lamarca. Each subdirectory holds materials for a specific talk or workshop. This folder (`AgentesIA-FMMAPE-2026`) is for an introductory presentation on Claude Code delivered at the Formação Metodológica do MAPE (Laboratório de Monitoramento e Avaliação de Políticas e Eleições, IESP-UERJ) -- a single-day, ~2h session.

This deck is derived from `Intro-to-ClaudeCode-CERES-2026` (also single-day/2h) but folds in the content improvements made for the two-day `Intro-to-ClaudeCode-LABIIA-2026` version: the installation flow where Claude installs itself, the IDE-integration overview (terminal/VSCode/Positron), the "other AI agents" comparison table, the concrete CLAUDE.md example, the `.claude` folder diagram, plan mode, and the expanded "what journals say" section on AI-use policy. The three live demos were swapped for research-relevant ones: literature fichamento via the OpenAlex API, a Câmara dos Deputados API panel, and the personal-website-from-résumé demo carried over from CERES.

## Building and previewing slides

```bash
quarto render slides.qmd

# Open in browser
open slides.html

# Export to PDF (requires Node.js)
npx decktape reveal slides.html slides.pdf
```

The rendered `slides.html` is committed to the repo so attendees can access it without a Quarto installation.

### Pending before the session

- The root `README.md` talk listing still needs an entry for this course (date: 2026-08-17).
- `logo-mape.svg` (the lab's monogram mark) and `.claude/` are gitignored, matching the convention already used in `Intro-to-ClaudeCode-CERES-2026` and `Intro-to-ClaudeCode-LABIIA-2026` -- they exist locally but are not pushed. If cloning this repo fresh, re-fetch the logo before rendering: it comes from `mape-iesp/mape-iesp.github.io`'s `src/assets/favicons/favicon.svg`.
- Unlike CERES/LABIIA, the title slide here has no "Feito com Claude Code" credit line -- dropped on request.

## Styling

Visual customization lives in `custom.scss`, identical to the CERES/LABIIA decks (Inter for body text, JetBrains Mono for code, loaded from Google Fonts via the `include-in-header` block in the YAML). Do not fork the stylesheet without a reason -- keep the three decks visually consistent.

## Assets

Screenshots under `prints/` are mostly reused as-is from `Intro-to-ClaudeCode-LABIIA-2026/dia1_prints` and `dia2_prints` (terminal, VSCode, Positron, usage dashboard, plan mode). `claudecode_app.png` is new to this deck -- a cropped screenshot of the Claude desktop app's "Code" tab, used as the fourth card on "Integrações no dia a dia" alongside terminal/VSCode/Positron. It shows the presenter's real project sidebar (project names like `CEBRAP-Szwako`, `MAPEmunicipios-ETL`) and a "near weekly limit" banner -- worth a quick look before presenting in case any project name shouldn't be shown to that audience.

Unlike the CERES/LABIIA decks, the closing slide here has no LinkedIn QR code -- it was dropped on request, so there's no `linkedin-qr.png` in this folder.

## Code samples

Code blocks favor R (tidyverse/ggplot2, `httr2` for the Câmara dos Deputados API demo) over the Python used in the CERES/LABIIA versions, since the audience works mostly in R. The one exception is the OpenAlex demo, kept in Python -- OpenAlex's R wrappers are less mature than calling the REST API directly.

## Language

The presenter writes primarily in Portuguese (pt-BR). Commit messages and file names in this repo are often in Portuguese.
