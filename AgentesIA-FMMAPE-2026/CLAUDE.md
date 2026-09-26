# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository context

This directory is part of a broader `Presentations` repository maintained by Felipe Lamarca. Each subdirectory holds materials for a specific talk or workshop. This folder (`AgentesIA-FMMAPE-2026`) is for an introductory presentation on Claude Code delivered at the Formação Metodológica do MAPE (Laboratório de Monitoramento e Avaliação de Políticas e Eleições, IESP-UERJ) -- a single-day, ~2h session.

This deck is derived from `Intro-to-ClaudeCode-CERES-2026` (also single-day/2h) but folds in the content improvements made for the two-day `Intro-to-ClaudeCode-LABIIA-2026` version: the installation flow where Claude installs itself, the IDE-integration overview (terminal/VSCode/Positron), the "other AI agents" comparison table, the concrete CLAUDE.md example, the `.claude` folder diagram, plan mode, and the expanded "what journals say" section on AI-use policy. The three live demos were swapped for research-relevant ones: literature fichamento via the OpenAlex API, a Câmara dos Deputados API panel, and the personal-website-from-résumé demo carried over from CERES.

## Rendering and publishing

The deck uses `format: plenario-revealjs` with `estilo: curso` (Plenário Slides, see the root `CLAUDE.md`). From the repository root:

```bash
quarto preview AgentesIA-FMMAPE-2026/slides.qmd
quarto render AgentesIA-FMMAPE-2026/slides.qmd   # HTML + slides.pdf in _site/
```

CI publishes it at `felipelamarca.com/Presentations/AgentesIA-FMMAPE-2026/slides.html`, with `slides.pdf` next to it. Neither file is committed.

- Title badge: `evento: "Formação Metodológica do MAPE"` + `logo-mape.svg` (now committed). There is no "Feito com Claude Code" credit on this deck, dropped on request.
- In September 2026 the deck moved from `custom.scss` + inline HTML to the Plenário Slides components, with the text unchanged. The four blank slides, caused by `---` right before a `#` heading, are gone: 42 slides instead of 46.

## Assets

Screenshots under `prints/` are mostly reused as-is from `Intro-to-ClaudeCode-LABIIA-2026/dia1_prints` and `dia2_prints` (terminal, VSCode, Positron, usage dashboard, plan mode). `claudecode_app.png` is new to this deck -- a cropped screenshot of the Claude desktop app's "Code" tab, used as the fourth card on "Integrações no dia a dia" alongside terminal/VSCode/Positron. It shows the presenter's real project sidebar (project names like `CEBRAP-Szwako`, `MAPEmunicipios-ETL`) and a "near weekly limit" banner -- worth a quick look before presenting in case any project name shouldn't be shown to that audience.

The closing slide has no LinkedIn QR code (dropped on request), unlike CERES and LABIIA.

## Code samples

Code blocks favor R (tidyverse/ggplot2, `httr2` for the Câmara dos Deputados API demo) over the Python used in the CERES/LABIIA versions, since the audience works mostly in R. The one exception is the OpenAlex demo, kept in Python -- OpenAlex's R wrappers are less mature than calling the REST API directly.

## Language

The presenter writes primarily in Portuguese (pt-BR). Commit messages and file names in this repo are often in Portuguese.
