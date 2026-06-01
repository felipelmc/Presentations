# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository context

This directory is part of a broader `Presentations` repository maintained by Felipe Lamarca. Each subdirectory holds materials for a specific talk or workshop. This folder (`Intro-to-ClaudeCode-LABIIA-2026`) is for an introductory presentation on Claude Code delivered at the Workshop CERES 2026.1 (IESP-UERJ).

Other presentations in the parent repo follow the same flat structure: slides and assets live together with no build system.

## Building and previewing slides

The course spans two days. Each day has its own source file:

| File | Output | Content |
|---|---|---|
| `dia1.qmd` | `dia1.html` | Setup, installation, IDE integrations, personal website demo |
| `dia2.qmd` | `dia2.html` | Skills, CEAP analysis, Câmara API demo, ethics |

```bash
# Render individual day
quarto render dia1.qmd
quarto render dia2.qmd

# Open in browser
open dia1.html
open dia2.html

# Export to PDF (requires Node.js)
npx decktape reveal dia1.html dia1.pdf
npx decktape reveal dia2.html dia2.pdf
```

The rendered HTML files are committed to the repo so attendees can access them without a Quarto installation.

### Screenshot placeholders

`dia1.qmd` contains placeholder divs for screenshots of the installation steps. Replace the dashed-border divs with actual `![](filename.png)` references before the event. Expected filenames are noted in comments within the source.

## Styling

Visual customization lives in `custom.scss`, which extends the default Quarto revealjs theme. Edit that file to change fonts, colors, or slide-level CSS. The presentation uses Inter (body) and JetBrains Mono (code), loaded from Google Fonts via the `include-in-header` block in the YAML front matter.

## Language

The presenter writes primarily in Portuguese (pt-BR). Commit messages and file names in this repo are often in Portuguese.
