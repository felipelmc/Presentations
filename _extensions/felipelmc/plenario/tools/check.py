#!/usr/bin/env python3
"""Plenário Slides · checagens antes e depois do render.

  python3 check.py pre               # fontes: decks que o projeto renderiza
  python3 check.py post _site [log]  # saída: citações, links locais, PDFs e log do render

Falha (saída 1) com a lista de problemas encontrados.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path.cwd()
PADROES = ("*/slides.qmd", "*/dia*.qmd")


def decks() -> list[Path]:
    achados: set[Path] = set()
    for p in PADROES:
        achados.update(RAIZ.glob(p))
    return sorted(achados)


def front_matter(texto: str) -> tuple[str, str]:
    m = re.match(r"^---\n(.*?)\n---[ \t]*\n", texto, re.S)
    return (m.group(1), texto[m.end():]) if m else ("", texto)


def pre() -> list[str]:
    problemas = []
    webr_instalado = (RAIZ / "_extensions" / "coatless" / "webr").exists()
    for deck in decks():
        rel = deck.relative_to(RAIZ)
        texto = deck.read_text(encoding="utf-8")
        yaml, corpo = front_matter(texto)
        if "format: plenario-revealjs" not in yaml:
            problemas.append(f"{rel}: use `format: plenario-revealjs`")
        if re.search(r'\sstyle="', corpo):
            problemas.append(f"{rel}: há `style=` inline; use os componentes do tema")
        if re.search(r"background-color=", corpo):
            problemas.append(f"{rel}: fundo fixo; use {{.divisoria}}, {{.escuro}} ou {{.encerramento}}")
        if re.search(r"^---\s*\n\s*\n?#\s", corpo, re.M):
            problemas.append(f"{rel}: `---` antes de `#` cria um slide em branco")
        if "```{webr-r}" in corpo:
            if not webr_instalado:
                problemas.append(f"{rel}: usa webR, mas _extensions/coatless/webr não está instalada "
                                 "(quarto add coatless/quarto-webr@0.4.3 --no-prompt)")
            if not re.search(r"^engine:", yaml, re.M):
                problemas.append(f"{rel}: decks com webR precisam de `engine: markdown`")
        for img in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", corpo):
            if re.match(r"^(https?:|data:)", img):
                continue
            if not (deck.parent / img).exists():
                problemas.append(f"{rel}: imagem não encontrada: {img}")
    return problemas


def post(saida: Path, log: Path | None) -> list[str]:
    problemas = []
    for pagina in sorted(saida.glob("*/*.html")):
        rel = pagina.relative_to(saida)
        if rel.parts[0].startswith("_") or rel.parts[0] == "site_libs":
            continue
        html = pagina.read_text(encoding="utf-8", errors="replace")
        for chave in sorted(set(re.findall(r"\?@([\w:.-]+)", html))):
            problemas.append(f"{rel}: citação não resolvida @{chave}")
        for ref in re.findall(r'(?:src|href)="([^"#?]+)"', html):
            if re.match(r"^(https?:|mailto:|data:|javascript:|//)", ref) or ref.startswith("/"):
                continue
            if not (pagina.parent / ref).exists():
                problemas.append(f"{rel}: link local quebrado: {ref}")
        if '<meta name="plenario-pdf"' in html and not pagina.with_suffix(".pdf").exists():
            problemas.append(f"{rel}: marcado para PDF, mas {pagina.with_suffix('.pdf').name} não existe")
    if log and log.exists():
        texto = log.read_text(encoding="utf-8", errors="replace")
        for linha in texto.splitlines():
            if re.search(r"citation .* not found|Unable to resolve crossref|WARN.*(not found|missing)", linha, re.I):
                problemas.append(f"log: {linha.strip()}")
    return problemas


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("pre", "post"):
        print(__doc__)
        return 2
    if argv[0] == "pre":
        problemas = pre()
    else:
        saida = Path(argv[1]) if len(argv) > 1 else RAIZ / "_site"
        log = Path(argv[2]) if len(argv) > 2 else None
        problemas = post(saida, log)
    for p in problemas:
        print(f"✗ {p}")
    print(f"check {argv[0]}: {'ok' if not problemas else f'{len(problemas)} problema(s)'}")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
