#!/usr/bin/env python3
"""Plenário Slides · PDF automático dos decks (post-render do projeto).

Gera <deck>.pdf ao lado de cada <deck>.html marcado com <meta name="plenario-pdf">
(o plenario.lua marca os decks com `gerar-pdf: true`, o padrão do estilo academico).

  - serve a pasta de saída em 127.0.0.1 (OJS não carrega via file://);
  - roda o decktape (npx) com o mesmo tamanho do canvas: 1280×720 → 960×540 pt;
  - guarda PDFs em .quarto/plenario-pdf/ por hash do HTML + CSS + imagens, então um
    render completo (que limpa _site) não refaz PDFs de decks que não mudaram.

Variáveis de ambiente:
  PLENARIO_PDF=0   não gera nada;   PLENARIO_PDF=1   gera mesmo durante `quarto preview`.
  CHROME_PATH      navegador a usar (padrão: Chrome instalado, senão o do puppeteer).

Uso manual: python3 pdf.py [_site/pasta/slides.html ...]
"""

from __future__ import annotations

import functools
import hashlib
import html
import http.server
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

DECKTAPE = "decktape@3.16.1"
TAMANHO = "1280x720"
MARCA = re.compile(r'<meta\s+name="plenario-pdf"', re.I)


def log(msg: str) -> None:
    print(f"plenario-pdf: {msg}", flush=True)


def em_preview() -> bool:
    """`quarto preview` também roda o post-render; lá o PDF só atrasaria."""
    try:
        args = subprocess.run(["ps", "-o", "args=", "-p", str(os.getppid())],
                              capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return " preview" in args


def navegador() -> str | None:
    env = os.environ.get("CHROME_PATH")
    if env:
        return env
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(mac):
        return mac
    for nome in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        achado = shutil.which(nome)
        if achado:
            return achado
    return None


def locais(texto: str, base: Path, raiz: Path) -> list[Path]:
    """Arquivos locais citados pelo HTML que mudam a aparência do PDF (CSS e imagens)."""
    refs = re.findall(r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"', texto)
    refs += re.findall(r'<img[^>]+src="([^"]+)"', texto)
    refs += re.findall(r'data-background-image="([^"]+)"', texto)
    achados = []
    for ref in refs:
        if re.match(r"^(https?:|data:|//)", ref):
            continue
        ref = html.unescape(ref.split("#")[0].split("?")[0])
        caminho = (raiz / ref.lstrip("/")) if ref.startswith("/") else (base / ref)
        if caminho.is_file():
            achados.append(caminho.resolve())
    return sorted(set(achados))


def chave(deck: Path, texto: str, raiz: Path) -> str:
    h = hashlib.sha256()
    h.update(DECKTAPE.encode())
    h.update(TAMANHO.encode())
    h.update(texto.encode())
    for arq in locais(texto, deck.parent, raiz):
        h.update(str(arq.relative_to(raiz) if arq.is_relative_to(raiz) else arq).encode())
        h.update(arq.read_bytes())
    return h.hexdigest()[:24]


class Silencioso(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401
        pass


def servir(raiz: Path):
    handler = functools.partial(Silencioso, directory=str(raiz))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def metadados(texto: str) -> tuple[str, str]:
    titulo = re.search(r"<title>(.*?)</title>", texto, re.S)
    autor = re.search(r'<meta\s+name="author"\s+content="([^"]*)"', texto)
    t = html.unescape(titulo.group(1)).strip() if titulo else "Slides"
    t = re.sub(r"^.*? – ", "", t)  # tira o prefixo "Site – " dos projetos website
    return t, html.unescape(autor.group(1)) if autor else ""


def paginas(pdf: Path) -> int | str:
    if shutil.which("pdfinfo"):
        r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
        if m:
            return int(m.group(1))
    try:
        from pypdf import PdfReader  # opcional
        return len(PdfReader(str(pdf)).pages)
    except Exception:
        return "?"


def gerar(deck: Path, raiz: Path, porta: int, chrome: str | None, cache: Path) -> bool:
    texto = deck.read_text(encoding="utf-8", errors="replace")
    pdf = deck.with_suffix(".pdf")
    k = chave(deck, texto, raiz)
    guardado = cache / f"{k}.pdf"
    rel = deck.relative_to(raiz).as_posix()
    if guardado.exists():
        shutil.copyfile(guardado, pdf)
        log(f"{rel} → {pdf.name} (cache, {paginas(pdf)} páginas)")
        return True

    titulo, autor = metadados(texto)
    url = f"http://127.0.0.1:{porta}/{rel}"
    cmd = ["npx", "-y", DECKTAPE, "reveal", "--size", TAMANHO,
           "--pause", "500", "--load-pause", "1500",
           "--pdf-title", titulo]
    if autor:
        cmd += ["--pdf-author", autor]
    if chrome:
        cmd += ["--chrome-path", chrome]
    if os.environ.get("CI"):
        cmd += ["--chrome-arg=--no-sandbox", "--chrome-arg=--disable-dev-shm-usage"]
    cmd += [url, str(pdf)]
    env = dict(os.environ, PUPPETEER_SKIP_DOWNLOAD="true") if chrome else dict(os.environ)

    for tentativa in (1, 2):
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode == 0 and pdf.exists() and pdf.stat().st_size > 0:
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, guardado)
            log(f"{rel} → {pdf.name} ({paginas(pdf)} páginas)")
            return True
        log(f"{rel}: tentativa {tentativa} falhou\n{(r.stdout + r.stderr)[-1500:]}")
    return False


def main(argv: list[str]) -> int:
    flag = os.environ.get("PLENARIO_PDF")
    if flag == "0":
        log("PLENARIO_PDF=0, pulando")
        return 0
    if flag != "1" and em_preview():
        return 0

    projeto = Path(os.environ.get("QUARTO_PROJECT_DIR", os.getcwd())).resolve()
    saida = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
    raiz = (projeto / saida).resolve() if saida else None

    if argv:
        candidatos = [Path(a).resolve() for a in argv]
        raiz = raiz or Path(os.path.commonpath([c.parent for c in candidatos]))
    else:
        if raiz is None:
            log("sem QUARTO_PROJECT_OUTPUT_DIR; passe os HTML como argumentos")
            return 0
        lista = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "").strip()
        if lista:
            candidatos = [(projeto / f).resolve() for f in lista.splitlines() if f.endswith(".html")]
        else:
            candidatos = sorted(raiz.rglob("*.html"))

    decks = [c for c in candidatos
             if c.is_file() and MARCA.search(c.read_text(encoding="utf-8", errors="replace")[:20000])]
    if not decks:
        return 0

    chrome = navegador()
    cache = projeto / ".quarto" / "plenario-pdf"
    srv = servir(raiz)
    try:
        falhas = [d for d in decks if not gerar(d, raiz, srv.server_address[1], chrome, cache)]
    finally:
        srv.shutdown()
    if falhas:
        log("falhou: " + ", ".join(str(f.relative_to(raiz)) for f in falhas))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
