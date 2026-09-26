#!/usr/bin/env python3
"""Plenário Slides · galeria de apresentações (post-render do projeto, depois do pdf.py).

Lê `palestras.yml` na raiz do projeto e escreve `<saída>/index.html`, uma página com o visual de
felipelamarca.com: cabeçalho, títulos de seção numerados, cards com a capa de cada deck e rodapé.

  - a capa de cada card é a primeira página do PDF (pdftoppm), salva como <pasta>/capa.png;
  - num render completo, falha no CI (variável CI) se um deck renderizado não estiver no
    palestras.yml ou se um arquivo citado não existir; fora do CI, só avisa.

Formato do palestras.yml: veja o README do Slides-Template.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("galeria: instale o pyyaml (pip install pyyaml)")

EXT = Path(__file__).resolve().parent.parent
RAIZ = Path(os.environ.get("QUARTO_PROJECT_DIR", os.getcwd())).resolve()
MESES = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto",
         "Setembro", "Outubro", "Novembro", "Dezembro"]

PADRAO_SITE = {
    "titulo": "Slides",
    "eyebrow": "Apresentações",
    "subtitulo": "",
    "titulo-pagina": "Slides · Felipe Lamarca",
    "descricao": "Slides de palestras, aulas e workshops.",
    "url": "",
    "repo": "",
    "nome": "Felipe Lamarca",
    "cargo": "Cientista Político Computacional",
    "casa": "https://felipelamarca.com/pt-br/",
    "navegacao": [
        {"texto": "Projetos", "href": "https://felipelamarca.com/pt-br/projects"},
        {"texto": "Pesquisa", "href": "https://felipelamarca.com/pt-br/publications"},
        {"texto": "Ensino", "href": "https://felipelamarca.com/pt-br/teaching"},
        {"texto": "CV", "href": "https://felipelamarca.com/pt-br/cv"},
    ],
    "redes": [
        {"id": "github", "rotulo": "GitHub", "href": "https://github.com/felipelmc"},
        {"id": "scholar", "rotulo": "Google Scholar", "href": "https://scholar.google.com.br/citations?user=xPf8_64AAAAJ&hl=pt-BR&oi=ao"},
        {"id": "linkedin", "rotulo": "LinkedIn", "href": "https://www.linkedin.com/in/felipe-lamarca/"},
        {"id": "orcid", "rotulo": "ORCID", "href": "https://orcid.org/0000-0003-0002-3627"},
        {"id": "lattes", "rotulo": "Lattes", "href": "http://lattes.cnpq.br/2606938112682925"},
        {"id": "email", "rotulo": "Email", "href": "mailto:felipe.lamarca@hotmail.com"},
    ],
    "colofao": 'Feito com Quarto e o <a href="https://felipelamarca.com/Slides-Template/">Plenário Slides</a>.',
}

# Ícones de src/components/ui/Icon.astro (grade 24×24).
ICONES = {
    "github": (True, "M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"),
    "linkedin": (True, "M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"),
    "scholar": (True, "M5.242 13.769L0 9.5L12 0l12 9.5l-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 1 0 0 14 7 7 0 0 0 0-14z"),
    "orcid": (True, "M12 0C5.372 0 0 5.372 0 12s5.372 12 12 12 12-5.372 12-12S18.628 0 12 0zM7.369 4.378c.525 0 .947.431.947.947s-.422.947-.947.947a.95.95 0 0 1-.947-.947c0-.525.422-.947.947-.947zm-.722 3.038h1.444v10.041H6.647V7.416zm3.562 0h3.9c3.712 0 5.344 2.653 5.344 5.025 0 2.578-2.016 5.025-5.325 5.025h-3.919V7.416zm1.444 1.303v7.444h2.297c3.272 0 4.022-2.484 4.022-3.722 0-2.016-1.284-3.722-4.097-3.722h-2.222z"),
    "lattes": (True, "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"),
    "email": (False, "M3 7l8.2 5.47a1.5 1.5 0 0 0 1.6 0L21 7M5 19h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2z"),
    "arrow-up-right": (False, "M7 17L17 7M9 7h8v8"),
    "arrow-right": (False, "M5 12h14M13 6l6 6-6 6"),
    "download": (False, "M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19.5h14"),
    "sun": (False, "M12 3v1.5m0 15V21m9-9h-1.5M4.5 12H3m15.36 6.36l-1.06-1.06M6.7 6.7L5.64 5.64m12.72 0L17.3 6.7M6.7 17.3l-1.06 1.06M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0z"),
    "moon": (False, "M20.35 15.35A9 9 0 0 1 8.65 3.65 9 9 0 1 0 20.35 15.35z"),
    "menu": (False, "M4 7h16M4 12h16M4 17h16"),
}


def icone(nome: str, classe: str = "") -> str:
    cheio, d = ICONES[nome]
    pintura = 'fill="currentColor" stroke="none"' if cheio else \
        'fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"'
    extra = ' fill-rule="evenodd" clip-rule="evenodd"' if nome == "github" else ""
    c = f' class="{classe}"' if classe else ""
    return f'<svg{c} viewBox="0 0 24 24" {pintura} aria-hidden="true" focusable="false"><path d="{d}"{extra}/></svg>'


def marca() -> str:
    pontos = []
    for r in range(3):
        for c in range(3):
            pontos.append(f'<circle cx="{4 + c * 8}" cy="{4 + r * 8}" r="3.3" style="fill: rgb(var(--d{r + c + 1}))"/>')
    return f'<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">{"".join(pontos)}</svg>'


def e(texto) -> str:
    return html.escape(str(texto), quote=True)


def data_rotulo(valor) -> str:
    if isinstance(valor, str):
        valor = dt.date.fromisoformat(valor[:10])
    return f"{MESES[valor.month - 1]} {valor.year}"


def data_iso(valor) -> str:
    return valor[:10] if isinstance(valor, str) else valor.isoformat()


def log(msg: str) -> None:
    print(f"galeria: {msg}", flush=True)


def capa(pdf: Path, destino: Path) -> bool:
    if not pdf.exists() or not shutil.which("pdftoppm"):
        return False
    if destino.exists() and destino.stat().st_mtime >= pdf.stat().st_mtime:
        return True
    base = destino.with_suffix("")
    r = subprocess.run(["pdftoppm", "-png", "-f", "1", "-l", "1", "-singlefile", "-scale-to", "960",
                        str(pdf), str(base)], capture_output=True, text=True)
    return r.returncode == 0 and destino.exists()


def proporcao(png: Path) -> float | None:
    try:
        dados = png.read_bytes()[16:24]
        w, h = int.from_bytes(dados[:4], "big"), int.from_bytes(dados[4:], "big")
        return w / h if h else None
    except Exception:
        return None


def card(p: dict, site: dict, saida: Path, problemas: list[str]) -> str:
    pasta = p["pasta"]
    base = saida / pasta
    partes = p.get("partes") or [{"rotulo": None, "slides": p.get("slides"), "pdf": p.get("pdf")}]
    for parte in partes:
        for chave in ("slides", "pdf"):
            alvo = parte.get(chave)
            if alvo and not (base / alvo).exists():
                problemas.append(f"{pasta}/{alvo} não existe na saída")

    principal = next((x for x in partes if x.get("slides")), None)
    href_principal = f"{pasta}/{principal['slides']}" if principal else \
        (f"{pasta}/{partes[0]['pdf']}" if partes[0].get("pdf") else "#")

    pdf_capa = next((base / x["pdf"] for x in partes if x.get("pdf")), None)
    img = ""
    if p.get("capa"):  # imagem explícita, relativa à pasta (decks sem PDF, como os de webR)
        origem = RAIZ / pasta / p["capa"]
        tem_capa = origem.exists()
        if tem_capa:
            shutil.copyfile(origem, base / "capa.png")
        else:
            problemas.append(f"{pasta}/{p['capa']} (capa) não existe")
    else:
        tem_capa = pdf_capa is not None and capa(pdf_capa, base / "capa.png")
    if tem_capa:
        prop = proporcao(base / "capa.png") or 16 / 9
        classe = "capa" + (" quatro-tercos" if prop < 1.6 else "")
        img = (f'<a class="{classe}" href="{e(href_principal)}" tabindex="-1" aria-hidden="true">'
               f'<img src="{e(pasta)}/capa.png" alt="" loading="lazy" decoding="async" width="960" height="{round(960 / prop)}"></a>')
    else:
        img = f'<a class="capa" href="{e(href_principal)}" tabindex="-1" aria-hidden="true"><span class="sem-capa">{marca()}</span></a>'

    dominio = (site.get("url") or "").split("://")[-1].rstrip("/") + f"/{pasta}/"
    meta = [data_rotulo(p["data"])]
    if p.get("detalhe"):
        meta.append(p["detalhe"])

    chips = []
    formato = p.get("formato", "revealjs")
    chips.append('<li class="chip">Beamer</li>' if formato == "beamer" else '<li class="chip">revealjs</li>')
    if any(x.get("pdf") for x in partes):
        chips.append('<li class="chip">PDF</li>')
    if len(partes) > 1:
        chips.append(f'<li class="chip">{len(partes)} partes</li>')
    if p.get("idioma") and p["idioma"] not in ("pt", "pt-BR"):
        chips.append(f'<li class="chip">{e(p["idioma"])}</li>')
    if p.get("interativo"):
        chips.append('<li class="chip acento">interativo</li>')

    acoes = []
    for parte in partes:
        rotulo = parte.get("rotulo")
        if parte.get("slides"):
            texto = f"Slides, {rotulo.lower()}" if rotulo else "Slides"
            acoes.append(f'<a class="acao" href="{e(pasta)}/{e(parte["slides"])}">{e(texto)}{icone("arrow-right")}</a>')
        if parte.get("pdf"):
            texto = f"PDF, {rotulo.lower()}" if rotulo else "PDF"
            acoes.append(f'<a class="acao" href="{e(pasta)}/{e(parte["pdf"])}">{e(texto)}{icone("download")}</a>')
    for extra in p.get("extras") or []:
        acoes.append(f'<a class="acao" href="{e(extra["href"])}" target="_blank" rel="noopener noreferrer">'
                     f'{e(extra["rotulo"])}{icone("arrow-up-right")}</a>')
    if site.get("repo"):
        acoes.append(f'<a class="acao" href="{e(site["repo"])}/tree/main/{e(pasta)}" target="_blank" '
                     f'rel="noopener noreferrer">Arquivos{icone("arrow-up-right")}</a>')

    subtitulo = f'<p class="subtitulo">{e(p["subtitulo"])}</p>' if p.get("subtitulo") else ""
    evento = f'<p class="evento">{e(p["evento"])}</p>' if p.get("evento") else ""
    lang = f' lang="{e(p["idioma"])}"' if p.get("idioma") and p["idioma"] not in ("pt", "pt-BR") else ""

    return f"""
        <article class="card" id="{e(pasta)}">
          <div class="barra" aria-hidden="true"><span class="ponto"></span><span class="dominio">{e(dominio)}</span></div>
          {img}
          <div class="corpo">
            <p class="meta"><time datetime="{data_iso(p['data'])}">{e(" · ".join(meta))}</time></p>
            <h3{lang}><a href="{e(href_principal)}">{e(p["titulo"])}</a></h3>
            {subtitulo}
            {evento}
            <div class="rodape">
              <ul class="chips">{"".join(chips)}</ul>
              <div class="acoes">{"".join(acoes)}</div>
            </div>
          </div>
        </article>"""


def pagina(cfg: dict, saida: Path, problemas: list[str]) -> str:
    site = {**PADRAO_SITE, **(cfg.get("site") or {})}
    secoes = cfg.get("secoes") or [{"id": "pesquisa", "titulo": "Pesquisa"},
                                   {"id": "curso", "titulo": "Cursos e workshops"}]
    palestras = sorted(cfg.get("palestras") or [], key=lambda p: data_iso(p["data"]), reverse=True)

    blocos, indice = [], []
    n = 0
    for s in secoes:
        itens = [p for p in palestras if p.get("secao", "curso") == s["id"]]
        if not itens:
            continue
        n += 1
        num = f"{n:02d}"
        indice.append(f'<li><a href="#{e(s["id"])}" data-indice="{e(s["id"])}"><span aria-hidden="true">{num}</span>{e(s["titulo"])}</a></li>')
        cards = "".join(card(p, site, saida, problemas) for p in itens)
        blocos.append(f"""
      <section aria-labelledby="{e(s['id'])}">
        <div class="titulo-secao">
          <span class="num" aria-hidden="true">{num}</span>
          <h2 id="{e(s['id'])}">{e(s['titulo'])}</h2>
          <span class="linha" aria-hidden="true"></span>
          <span class="conta">{len(itens)}</span>
        </div>
        <div class="cards">{cards}
        </div>
      </section>""")

    nav = "".join(f'<li><a href="{e(i["href"])}">{e(i["texto"])}</a></li>' for i in site["navegacao"])
    nav_movel = "".join(f'<li><a href="{e(i["href"])}">{e(i["texto"])}{icone("arrow-right")}</a></li>'
                        for i in site["navegacao"])
    redes = "".join(
        f'<li><a class="icone-btn" href="{e(r["href"])}" aria-label="{e(r["rotulo"])}" title="{e(r["rotulo"])}"'
        + ("" if r["id"] == "email" else ' target="_blank" rel="noopener noreferrer"')
        + f'>{icone(r["id"])}</a></li>'
        for r in site["redes"] if r["id"] in ICONES)
    nav_rodape = "".join(f'<li><a href="{e(i["href"])}">{e(i["texto"])}</a></li>' for i in site["navegacao"])
    ano = max([int(data_iso(p["data"])[:4]) for p in palestras] or [dt.date.today().year])
    subtitulo = f'<p class="subtitulo">{site["subtitulo"]}</p>' if site.get("subtitulo") else ""
    canonica = f'<link rel="canonical" href="{e(site["url"])}">' if site.get("url") else ""

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(site["titulo-pagina"])}</title>
  <meta name="description" content="{e(site["descricao"])}">
  <meta name="theme-color" content="#F6F6F3">
  {canonica}
  <link rel="icon" type="image/svg+xml" href="_galeria/favicon.svg">
  <link rel="stylesheet" href="_galeria/fonts/fonts.css">
  <link rel="stylesheet" href="_galeria/galeria.css">
  <script>
    (function () {{
      var tema;
      try {{ tema = localStorage.getItem("theme"); }} catch (err) {{}}
      tema = tema || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      document.documentElement.classList.toggle("dark", tema === "dark");
      var m = document.querySelector('meta[name="theme-color"]');
      if (m) m.setAttribute("content", tema === "dark" ? "#0B0D10" : "#F6F6F3");
    }})();
  </script>
</head>
<body>
  <a class="pular" href="#main">Pular para o conteúdo</a>
  <header class="topo">
    <nav class="container" aria-label="Principal">
      <a class="casa" href="{e(site["casa"])}" aria-label="{e(site["nome"])} · início">{marca()}<span>{e(site["nome"])}</span></a>
      <div class="direita">
        <ul class="links">{nav}</ul>
        <button id="tema" class="icone-btn tema" type="button" aria-label="Alternar modo escuro" aria-pressed="false">{icone("sun", "sol")}{icone("moon", "lua")}</button>
        <button id="menu-toggle" class="icone-btn" type="button" aria-label="Menu" aria-expanded="false" aria-controls="menu-movel">{icone("menu")}</button>
      </div>
    </nav>
    <div id="menu-movel"><ul class="container">{nav_movel}</ul></div>
  </header>

  <main id="main" class="container">
    <header class="cabecalho-pagina">
      <p class="eyebrow">{e(site["eyebrow"])}</p>
      <h1>{e(site["titulo"])}</h1>
      {subtitulo}
    </header>
    <div class="grade-pagina">
      <nav class="indice" aria-label="Seções"><ol>{"".join(indice)}</ol></nav>
      <div class="secoes">{"".join(blocos)}
      </div>
    </div>
  </main>

  <footer class="rodape-site">
    <div class="container">
      <div>
        <p class="quem">{marca()}<span class="nome">{e(site["nome"])}</span><span class="cargo">{e(site["cargo"])}</span></p>
        <ul class="navegacao">{nav_rodape}</ul>
      </div>
      <ul class="redes">{redes}</ul>
      <p class="meta colofao">© {ano} {e(site["nome"])} · {site["colofao"]}</p>
    </div>
  </footer>

  <script>
    (function () {{
      var html = document.documentElement, botao = document.getElementById("tema");
      function sincroniza() {{
        var escuro = html.classList.contains("dark");
        botao.setAttribute("aria-pressed", String(escuro));
        var m = document.querySelector('meta[name="theme-color"]');
        if (m) m.setAttribute("content", escuro ? "#0B0D10" : "#F6F6F3");
      }}
      sincroniza();
      botao.addEventListener("click", function () {{
        var escuro = html.classList.toggle("dark");
        try {{ localStorage.setItem("theme", escuro ? "dark" : "light"); }} catch (err) {{}}
        sincroniza();
      }});
      window.addEventListener("storage", function (ev) {{
        if (ev.key === "theme") {{ html.classList.toggle("dark", ev.newValue === "dark"); sincroniza(); }}
      }});
      var menu = document.getElementById("menu-movel"), toggle = document.getElementById("menu-toggle");
      toggle.addEventListener("click", function () {{
        var aberto = menu.classList.toggle("aberto");
        toggle.setAttribute("aria-expanded", String(aberto));
      }});
      var links = Array.prototype.slice.call(document.querySelectorAll("[data-indice]"));
      var alvos = links.map(function (a) {{ return document.getElementById(a.getAttribute("data-indice")); }});
      function marca() {{
        var linha = window.innerHeight * 0.33, atual = alvos[0] && alvos[0].id;
        alvos.forEach(function (el) {{ if (el && el.getBoundingClientRect().top <= linha) atual = el.id; }});
        links.forEach(function (a) {{ a.setAttribute("aria-current", String(a.getAttribute("data-indice") === atual)); }});
      }}
      if (links.length) {{ window.addEventListener("scroll", marca, {{ passive: true }}); marca(); }}
    }})();
  </script>
</body>
</html>
"""


def main() -> int:
    projeto = Path(os.environ.get("QUARTO_PROJECT_DIR", os.getcwd())).resolve()
    saida_env = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
    saida = (projeto / saida_env).resolve() if saida_env else projeto / "_site"
    arquivo = projeto / "palestras.yml"
    if not arquivo.exists():
        log("sem palestras.yml na raiz do projeto; nada a fazer")
        return 0
    cfg = yaml.safe_load(arquivo.read_text(encoding="utf-8")) or {}

    # Recursos da página: CSS, fontes e favicon da extensão.
    destino = saida / "_galeria"
    (destino / "fonts").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXT / "galeria" / "galeria.css", destino / "galeria.css")
    shutil.copyfile(EXT / "assets" / "favicon.svg", destino / "favicon.svg")
    for f in (EXT / "fonts").iterdir():
        if f.suffix in (".woff2", ".css"):
            shutil.copyfile(f, destino / "fonts" / f.name)

    problemas: list[str] = []
    (saida / "index.html").write_text(pagina(cfg, saida, problemas), encoding="utf-8")

    # Completude: todo deck renderizado precisa estar na galeria.
    citados = set()
    for p in cfg.get("palestras") or []:
        for parte in p.get("partes") or [p]:
            if parte.get("slides"):
                citados.add(f'{p["pasta"]}/{parte["slides"]}')
    for deck in sorted(saida.glob("*/*.html")):
        rel = deck.relative_to(saida).as_posix()
        if rel.startswith(("_", "site_libs/")):
            continue
        if rel not in citados:
            problemas.append(f"{rel} foi renderizado mas não está no palestras.yml")

    n = len(cfg.get("palestras") or [])
    log(f"index.html com {n} apresentações")
    if problemas:
        for pr in problemas:
            log("⚠ " + pr)
        if os.environ.get("CI") and os.environ.get("QUARTO_PROJECT_RENDER_ALL"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
