#!/usr/bin/env bash
# Plenário Slides · cria a pasta de uma apresentação nova e a registra no palestras.yml.
#
#   bash _extensions/felipelmc/plenario/tools/nova-palestra.sh <Pasta-Evento-AAAA> <curso|academico> "Título" [AAAA-MM-DD] ["Evento"]
#
# Rode na raiz do projeto (a pasta que tem o palestras.yml).
set -euo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

pasta="$1"; estilo="$2"; titulo="$3"
data="${4:-$(date +%F)}"; evento="${5:-}"

[[ "$pasta" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]] || { echo "Nome de pasta inválido: use letras, números e hífens."; exit 1; }
[[ "$estilo" == "curso" || "$estilo" == "academico" ]] || { echo "Estilo deve ser curso ou academico."; exit 1; }
[ -e "$pasta" ] && { echo "A pasta $pasta já existe."; exit 1; }
[[ "$data" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "Data deve ser AAAA-MM-DD."; exit 1; }

mkdir -p "$pasta/img"
secao="curso"; bib=""
if [ "$estilo" = "academico" ]; then
  secao="pesquisa"
  bib="bibliography: referencias.bib"
  : > "$pasta/referencias.bib"
fi

{
  cat <<EOF
---
title: "$titulo"
subtitle: ""
author:
  - name: Felipe Lamarca
    affiliations:
      - name: IESP-UERJ
date: $data
format: plenario-revealjs
estilo: $estilo
evento: "$evento"
kicker: ""
footer: "Felipe Lamarca · $evento"
# logo-evento: logo.svg
# gerar-pdf: false
$bib
---

EOF
  if [ "$estilo" = "curso" ]; then
    cat <<'EOF'
## Roteiro {.escuro}

::: agenda
1. **Primeira parte:** do que se trata
2. **Segunda parte:** do que se trata
:::

# Primeira parte {.divisoria numero="01"}

## Um slide

Texto.

## Obrigado! {.encerramento}

::: contato
[Felipe Lamarca]{.nome}

[felipelamarca.com](https://felipelamarca.com)
:::
EOF
  else
    cat <<'EOF'
## A pergunta

Texto.

## Resultados {kicker="Passo 1"}

Texto.

## Referências {.scrollable}

::: {#refs}
:::

## Obrigado! {.encerramento}

::: contato
[Felipe Lamarca]{.nome}

[felipelamarca.com](https://felipelamarca.com)
:::
EOF
  fi
} > "$pasta/slides.qmd"

if [ -f palestras.yml ]; then
  pdf_linha="    pdf: slides.pdf"
  [ "$estilo" = "curso" ] && pdf_linha="    # pdf: slides.pdf   (ligue gerar-pdf: true no deck)"
  cat >> palestras.yml <<EOF

  - pasta: $pasta
    titulo: "$titulo"
    data: $data
    evento: "$evento"
    secao: $secao
    slides: slides.html
$pdf_linha
EOF
  echo "Entrada adicionada ao palestras.yml."
fi

echo "Pronto: $pasta/slides.qmd"
echo "Pré-visualizar: quarto preview $pasta/slides.qmd"
