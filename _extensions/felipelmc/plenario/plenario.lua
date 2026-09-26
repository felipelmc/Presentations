-- Plenário Slides · filtro principal (roda em pre-ast).
--
--   * carrega as fontes e o CSS do estilo (estilo: curso | academico);
--   * pinta o fundo dos slides com classe .divisoria, .escuro e .encerramento;
--   * marca o deck para o PDF automático (gerar-pdf) via <meta name="plenario-pdf">;
--   * transforma [~55% da janela]{.medidor uso=55} numa barra de uso.

local ESTILOS = {
  curso = { divisoria = "#0A6F69", escuro = "#0E1116", encerramento = "#0E1116" },
  academico = { escuro = "#0E1116", encerramento = "#0E1116" },
}

local estilo = "academico"

local function texto(v)
  if v == nil then return nil end
  return pandoc.utils.stringify(v)
end

local function Meta(meta)
  if not quarto.doc.is_format("html") then return nil end

  quarto.doc.add_html_dependency({
    name = "plenario-fontes",
    version = "1.0.0",
    stylesheets = { "fonts/fonts.css" },
    resources = {
      "fonts/geist-latin-wght-normal.woff2",
      "fonts/geist-latin-ext-wght-normal.woff2",
      "fonts/geist-mono-latin-wght-normal.woff2",
      "fonts/geist-mono-latin-ext-wght-normal.woff2",
      "fonts/newsreader-latin-opsz-normal.woff2",
      "fonts/newsreader-latin-ext-opsz-normal.woff2",
      "fonts/newsreader-latin-opsz-italic.woff2",
      "fonts/newsreader-latin-ext-opsz-italic.woff2",
    },
  })

  if not quarto.doc.is_format("revealjs") then return meta end

  estilo = texto(meta.estilo) or "academico"
  if not ESTILOS[estilo] then
    quarto.log.warning("plenario: estilo '" .. estilo .. "' desconhecido; use curso ou academico. Usando academico.")
    estilo = "academico"
  end
  quarto.doc.add_html_dependency({
    name = "plenario-" .. estilo,
    version = "0.1.0",
    stylesheets = { "estilos/" .. estilo .. ".css" },
  })
  meta["estilo-" .. estilo] = true

  -- O mermaid do Quarto mede os rótulos dentro do slide, que o reveal reduz com
  -- transform: scale(); a medida sai menor que o texto e os rótulos estouram as caixas
  -- (bem visível no decktape). Sem o contêiner, o mermaid mede num <div> solto no <body>.
  quarto.doc.include_text("in-header", [[
<script>
(function () {
  function corrige() {
    if (!window.mermaid || !mermaid.mermaidAPI || mermaid.mermaidAPI.plenario) return;
    var api = mermaid.mermaidAPI;
    mermaid.mermaidAPI = Object.assign({}, api, {
      plenario: true,
      render: function (id, texto) { return api.render(id, texto); }
    });
  }
  // Texto maior nos diagramas, para projeção (o mermaid mede já com esse tamanho).
  if (window.mermaid && typeof mermaidOpts !== "undefined") {
    mermaid.initialize(Object.assign({}, mermaidOpts, { fontSize: 22, themeVariables: { fontSize: "22px" } }));
  }
  corrige();
  document.addEventListener("DOMContentLoaded", corrige);
})();
</script>]])

  -- PDF automático: padrão ligado no acadêmico e desligado no curso.
  local gerar = meta["gerar-pdf"]
  local liga
  if gerar == nil then
    liga = (estilo == "academico")
  else
    liga = (gerar == true) or (texto(gerar) == "true")
  end
  if liga then
    quarto.doc.include_text("in-header", '<meta name="plenario-pdf" content="1">')
  end

  -- Caminho do logo como texto literal (sem interpretação de markdown).
  if meta["logo-evento"] ~= nil then
    meta["logo-evento"] = pandoc.MetaString(texto(meta["logo-evento"]))
  end

  return meta
end

local function Header(h)
  -- Rótulo acima do título do slide: ## Dados {kicker="Passo 1"}
  if h.attributes.kicker then
    h.content:insert(1, pandoc.Span({ pandoc.Str(h.attributes.kicker) }, pandoc.Attr("", { "slide-kicker" })))
    h.attributes.kicker = nil
  end
  -- Número da divisória: # Título {.divisoria numero="02"}
  if h.classes:includes("divisoria") and h.attributes.numero then
    h.content:insert(1, pandoc.Span({ pandoc.Str(h.attributes.numero) }, pandoc.Attr("", { "divisoria-numero" })))
  end
  -- Divisória e encerramento ficam centralizados na vertical.
  if (h.classes:includes("divisoria") or h.classes:includes("encerramento"))
      and not h.classes:includes("center") then
    h.classes:insert("center")
  end
  local fundos = ESTILOS[estilo]
  for classe, cor in pairs(fundos) do
    if h.classes:includes(classe) and h.attributes["background-color"] == nil then
      h.attributes["background-color"] = cor
    end
  end
  return h
end

local function Span(s)
  if not s.classes:includes("medidor") then return nil end
  local uso = tonumber(s.attributes.uso or "")
  if not uso then return nil end
  local nivel = (uso >= 80 and "alto") or (uso >= 40 and "medio") or "baixo"
  local barra = string.format(
    '<span class="medidor-barra nivel-%s" role="img" aria-label="%d%%"><span style="--uso:%d%%"></span></span>',
    nivel, uso, uso)
  return {
    pandoc.RawInline("html", barra),
    pandoc.Span(s.content, pandoc.Attr("", { "medidor-rotulo" })),
  }
end

return {
  { Meta = Meta },
  { Header = Header, Span = Span },
}
