"""
agents/clinical_agent.py
========================
Motor de Inteligência Clínica Veterinária ArkIve.

Arquitetura de chamada única à API (Single-Call) via Groq:

  Etapa 1 — Extração Oracle (READ-ONLY, sem LLM)
  Etapa 2 — Heurística local de ambiguidade (sem LLM)
  Etapa 3 — Busca web DuckDuckGo condicional (sem LLM)
  Etapa 4 — UMA ÚNICA chamada ao Groq com contexto completo
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import (
    AMBIGUITY_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)
from database.connection import get_connection
from database.queries import ClinicalContext, fetch_clinical_data
from schemas.diagnostic import DiagnosticoOutput

logger = logging.getLogger(__name__)

_ESPECIES_COMUNS: frozenset[str] = frozenset({
    "canino", "cão", "cao", "cachorro", "dog",
    "felino", "gato", "cat",
})

_DIAGNOSTIC_SYSTEM_PROMPT = """\
Você é o Motor de Inteligência Clínica Veterinária do sistema ArkIve, \
desenvolvido para auxiliar médicos veterinários brasileiros na formulação \
de hipóteses diagnósticas fundamentadas.

PAPEL E RESPONSABILIDADE:
Analise os dados clínicos fornecidos e gere uma suspeita diagnóstica estruturada, \
priorizando a segurança do paciente e a precisão clínica. Você está produzindo \
uma HIPÓTESE DIAGNÓSTICA para orientar o veterinário — não um diagnóstico definitivo.

INSTRUÇÕES CRÍTICAS ANTI-ALUCINAÇÃO:
1. Baseie-se EXCLUSIVAMENTE nos dados clínicos fornecidos e em evidências \
   médico-veterinárias estabelecidas (ou nas fontes web incluídas no contexto).
2. NUNCA invente sintomas, resultados laboratoriais ou informações ausentes.
3. Calibre o pc_confianca pela seguinte escala obrigatória:
   - 80-100: sintomas específicos + predisposição genética confirmada + \
     todos os dados de bem-estar presentes e coerentes.
   - 60-79:  sintomas moderadamente específicos + pelo menos um fator \
     corroborante (predisposição ou bem-estar alterado).
   - 40-59:  sintomas inespecíficos ou dados parcialmente ausentes.
   - 0-39:   dados insuficientes para qualquer hipótese confiável. \
     Neste caso reduza a severidade para LEVE salvo evidência contrária.
4. Predisposições genéticas são FATORES DE RISCO, não diagnósticos definitivos.
5. Se fontes web foram consultadas, integre as evidências de forma crítica \
   e mencione-as no ds_insight_ia.

RACIOCÍNIO CLÍNICO ESPERADO:
- Correlacione sintomas com espécie, raça, sexo e status reprodutivo.
- Considere dados de bem-estar (apetite, atividade, comportamento) como \
  indicadores sistêmicos relevantes.
- Priorize as predisposições genéticas mapeadas como diferenciais prioritários.
- Apresente o raciocínio em português claro e técnico.

Responda EXCLUSIVAMENTE no formato JSON estruturado conforme o schema fornecido. \
Não inclua texto adicional fora do JSON.
"""


@dataclass
class _LocalAmbiguityResult:
    needs_web_search: bool
    score: int
    reason: str
    search_query: str


class ClinicalIntelligenceEngine:
    """
    Motor de Inteligência Clínica com RAG local (Oracle) e fallback DuckDuckGo.
    Faz NO MÁXIMO 1 chamada à API do Groq por execução.
    """

    def __init__(self) -> None:
        logger.info("Inicializando ClinicalIntelligenceEngine | Modelo: %s", GROQ_MODEL)

        self._llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=GROQ_TEMPERATURE,
        )

        self._diagnostic_chain = self._llm.with_structured_output(DiagnosticoOutput)
        logger.info("Chain de diagnóstico inicializada (1 chamada de API por execução).")

    def analyze(self, id_consulta: int) -> dict[str, Any]:
        """
        Ponto de entrada principal. Faz exatamente 1 chamada ao Groq.

        Args:
            id_consulta: ID numérico da consulta a ser analisada.

        Returns:
            Dict com os campos de DiagnosticoOutput serializados.
        """
        logger.info("── Iniciando análise clínica | ID_CONSULTA=%d ──", id_consulta)

        # Etapa 1: Oracle
        ctx: ClinicalContext = self._fetch_oracle_data(id_consulta)
        clinical_summary: str = ctx.to_clinical_summary()

        # Etapa 2: Heurística local
        ambiguity = _evaluate_ambiguity_locally(ctx)
        logger.info(
            "Heurística local | score=%d%% | busca_web=%s | motivo: %s",
            ambiguity.score, ambiguity.needs_web_search, ambiguity.reason,
        )

        # Etapa 3: Busca web condicional
        web_context: str = ""
        sources: list[str] = []

        if ambiguity.needs_web_search:
            logger.info("Acionando busca web | Query: '%s'", ambiguity.search_query)
            web_context, sources = self._perform_web_search(ambiguity.search_query)
            logger.info("%d fonte(s) recuperada(s) do DuckDuckGo.", len(sources))
        else:
            logger.info("Dados locais suficientes — busca web não acionada.")

        # Etapa 4: Única chamada ao Groq
        logger.info("Enviando requisição ao Groq (chamada 1/1)...")
        diagnostic: DiagnosticoOutput = self._generate_diagnostic(
            clinical_summary=clinical_summary,
            web_context=web_context,
            sources=sources,
        )

        logger.info(
            "Diagnóstico gerado | '%s' | Severidade: %s | Confiança: %d%%",
            diagnostic.ds_diagnostico,
            diagnostic.tp_severidade,
            diagnostic.pc_confianca,
        )

        return diagnostic.model_dump()

    def _fetch_oracle_data(self, id_consulta: int) -> ClinicalContext:
        logger.info("Conectando ao Oracle (Thin mode)...")
        with get_connection() as conn:
            ctx = fetch_clinical_data(conn, id_consulta)
        logger.info(
            "Dados extraídos | Animal: %s (%s) | Predisposições: %d",
            ctx.nm_animal, ctx.nm_especie, len(ctx.predisposicoes),
        )
        return ctx

    def _perform_web_search(self, query: str) -> tuple[str, list[str]]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("duckduckgo-search não instalado: pip install duckduckgo-search")
            return "", []

        snippets: list[str] = []
        urls: list[str] = []

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5, safesearch="moderate"))
            for result in results:
                href = result.get("href", "")
                if href:
                    snippets.append(
                        f"📄 {result.get('title', '')}\n{result.get('body', '')}\nFonte: {href}"
                    )
                    urls.append(href)
            web_context = "\n\n" + ("─" * 60) + "\n\n".join(snippets)
        except Exception as exc:
            logger.warning("Busca web falhou: %s.", exc)
            web_context = ""
            urls = []

        return web_context, urls

    def _generate_diagnostic(
        self,
        clinical_summary: str,
        web_context: str,
        sources: list[str],
    ) -> DiagnosticoOutput:
        parts = [
            "Analise os seguintes dados clínicos veterinários e gere o "
            "diagnóstico estruturado:\n\n",
            clinical_summary,
        ]

        if web_context:
            parts.extend([
                "\n\n" + "═" * 60,
                "\n🌐 CONTEXTO ADICIONAL — LITERATURA VETERINÁRIA (BUSCA WEB):\n",
                web_context,
                "\n" + "═" * 60,
                "\nIMPORTANTE: integre as evidências ao raciocínio e cite as fontes "
                "no campo ds_insight_ia.",
            ])

        if sources:
            parts.append("\nURLs consultadas:\n" + "\n".join(
                f"  {i + 1}. {url}" for i, url in enumerate(sources)
            ))

        messages = [
            SystemMessage(content=_DIAGNOSTIC_SYSTEM_PROMPT),
            HumanMessage(content="".join(parts)),
        ]

        try:
            diagnostic: DiagnosticoOutput = self._diagnostic_chain.invoke(messages)
        except Exception as exc:
            logger.error("Falha na chamada ao Groq: %s", exc)
            raise RuntimeError(
                f"O modelo não gerou um diagnóstico estruturado válido: {exc}"
            ) from exc

        if sources and not diagnostic.fontes_pesquisadas:
            diagnostic.fontes_pesquisadas = sources

        return diagnostic


# ─────────────────────────────────────────────────────────────────────────────
#  Heurística Local de Ambiguidade (zero chamadas de API)
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_ambiguity_locally(ctx: ClinicalContext) -> _LocalAmbiguityResult:
    """
    Avalia a qualidade dos dados clínicos com regras determinísticas.

    Penalidades:
      - Sintomas ausentes / < 20 chars  → -35
      - Sintomas genéricos (1 palavra)  → -15
      - Motivo ausente / < 10 chars     → -20
      - Espécie exótica                 → -20
      - Sem predisposições mapeadas     → -10
      - Sem avaliação de bem-estar      → -10
    """
    score = 100
    reasons: list[str] = []

    sintomas = (ctx.ds_sintomas or "").strip()
    motivo = (ctx.ds_motivo or "").strip()
    especie = (ctx.nm_especie or "").lower().strip()

    if len(sintomas) < 20:
        score -= 35
        reasons.append("sintomas ausentes ou insuficientes")
    elif len(re.findall(r"\w+", sintomas)) <= 1:
        score -= 15
        reasons.append("sintomas excessivamente genéricos")

    if len(motivo) < 10:
        score -= 20
        reasons.append("motivo da consulta não informado")

    if not any(comum in especie for comum in _ESPECIES_COMUNS):
        score -= 20
        reasons.append(f"espécie exótica ({ctx.nm_especie or 'desconhecida'})")

    if not ctx.predisposicoes:
        score -= 10
        reasons.append("nenhuma predisposição genética mapeada")

    if not ctx.ds_apetite and not ctx.ds_atividade and not ctx.ds_comportamento:
        score -= 10
        reasons.append("avaliação de bem-estar ausente")

    score = max(0, score)

    return _LocalAmbiguityResult(
        needs_web_search=score < AMBIGUITY_THRESHOLD,
        score=score,
        reason="; ".join(reasons) if reasons else "dados clínicos suficientes",
        search_query=_build_search_query(ctx, sintomas, motivo),
    )


def _build_search_query(ctx: ClinicalContext, sintomas: str, motivo: str) -> str:
    parts: list[str] = []
    if ctx.nm_especie:
        parts.append(ctx.nm_especie.lower())
    if ctx.nm_raca:
        parts.append(ctx.nm_raca.lower())
    texto = sintomas or motivo
    if texto:
        parts.extend(re.findall(r"\b\w{4,}\b", texto)[:5])
    parts.append("veterinary diagnosis treatment")
    return " ".join(parts)[:200]