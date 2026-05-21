"""
agents/clinical_agent.py
========================
Motor de Inteligência Clínica Veterinária ArkIve.
Pipeline: Oracle (READ-ONLY) → heurística local → pc_confianca determinístico
          → DuckDuckGo opcional → 1 chamada ao Groq.
"""

from __future__ import annotations

import logging
import re
import time
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

# System Prompt — injetado na única chamada ao Groq

_DIAGNOSTIC_SYSTEM_PROMPT = (
    "Você é o Motor de Inteligência Clínica Veterinária do sistema ArkIve, "
    "desenvolvido para auxiliar médicos veterinários brasileiros na formulação "
    "de hipóteses diagnósticas fundamentadas. O sistema atende qualquer espécie "
    "animal — doméstica, silvestre, zoológica ou de produção.\n\n"

    "PAPEL E RESPONSABILIDADE:\n"
    "Analise os dados clínicos fornecidos e gere uma suspeita diagnóstica "
    "estruturada, priorizando a segurança do paciente e a precisão clínica. "
    "Você está produzindo uma HIPÓTESE DIAGNÓSTICA para orientar o veterinário "
    "— não um diagnóstico definitivo.\n\n"

    "INSTRUÇÕES CRÍTICAS ANTI-ALUCINAÇÃO:\n"
    "1. Baseie-se EXCLUSIVAMENTE nos dados clínicos fornecidos e em evidências "
    "médico-veterinárias estabelecidas (ou nas fontes web incluídas no contexto).\n"
    "2. NUNCA invente sintomas, resultados laboratoriais ou informações ausentes.\n"
    "3. Predisposições genéticas são FATORES DE RISCO, não diagnósticos definitivos.\n"
    "4. Se fontes web foram consultadas, integre as evidências de forma crítica "
    "no raciocínio clínico. Não liste URLs dentro do ds_insight_ia — as fontes "
    "já serão registradas separadamente no campo fontes_pesquisadas.\n\n"

    "ESTRUTURA OBRIGATÓRIA DO ds_insight_ia:\n"
    "Escreva em português técnico e objetivo, seguindo exatamente esta ordem, "
    "sem títulos ou marcadores — apenas parágrafos fluidos:\n"
    "  1º parágrafo: perfil do paciente e apresentação clínica principal.\n"
    "  2º parágrafo: correlação entre sintomas, bem-estar e hipótese diagnóstica.\n"
    "  3º parágrafo: papel das predisposições genéticas no raciocínio clínico.\n"
    "  4º parágrafo: limitações do diagnóstico e exames complementares sugeridos. "
    "Se fontes web enriqueceram o diagnóstico, mencione apenas que evidências "
    "da literatura veterinária corroboram a hipótese — sem colar URLs.\n\n"

    "RACIOCÍNIO CLÍNICO ESPERADO:\n"
    "- Correlacione sintomas com espécie, raça, sexo e status reprodutivo.\n"
    "- Considere dados de bem-estar como indicadores sistêmicos relevantes.\n"
    "- Priorize predisposições genéticas mapeadas como diferenciais prioritários.\n\n"

    "GARANTIAS DE TIPO OBRIGATÓRIAS — VIOLAÇÕES CAUSAM FALHA NO SISTEMA:\n"
    "• `ds_diagnostico`     → string de texto, entre 5 e 500 caracteres.\n"
    "• `tp_severidade`      → exatamente uma destas strings: 'LEVE', 'MODERADA' "
    "ou 'GRAVE'. Nunca use outros valores.\n"
    "• `ds_insight_ia`      → string de texto, mínimo 50 caracteres. "
    "PROIBIDO incluir URLs, links ou endereços web neste campo. "
    "Se fontes web foram consultadas, mencione apenas que a literatura "
    "veterinária corrobora a hipótese — as URLs ficam exclusivamente "
    "em fontes_pesquisadas.\n"
    "• `pc_confianca`       → inteiro puro fornecido pelo sistema no campo "
    "'>>> VALOR OBRIGATÓRIO: pc_confianca <<<'. Use EXATAMENTE este número. "
    "NUNCA recalcule, NUNCA ajuste, NUNCA envie como string ou float.\n"
    "• `fontes_pesquisadas` → lista de strings com URLs. Lista vazia [] se "
    "busca web não foi realizada. NUNCA null ou omitido.\n\n"

    "Responda EXCLUSIVAMENTE no formato JSON estruturado conforme o schema "
    "fornecido. Não inclua texto adicional fora do JSON."
)

# Resultado da heurística local de ambiguidade (sem LLM)


@dataclass
class _LocalAmbiguityResult:
    """Resultado da avaliação de ambiguidade feita em Python puro (sem LLM)."""
    needs_web_search: bool
    score: int          # 0-100: estimativa local de confiança
    reason: str         # Descrição legível do motivo
    search_query: str   # Query sugerida para o DuckDuckGo



class ClinicalIntelligenceEngine:
    """
    Motor de Inteligência Clínica Veterinária com RAG local (Oracle) e
    fallback automático de busca web (DuckDuckGo).

    Uso::

        engine = ClinicalIntelligenceEngine()
        result: dict = engine.analyze(id_consulta=42)
    """

    def __init__(self) -> None:
        """Inicializa LLM Groq e a chain de diagnóstico estruturado."""
        logger.info("Inicializando ClinicalIntelligenceEngine | Modelo: %s", GROQ_MODEL)

        self._llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=GROQ_TEMPERATURE,
        )

        # Structured output via function calling do Groq
        self._diagnostic_chain = self._llm.with_structured_output(DiagnosticoOutput)

        logger.info("Chain de diagnóstico inicializada (1 chamada de API por execução).")

    # Método Público Principal

    def analyze(self, id_consulta: int) -> dict[str, Any]:
        """
        Ponto de entrada principal. Faz exatamente 1 chamada ao Groq.

        Raises:
            ValueError: Consulta não encontrada no banco.
            RuntimeError: Falha ao gerar diagnóstico no Groq.
            oracledb.DatabaseError: Falha de acesso ao Oracle.
        """
        logger.info("── Iniciando análise clínica | ID_CONSULTA=%d ──", id_consulta)

        # Etapa 1: Extração Oracle
        ctx: ClinicalContext = self._fetch_oracle_data(id_consulta)
        clinical_summary: str = ctx.to_clinical_summary()

        # Etapa 2: Heurística local de ambiguidade
        ambiguity = _evaluate_ambiguity_locally(ctx)
        logger.info(
            "Heurística local | score=%d%% | busca_web=%s | motivo: %s",
            ambiguity.score,
            ambiguity.needs_web_search,
            ambiguity.reason,
        )

        # Etapa 3: Cálculo determinístico do pc_confianca
        sintomas = (ctx.ds_sintomas or "").strip()
        confianca_calculada = _calculate_confidence(ctx, sintomas)
        logger.info("Confiança calculada deterministicamente: %d%%", confianca_calculada)

        # Etapa 4: Busca web condicional
        web_context: str = ""
        sources: list[str] = []

        if ambiguity.needs_web_search:
            logger.info("Acionando busca web | Query: '%s'", ambiguity.search_query)
            web_context, sources = self._perform_web_search(ambiguity.search_query)
            logger.info("%d fonte(s) recuperada(s) do DuckDuckGo.", len(sources))
        else:
            logger.info("Dados locais suficientes — busca web não acionada.")

        # Etapa 5: Chamada ao Groq
        logger.info("Enviando requisição ao Groq (chamada 1/1)...")
        diagnostic: DiagnosticoOutput = self._generate_diagnostic(
            clinical_summary=clinical_summary,
            web_context=web_context,
            sources=sources,
            ctx=ctx,
            confianca_calculada=confianca_calculada,
        )

        logger.info(
            "Diagnóstico gerado | '%s' | Severidade: %s | Confiança: %d%%",
            diagnostic.ds_diagnostico,
            diagnostic.tp_severidade,
            diagnostic.pc_confianca,
        )

        return diagnostic.model_dump()

    # Métodos Privados

    def _fetch_oracle_data(self, id_consulta: int) -> ClinicalContext:
        """Extrai dados clínicos do Oracle usando conexão READ-ONLY em Thin mode."""
        logger.info("Conectando ao Oracle (Thin mode)...")
        with get_connection() as conn:
            ctx = fetch_clinical_data(conn, id_consulta)
        logger.info(
            "Dados extraídos | Animal: %s | Espécie: %s | Raça: %s | Predisposições: %d",
            ctx.nm_animal,
            ctx.nm_especie,
            ctx.nm_raca or "SRD/Não informada",
            len(ctx.predisposicoes),
        )
        return ctx

    def _perform_web_search(self, query: str) -> tuple[str, list[str]]:
        """Busca no DuckDuckGo. Retorna (texto_contexto, lista_de_urls)."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("ddgs não instalado: pip install ddgs")
            return "", []

        snippets: list[str] = []
        urls: list[str] = []

        try:
            time.sleep(2)  # Reduz chance de rate limit em execuções consecutivas
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5, safesearch="moderate"))
            for result in results:
                href = result.get("href", "")
                if href:
                    snippets.append(
                        f"📄 {result.get('title', '')}\n"
                        f"{result.get('body', '')}\n"
                        f"Fonte: {href}"
                    )
                    urls.append(href)
            web_context = "\n\n" + ("─" * 60) + "\n\n".join(snippets)
        except Exception as exc:
            logger.warning("Busca web falhou: %s. Prosseguindo sem contexto externo.", exc)
            web_context = ""
            urls = []

        return web_context, urls

    def _generate_diagnostic(
        self,
        clinical_summary: str,
        web_context: str,
        sources: list[str],
        ctx: ClinicalContext,
        confianca_calculada: int,
    ) -> DiagnosticoOutput:
        """Monta o prompt completo e faz a única chamada ao Groq. Retorna DiagnosticoOutput validado."""
        parts = [
            "Analise os seguintes dados clínicos veterinários e gere o "
            "diagnóstico estruturado:\n\n",
            clinical_summary,
            f"\n\n>>> VALOR OBRIGATÓRIO: pc_confianca = {confianca_calculada} <<<\n"
            "Este valor foi calculado deterministicamente pelo sistema com base "
            "nos dados clínicos reais. Use EXATAMENTE este número no campo "
            "pc_confianca — não recalcule, não ajuste, não arredonde.\n",
        ]

        if web_context:
            parts.extend([
                "\n\n" + "═" * 60,
                "\n🌐 CONTEXTO ADICIONAL — LITERATURA VETERINÁRIA (BUSCA WEB):\n",
                web_context,
                "\n" + "═" * 60,
                "\nIMPORTANTE: integre as evidências ao raciocínio clínico no "
                "ds_insight_ia. Não cole URLs no insight — elas já estão em "
                "fontes_pesquisadas.",
            ])

        if sources:
            parts.append(
                "\nURLs consultadas (para fontes_pesquisadas):\n"
                + "\n".join(f"  {i + 1}. {url}" for i, url in enumerate(sources))
            )

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

        # Garante que pc_confianca é sempre o valor calculado pelo sistema
        diagnostic.pc_confianca = confianca_calculada

        # Preenche fontes caso a LLM não o tenha feito
        if sources and not diagnostic.fontes_pesquisadas:
            diagnostic.fontes_pesquisadas = sources

        return diagnostic


def _calculate_confidence(ctx: ClinicalContext, sintomas: str) -> int:
    """
    Calcula pc_confianca deterministicamente. O modelo recebe o valor pronto.

    Rubrica (base = 30):
    +25 Sintomas específicos e detalhados (> 3 palavras relevantes)
    +10 Sintomas moderadamente descritivos (1–3 palavras relevantes)
    +20 Predisposição genética diretamente relacionada aos sintomas
    +10 Predisposição genética presente mas indiretamente relacionada
    +10 Bem-estar completo (apetite + atividade + comportamento)
    +5  Peso registrado
    -10 Dados relevantes ausentes (idade, peso ou bem-estar)
    -15 Sintomas vagos ou genéricos

    Retorna inteiro entre 0 e 100.
    """
    score = 30  # BASE sempre

    # Sintomas
    palavras_relevantes = re.findall(r"\b\w{4,}\b", sintomas)
    if len(palavras_relevantes) > 3:
        score += 25
    elif len(palavras_relevantes) >= 1:
        score += 10
    else:
        score -= 15

    # Predisposições genéticas
    if ctx.predisposicoes:
        sintomas_lower = sintomas.lower()
        # Verifica se alguma doença mapeada tem termos presentes nos sintomas
        diretamente_relacionada = any(
            any(
                termo in sintomas_lower
                for termo in re.findall(r"\b\w{4,}\b", doenca.get("nm_doenca", "").lower())
            )
            for doenca in ctx.predisposicoes
        )
        score += 20 if diretamente_relacionada else 10

    # Bem-estar
    if ctx.ds_apetite and ctx.ds_atividade and ctx.ds_comportamento:
        score += 10

    # Peso
    if ctx.peso_efetivo_kg:
        score += 5

    # Penalidade: dados relevantes ausentes
    dados_ausentes = (
        not ctx.nr_idade
        or not ctx.peso_efetivo_kg
        or (not ctx.ds_apetite and not ctx.ds_atividade and not ctx.ds_comportamento)
    )
    if dados_ausentes:
        score -= 10

    resultado = max(0, min(100, score))

    logger.debug(
        "Rubrica pc_confianca | sintomas=%d palavras | predisposições=%d | "
        "bem-estar=%s | peso=%s | resultado=%d",
        len(palavras_relevantes),
        len(ctx.predisposicoes),
        "completo" if ctx.ds_apetite and ctx.ds_atividade and ctx.ds_comportamento else "parcial/ausente",
        f"{ctx.peso_efetivo_kg}kg" if ctx.peso_efetivo_kg else "ausente",
        resultado,
    )

    return resultado


def _evaluate_ambiguity_locally(ctx: ClinicalContext) -> _LocalAmbiguityResult:
    """
    Avalia qualidade dos dados clínicos com regras determinísticas (sem LLM).
    Se score < AMBIGUITY_THRESHOLD → needs_web_search = True.

    Penalidades (base = 100):
    -35 DS_SINTOMAS ausente ou < 20 caracteres
    -15 DS_SINTOMAS genérico (1 palavra)
    -20 DS_MOTIVO ausente ou < 10 caracteres
    -20 Sem predisposições genéticas mapeadas para a espécie/raça
    -10 Avaliação de bem-estar ausente
    """
    score = 100
    reasons: list[str] = []

    sintomas = (ctx.ds_sintomas or "").strip()
    motivo = (ctx.ds_motivo or "").strip()

    # Penalidade: sintomas ausentes ou muito curtos
    if len(sintomas) < 20:
        score -= 35
        reasons.append("sintomas ausentes ou insuficientes")
    elif len(re.findall(r"\w+", sintomas)) <= 1:
        score -= 15
        reasons.append("sintomas excessivamente genéricos")

    # Penalidade: motivo ausente ou muito curto
    if len(motivo) < 10:
        score -= 20
        reasons.append("motivo da consulta não informado")

    # Penalidade: sem predisposições mapeadas para espécie/raça
    if not ctx.predisposicoes:
        especie_str = ctx.nm_especie or "não identificada"
        raca_str = f" / raça '{ctx.nm_raca}'" if ctx.nm_raca else ""
        score -= 20
        reasons.append(
            f"nenhuma predisposição genética mapeada no banco para a espécie "
            f"'{especie_str}'{raca_str} — busca web pode enriquecer o diagnóstico"
        )

    # Penalidade: sem avaliação de bem-estar
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
    """Monta query veterinária para o DuckDuckGo priorizando NCBI e Merck."""
    parts: list[str] = []

    if ctx.nm_especie:
        parts.append(ctx.nm_especie.lower())
    if ctx.nm_raca:
        parts.append(ctx.nm_raca.lower())

    texto = sintomas or motivo
    if texto:
        parts.extend(re.findall(r"\b\w{4,}\b", texto)[:4])

    parts.append(
        "veterinary clinical diagnosis "
        "site:ncbi.nlm.nih.gov OR site:merckvetmanual.com"
    )

    return " ".join(parts)[:200]