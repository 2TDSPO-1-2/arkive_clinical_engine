"""
agents/clinical_agent.py
========================
Motor de Inteligência Clínica Veterinária ArkIve.

Arquitetura em duas etapas (Two-Stage Chain):

  Etapa 1 — Avaliação de Ambiguidade (AmbiguityChain):
      A LLM analisa os dados clínicos extraídos do Oracle e decide se a
      confiança local é suficiente (>= AMBIGUITY_THRESHOLD). Se não for,
      gera uma query de busca específica para literatura veterinária.

  Etapa 2 — Busca Web (condicional):
      Se acionada, consulta o DuckDuckGo e enriquece o contexto clínico
      com snippets e URLs de fontes veterinárias atuais.

  Etapa 3 — Geração de Diagnóstico Estruturado (DiagnosticChain):
      A LLM recebe o contexto clínico completo (dados Oracle + contexto web
      opcional) e produz um DiagnosticoOutput validado pelo Pydantic v2
      via `with_structured_output()` — função calling nativa do Gemini.

Referências:
  - oracledb Thin mode: https://python-oracledb.readthedocs.io/
  - LangChain structured output: https://python.langchain.com/docs/concepts/structured_outputs/
  - Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    AMBIGUITY_THRESHOLD,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GOOGLE_API_KEY,
)
from database.connection import get_connection
from database.queries import ClinicalContext, fetch_clinical_data
from schemas.diagnostic import AmbiguityAssessment, DiagnosticoOutput

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  System Prompts
# ─────────────────────────────────────────────────────────────────────────────

_AMBIGUITY_SYSTEM_PROMPT = """\
Você é um especialista em medicina veterinária clínica. Sua tarefa é avaliar \
se os dados clínicos fornecidos de uma consulta veterinária são suficientes \
para formular uma hipótese diagnóstica confiável, ou se é necessário realizar \
uma busca em literatura médico-veterinária externa para enriquecer o contexto.

CRITÉRIOS PARA ACIONAR BUSCA EXTERNA:
- Sintomas inespecíficos (ex: "prostração", "vômito" sem outros detalhes)
- Espécies exóticas ou não-convencionais (répteis, aves, roedores exóticos)
- Combinação incomum de sintomas sem predisposição raçal mapeada
- Confiança diagnóstica estimada inferior a {threshold}% apenas com dados locais
- Ausência quase total de dados clínicos (motivo/sintomas vazios ou genéricos)

Seja conservador: prefira buscar quando em dúvida. A segurança do paciente \
depende de uma hipótese bem fundamentada.

Responda EXCLUSIVAMENTE no formato JSON estruturado conforme o schema fornecido.
""".format(threshold=AMBIGUITY_THRESHOLD)

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
3. Quando dados forem insuficientes, declare explicitamente a limitação e \
   reduza o pc_confianca de forma conservadora.
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

# ─────────────────────────────────────────────────────────────────────────────
#  Classe Principal: ClinicalIntelligenceEngine
# ─────────────────────────────────────────────────────────────────────────────


class ClinicalIntelligenceEngine:
    """
    Motor de Inteligência Clínica Veterinária com RAG local (Oracle) e
    fallback automático de busca web (DuckDuckGo).

    Uso::

        engine = ClinicalIntelligenceEngine()
        result: dict = engine.analyze(id_consulta=42)
        print(result)  # DiagnosticoOutput serializado como dict
    """

    def __init__(self) -> None:
        """Inicializa a LLM Gemini e as chains estruturadas."""
        logger.info("Inicializando ClinicalIntelligenceEngine | Modelo: %s", GEMINI_MODEL)

        # LLM base — usada em ambas as etapas
        self._llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=GEMINI_TEMPERATURE,
            # convert_system_message_to_human=True é necessário para alguns modelos,
            # mas não para gemini-1.5-flash que suporta system messages nativas.
        )

        # Chain de avaliação de ambiguidade — retorna AmbiguityAssessment validado
        self._ambiguity_chain = self._llm.with_structured_output(AmbiguityAssessment)

        # Chain de diagnóstico estruturado — retorna DiagnosticoOutput validado
        self._diagnostic_chain = self._llm.with_structured_output(DiagnosticoOutput)

        logger.info("Chains LangChain inicializadas com sucesso.")

    # ──────────────────────────────────────────────────────────────────────────
    #  Método Público Principal
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(self, id_consulta: int) -> dict[str, Any]:
        """
        Ponto de entrada principal do motor de inteligência clínica.

        Pipeline:
          1. Extrai dados clínicos do Oracle (READ-ONLY).
          2. Avalia ambiguidade dos sintomas via LLM.
          3. Executa busca web (DuckDuckGo) se necessário.
          4. Gera diagnóstico estruturado validado pelo Pydantic.

        Args:
            id_consulta: ID numérico da consulta a ser analisada.

        Returns:
            Dict com os campos de DiagnosticoOutput serializados.

        Raises:
            ValueError:            Se a consulta não for encontrada no banco.
            RuntimeError:          Em caso de falha irrecuperável na LLM.
            oracledb.DatabaseError: Em caso de falha de acesso ao Oracle.
        """
        logger.info("── Iniciando análise clínica | ID_CONSULTA=%d ──", id_consulta)

        # ── Etapa 1: Extração de Dados do Oracle ─────────────────────────────
        ctx: ClinicalContext = self._fetch_oracle_data(id_consulta)

        # ── Etapa 2: Avaliação de Ambiguidade ────────────────────────────────
        clinical_summary: str = ctx.to_clinical_summary()
        ambiguity: AmbiguityAssessment = self._assess_ambiguity(clinical_summary)

        logger.info(
            "Avaliação de ambiguidade: needs_web_search=%s | confiança local=%d%% | %s",
            ambiguity.needs_web_search,
            ambiguity.confidence_with_local_data,
            ambiguity.reasoning,
        )

        # ── Etapa 3: Busca Web (Condicional) ──────────────────────────────────
        web_context: str = ""
        sources: list[str] = []

        should_search = (
            ambiguity.needs_web_search
            or ambiguity.confidence_with_local_data < AMBIGUITY_THRESHOLD
        )

        if should_search and ambiguity.suggested_search_query:
            logger.info(
                "Acionando busca web | Query: '%s'", ambiguity.suggested_search_query
            )
            web_context, sources = self._perform_web_search(ambiguity.suggested_search_query)
            logger.info("Busca web concluída | %d fonte(s) encontrada(s).", len(sources))
        else:
            logger.info(
                "Dados locais suficientes (confiança=%d%%) — busca web não acionada.",
                ambiguity.confidence_with_local_data,
            )

        # ── Etapa 4: Geração do Diagnóstico Estruturado ───────────────────────
        diagnostic: DiagnosticoOutput = self._generate_diagnostic(
            clinical_summary=clinical_summary,
            web_context=web_context,
            sources=sources,
        )

        logger.info(
            "Diagnóstico gerado | Suspeita: '%s' | Severidade: %s | Confiança: %d%%",
            diagnostic.ds_diagnostico,
            diagnostic.tp_severidade,
            diagnostic.pc_confianca,
        )

        return diagnostic.model_dump()

    # ──────────────────────────────────────────────────────────────────────────
    #  Métodos Privados
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_oracle_data(self, id_consulta: int) -> ClinicalContext:
        """Extrai dados clínicos do Oracle usando conexão READ-ONLY em Thin mode."""
        logger.info("Conectando ao Oracle (Thin mode) para extração de dados...")
        with get_connection() as conn:
            ctx = fetch_clinical_data(conn, id_consulta)
        logger.info(
            "Dados extraídos | Animal: %s (%s) | Predisposições: %d",
            ctx.nm_animal,
            ctx.nm_especie,
            len(ctx.predisposicoes),
        )
        return ctx

    def _assess_ambiguity(self, clinical_summary: str) -> AmbiguityAssessment:
        """
        Usa a LLM para avaliar se os dados clínicos locais são suficientes
        para uma hipótese diagnóstica confiável.

        Returns:
            AmbiguityAssessment com decisão de busca e query sugerida.
        """
        logger.debug("Avaliando ambiguidade dos dados clínicos...")

        messages = [
            SystemMessage(content=_AMBIGUITY_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Avalie os seguintes dados clínicos e determine se é necessário "
                    "realizar busca em literatura veterinária externa:\n\n"
                    f"{clinical_summary}"
                )
            ),
        ]

        try:
            assessment: AmbiguityAssessment = self._ambiguity_chain.invoke(messages)
        except Exception as exc:
            logger.warning(
                "Falha na avaliação de ambiguidade: %s. "
                "Assumindo necessidade de busca web como precaução.",
                exc,
            )
            # Fallback conservador: assume ambiguidade para não perder contexto relevante
            assessment = AmbiguityAssessment(
                needs_web_search=True,
                confidence_with_local_data=40,
                suggested_search_query=(
                    f"veterinary clinical diagnosis symptoms {clinical_summary[:100]}"
                ),
                reasoning="Falha na avaliação de ambiguidade — busca web acionada por precaução.",
            )

        return assessment

    def _perform_web_search(self, query: str) -> tuple[str, list[str]]:
        """
        Realiza busca no DuckDuckGo usando a biblioteca duckduckgo-search
        (a mesma usada internamente pelo DuckDuckGoSearchRun do LangChain Community).
        Acesso direto à API para extração de URLs estruturadas.

        Args:
            query: Query de busca em linguagem natural.

        Returns:
            Tupla (web_context_text, list_of_urls).
        """
        # Import local para evitar ImportError se a biblioteca não estiver instalada
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error(
                "duckduckgo-search não instalado. Execute: pip install duckduckgo-search"
            )
            return "", []

        snippets: list[str] = []
        urls: list[str] = []

        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        max_results=5,
                        safesearch="moderate",
                    )
                )

            for result in results:
                title = result.get("title", "Sem título")
                body = result.get("body", "")
                href = result.get("href", "")

                if href:
                    snippets.append(
                        f"📄 {title}\n{body}\nFonte: {href}"
                    )
                    urls.append(href)

            web_context = "\n\n" + ("─" * 60) + "\n\n".join(snippets)

        except Exception as exc:
            logger.warning(
                "Busca web falhou: %s. Prosseguindo sem contexto externo.", exc
            )
            web_context = ""
            urls = []

        return web_context, urls

    def _generate_diagnostic(
        self,
        clinical_summary: str,
        web_context: str,
        sources: list[str],
    ) -> DiagnosticoOutput:
        """
        Gera o diagnóstico clínico estruturado usando a LLM com o contexto
        completo (dados Oracle + contexto web opcional).

        O método `with_structured_output` do LangChain usa function calling
        nativo do Gemini para garantir que a saída seja validada pelo Pydantic.

        Args:
            clinical_summary: Resumo textual dos dados clínicos do Oracle.
            web_context:      Snippets de busca web (vazio se não realizada).
            sources:          URLs das fontes web consultadas.

        Returns:
            DiagnosticoOutput validado.
        """
        # ── Construção do prompt do usuário ───────────────────────────────
        user_content_parts = [
            "Analise os seguintes dados clínicos veterinários e gere o diagnóstico estruturado:\n\n",
            clinical_summary,
        ]

        if web_context:
            user_content_parts.extend([
                "\n\n" + "═" * 60,
                "\n🌐 CONTEXTO ADICIONAL DE LITERATURA VETERINÁRIA (BUSCA WEB):",
                "\nAs seguintes fontes foram consultadas para enriquecer o diagnóstico "
                "por insuficiência/ambiguidade dos dados locais:\n",
                web_context,
                "\n" + "═" * 60,
                "\nIMPORTANTE: Integre as evidências acima ao raciocínio clínico e "
                "cite as fontes relevantes no campo ds_insight_ia.",
            ])

        if sources:
            sources_block = "\nFontes web consultadas:\n" + "\n".join(
                f"  {i + 1}. {url}" for i, url in enumerate(sources)
            )
            user_content_parts.append(sources_block)

        user_content = "".join(user_content_parts)

        messages = [
            SystemMessage(content=_DIAGNOSTIC_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        # ── Invocação com Structured Output (Pydantic v2) ─────────────────
        logger.debug("Invocando DiagnosticChain com structured output...")
        try:
            diagnostic: DiagnosticoOutput = self._diagnostic_chain.invoke(messages)
        except Exception as exc:
            logger.error("Falha na geração do diagnóstico estruturado: %s", exc)
            raise RuntimeError(
                f"O modelo não conseguiu gerar um diagnóstico estruturado válido: {exc}"
            ) from exc

        # Garante que as fontes pesquisadas estejam preenchidas corretamente
        if sources and not diagnostic.fontes_pesquisadas:
            diagnostic.fontes_pesquisadas = sources

        return diagnostic
