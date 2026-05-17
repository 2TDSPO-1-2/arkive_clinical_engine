"""
schemas/diagnostic.py
=====================
Schemas Pydantic v2 que definem o contrato de saída estruturada da IA.

DiagnosticoOutput:
    Schema principal mapeado para gravação futura na tabela TB_ARKIVE_DIAGNOSTICO
    pelo serviço Java/Spring downstream. Os nomes de campo e domínios de valores
    refletem exatamente as colunas e CHECK constraints do banco Oracle.

AmbiguityAssessment:
    Schema interno usado na etapa de decisão sobre necessidade de busca web.
    Não é persistido — serve apenas para controle de fluxo interno do agente.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DiagnosticoOutput(BaseModel):
    """
    Saída estruturada do Motor de Inteligência Clínica Veterinária ArkIve.

    Mapeamento para TB_ARKIVE_DIAGNOSTICO (gravação pelo serviço Java):
    ┌──────────────────────┬─────────────────────────────────────────────────┐
    │ Campo Python         │ Coluna Oracle / Tipo / Restrição                │
    ├──────────────────────┼─────────────────────────────────────────────────┤
    │ ds_diagnostico       │ DS_DIAGNOSTICO  VARCHAR2(500)                   │
    │ tp_severidade        │ TP_SEVERIDADE   VARCHAR2(20)                    │
    │                      │   CHECK IN ('LEVE', 'MODERADA', 'GRAVE')        │
    │ ds_insight_ia        │ DS_INSIGHT_IA   CLOB                            │
    │ pc_confianca         │ PC_CONFIANCA    NUMBER(3) CHECK (0..100)        │
    │ fontes_pesquisadas   │ (não persistido — informativo ao chamador)      │
    └──────────────────────┴─────────────────────────────────────────────────┘
    """

    ds_diagnostico: str = Field(
        min_length=5,
        max_length=500,
        description=(
            "Linha fina / título conciso da suspeita diagnóstica clínica sugerida. "
            "Deve nomear a hipótese principal de forma objetiva. "
            "Exemplo: 'Suspeita de Hipotireoidismo Canino'."
        ),
    )

    tp_severidade: Literal["LEVE", "MODERADA", "GRAVE"] = Field(
        description=(
            "Classificação de severidade do quadro clínico. Valores aceitos: "
            "LEVE (quadro estável, sem risco imediato), "
            "MODERADA (requer atenção e acompanhamento próximo), "
            "GRAVE (risco de vida — intervenção urgente indicada)."
        )
    )

    ds_insight_ia: str = Field(
        min_length=50,
        description=(
            "Raciocínio clínico detalhado da IA em português. Deve incluir: "
            "(1) correlação entre os sintomas relatados e a hipótese diagnóstica; "
            "(2) cruzamento com dados de bem-estar (apetite, atividade, comportamento); "
            "(3) consideração das predisposições genéticas da raça/espécie; "
            "(4) limitações e fatores que reduzem a confiança diagnóstica; "
            "(5) integração de fontes web consultadas, se aplicável."
        ),
    )

    pc_confianca: int = Field(
        ge=0,
        le=100,
        description=(
            "Grau de certeza estimado pela IA (0 a 100%). "
            "Deve refletir honestamente a completude e especificidade dos dados clínicos. "
            "Dados insuficientes ou sintomas inespecíficos devem resultar em valor baixo (<50)."
        ),
    )

    fontes_pesquisadas: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de URLs ou identificadores de fontes consultadas na internet "
            "durante a etapa de busca web. "
            "DEVE estar vazia se os dados locais do Oracle foram suficientes para o diagnóstico."
        ),
    )

    @field_validator("ds_diagnostico", mode="before")
    @classmethod
    def strip_diagnostico(cls, v: str) -> str:
        """Remove espaços extras e normaliza o título do diagnóstico."""
        return v.strip() if isinstance(v, str) else v

    @field_validator("ds_insight_ia", mode="before")
    @classmethod
    def strip_insight(cls, v: str) -> str:
        """Remove espaços extras do campo de raciocínio clínico."""
        return v.strip() if isinstance(v, str) else v


class AmbiguityAssessment(BaseModel):
    """
    Avaliação interna de ambiguidade do quadro clínico.

    Usada pelo agente para decidir se a busca web deve ser acionada.
    Não é exposta externamente nem persistida no banco.
    """

    needs_web_search: bool = Field(
        description=(
            "True se os sintomas forem raros, ambíguos, inespecíficos ou insuficientes "
            "para sustentar uma hipótese diagnóstica confiável com base apenas nos dados "
            "locais. Exemplos de casos que exigem busca: espécies exóticas, apresentações "
            "atípicas de doenças comuns, sintomas isolados sem contexto clínico."
        )
    )

    confidence_with_local_data: int = Field(
        ge=0,
        le=100,
        description=(
            "Confiança diagnóstica estimada (0-100%) usando exclusivamente os dados "
            "extraídos do Oracle, sem consulta a fontes externas."
        ),
    )

    suggested_search_query: str = Field(
        default="",
        description=(
            "Query de busca veterinária sugerida em inglês ou português, a ser usada "
            "no DuckDuckGo se needs_web_search for True. "
            "Deve ser específica e incluir espécie, sintomas principais e contexto clínico. "
            "Deixar vazio se needs_web_search for False."
        ),
    )

    reasoning: str = Field(
        description=(
            "Explicação breve (1-3 frases) do motivo pelo qual a busca web é ou "
            "não é necessária neste caso clínico específico."
        )
    )
