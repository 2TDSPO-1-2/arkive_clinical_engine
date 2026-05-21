"""
database/queries.py
===================
SQLs parametrizados e lógica de extração de dados clínicos do Oracle.

CLINICAL_DATA_QUERY: JOIN único com avaliação de bem-estar via LATERAL JOIN
                     (requer Oracle 12c+).
PREDISPOSITION_QUERY: busca separada de predisposições genéticas da raça/espécie
                      (separada para evitar multiplicação de linhas).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import oracledb

logger = logging.getLogger(__name__)

# SQL 1: Dados Clínicos Agregados da Consulta

CLINICAL_DATA_QUERY: str = """
SELECT
    -- Animal
    a.ID_ANIMAL,
    a.NM_ANIMAL,
    a.DS_SEXO,
    a.DS_CASTRADO,

    -- Espécie
    e.ID_ESPECIE,
    e.NM_ESPECIE,

    -- Raça (LEFT JOIN: pode ser nula para animais SRD)
    r.ID_RACA,
    r.NM_RACA,
    r.TP_PORTE,

    -- Consulta atual
    c.ID_CONSULTA,
    c.DT_HORA,
    c.TP_MODALIDADE,
    c.DS_MOTIVO,
    c.DS_SINTOMAS,
    c.DS_OBSERVACAO    AS DS_OBS_CONSULTA,
    c.KG_PESO          AS KG_PESO_CONSULTA,

    -- Avaliação de Bem-Estar mais recente (via LATERAL JOIN)
    abe.NR_IDADE,
    abe.KG_PESO        AS KG_PESO_BEM_ESTAR,
    abe.DS_APETITE,
    abe.DS_ATIVIDADE,
    abe.DS_COMPORTAMENTO,
    abe.DS_OBSERVACAO  AS DS_OBS_BEM_ESTAR

FROM       TB_ARKIVE_CONSULTA             c
JOIN       TB_ARKIVE_ANIMAL               a   ON a.ID_ANIMAL  = c.ID_ANIMAL
JOIN       TB_ARKIVE_ESPECIE              e   ON e.ID_ESPECIE = a.ID_ESPECIE
LEFT JOIN  TB_ARKIVE_RACA                 r   ON r.ID_RACA    = a.ID_RACA

-- LATERAL JOIN: busca avaliação de bem-estar mais relevante para este animal.
-- ORDER BY: registros desta consulta têm prioridade 0 (sobre os demais = 1).
-- FETCH FIRST 1 ROW ONLY garante exatamente um registro por animal.
LEFT JOIN LATERAL (
    SELECT
        NR_IDADE,
        KG_PESO,
        DS_APETITE,
        DS_ATIVIDADE,
        DS_COMPORTAMENTO,
        DS_OBSERVACAO
    FROM   TB_ARKIVE_AVALIACAO_BEM_ESTAR
    WHERE  ID_ANIMAL = a.ID_ANIMAL
    ORDER BY
        CASE WHEN ID_CONSULTA = :id_consulta THEN 0 ELSE 1 END,
        ID_AVALIACAO_BEM_ESTAR DESC
    FETCH FIRST 1 ROW ONLY
) abe ON 1 = 1

WHERE c.ID_CONSULTA = :id_consulta
"""

# SQL 2: Doenças com Predisposição Genética da Raça / Espécie

PREDISPOSITION_QUERY: str = """
SELECT
    d.NM_DOENCA,
    d.DS_DOENCA

FROM   TB_ARKIVE_DOENCA d
WHERE  d.ID_DOENCA IN (
    SELECT p.ID_DOENCA
    FROM   TB_ARKIVE_PREDISPOSICAO p
    WHERE  p.ID_ESPECIE = :id_especie
      AND  (
               p.ID_RACA IS NULL
            OR :id_raca IS NULL
            OR p.ID_RACA = :id_raca
           )
)
ORDER BY d.NM_DOENCA
"""

# Contexto Clínico Completo


@dataclass
class ClinicalContext:
    """
    Contêiner tipado com todos os dados clínicos extraídos do Oracle.
    Serve como contrato entre a camada de banco e a camada de IA.
    """

    # Animal
    id_animal: int = 0
    nm_animal: str = ""
    ds_sexo: str = ""
    ds_castrado: str = ""

    # Espécie
    id_especie: int | None = None
    nm_especie: str = ""

    # Raça
    id_raca: int | None = None
    nm_raca: str = ""
    tp_porte: str = ""

    # Consulta
    id_consulta: int = 0
    dt_hora: datetime | None = None
    tp_modalidade: str = ""
    ds_motivo: str = ""
    ds_sintomas: str = ""
    ds_obs_consulta: str = ""
    kg_peso_consulta: float | None = None

    # Bem-Estar
    nr_idade: float | None = None
    kg_peso_bem_estar: float | None = None
    ds_apetite: str = ""
    ds_atividade: str = ""
    ds_comportamento: str = ""
    ds_obs_bem_estar: str = ""

    # Predisposições Genéticas
    predisposicoes: list[dict[str, str]] = field(default_factory=list)

    # Propriedades derivadas

    @property
    def peso_efetivo_kg(self) -> float | None:
        """Peso da consulta ou da avaliação de bem-estar como fallback."""
        return self.kg_peso_consulta or self.kg_peso_bem_estar

    def to_clinical_summary(self) -> str:
        """Renderiza resumo clínico textual para injeção no prompt da LLM."""
        _SEXO = {"M": "Macho", "F": "Fêmea"}
        _CASTRADO = {"S": "Castrado(a)", "N": "Inteiro(a)"}

        if self.predisposicoes:
            pred_lines = []
            for doenca in self.predisposicoes:
                nm = doenca.get("nm_doenca", "Desconhecida")
                ds = doenca.get("ds_doenca") or "Sem descrição disponível."
                # Trunca descrições muito longas para não exceder a janela de contexto
                ds_truncated = ds[:400] + "..." if len(ds) > 400 else ds
                pred_lines.append(f"  • {nm}: {ds_truncated}")
            predisposicoes_block = "\n".join(pred_lines)
        else:
            predisposicoes_block = "  Nenhuma predisposição genética mapeada para esta raça/espécie."

        welfare_items = {
            "Apetite": self.ds_apetite,
            "Atividade": self.ds_atividade,
            "Comportamento": self.ds_comportamento,
        }
        welfare_lines = [
            f"  {k}: {v}" for k, v in welfare_items.items() if v
        ] or ["  Avaliação de bem-estar não registrada nesta consulta."]

        peso_str = f"{self.peso_efetivo_kg:.2f} kg" if self.peso_efetivo_kg else "Não informado"
        idade_str = f"{self.nr_idade:.1f} anos" if self.nr_idade else "Não informada"
        dt_str = self.dt_hora.strftime("%d/%m/%Y %H:%M") if self.dt_hora else "Não informada"

        return (
            "=== DADOS DO PACIENTE ===\n"
            f"Nome:           {self.nm_animal}\n"
            f"Espécie:        {self.nm_especie}\n"
            f"Raça:           {self.nm_raca or 'SRD / Não informada'}"
            f" | Porte: {self.tp_porte or 'Não informado'}\n"
            f"Sexo:           {_SEXO.get(self.ds_sexo, self.ds_sexo)}\n"
            f"Status reprod.: {_CASTRADO.get(self.ds_castrado, self.ds_castrado)}\n"
            f"Idade estimada: {idade_str}\n"
            f"Peso:           {peso_str}\n"
            "\n=== DADOS DA CONSULTA ===\n"
            f"ID Consulta:    {self.id_consulta}\n"
            f"Data/Hora:      {dt_str}\n"
            f"Modalidade:     {self.tp_modalidade}\n"
            f"Motivo:         {self.ds_motivo or 'Não informado'}\n"
            f"Sintomas:       {self.ds_sintomas or 'Não descritos'}\n"
            f"Observações:    {self.ds_obs_consulta or 'Sem observações'}\n"
            "\n=== AVALIAÇÃO DE BEM-ESTAR (mais recente) ===\n"
            + "\n".join(welfare_lines)
            + (f"\n  Observações: {self.ds_obs_bem_estar}" if self.ds_obs_bem_estar else "")
            + "\n\n=== PREDISPOSIÇÕES GENÉTICAS DA RAÇA/ESPÉCIE ===\n"
            + predisposicoes_block
        )


# Função Principal de Extração


def fetch_clinical_data(conn: oracledb.Connection, id_consulta: int) -> ClinicalContext:
    """
    Executa as duas queries e retorna ClinicalContext populado.

    Raises:
        ValueError: Nenhuma consulta encontrada para o ID.
        oracledb.DatabaseError: Erro de banco de dados.
    """
    ctx = ClinicalContext(id_consulta=id_consulta)

    # Query 1: Dados Clínicos Principais
    logger.info("Executando CLINICAL_DATA_QUERY para ID_CONSULTA=%d", id_consulta)
    with conn.cursor() as cur:
        cur.execute(CLINICAL_DATA_QUERY, {"id_consulta": id_consulta})
        columns: list[str] = [col[0].lower() for col in cur.description]
        row: tuple[Any, ...] | None = cur.fetchone()

    if row is None:
        raise ValueError(
            f"Nenhuma consulta encontrada para ID_CONSULTA={id_consulta}. "
            "Verifique se o registro existe e se o usuário possui acesso SELECT."
        )

    row_dict: dict[str, Any] = dict(zip(columns, row))
    logger.debug("Dados clínicos retornados: %s", list(row_dict.keys()))

    # Mapeamento do resultado para o dataclass
    ctx.id_animal = int(row_dict["id_animal"] or 0)
    ctx.nm_animal = str(row_dict.get("nm_animal") or "")
    ctx.ds_sexo = str(row_dict.get("ds_sexo") or "")
    ctx.ds_castrado = str(row_dict.get("ds_castrado") or "")

    ctx.id_especie = row_dict.get("id_especie")
    ctx.nm_especie = str(row_dict.get("nm_especie") or "")

    ctx.id_raca = row_dict.get("id_raca")
    ctx.nm_raca = str(row_dict.get("nm_raca") or "")
    ctx.tp_porte = str(row_dict.get("tp_porte") or "")

    ctx.dt_hora = row_dict.get("dt_hora")  # datetime | None
    ctx.tp_modalidade = str(row_dict.get("tp_modalidade") or "")
    ctx.ds_motivo = str(row_dict.get("ds_motivo") or "")
    ctx.ds_sintomas = str(row_dict.get("ds_sintomas") or "")
    ctx.ds_obs_consulta = str(row_dict.get("ds_obs_consulta") or "")
    ctx.kg_peso_consulta = _safe_float(row_dict.get("kg_peso_consulta"))

    ctx.nr_idade = _safe_float(row_dict.get("nr_idade"))
    ctx.kg_peso_bem_estar = _safe_float(row_dict.get("kg_peso_bem_estar"))
    ctx.ds_apetite = str(row_dict.get("ds_apetite") or "")
    ctx.ds_atividade = str(row_dict.get("ds_atividade") or "")
    ctx.ds_comportamento = str(row_dict.get("ds_comportamento") or "")
    ctx.ds_obs_bem_estar = str(row_dict.get("ds_obs_bem_estar") or "")

    # Query 2: Predisposições Genéticas
    if ctx.id_especie is not None:
        logger.info(
            "Executando PREDISPOSITION_QUERY | ID_ESPECIE=%s | ID_RACA=%s",
            ctx.id_especie,
            ctx.id_raca,
        )
        with conn.cursor() as cur:
            cur.execute(
                PREDISPOSITION_QUERY,
                {"id_especie": ctx.id_especie, "id_raca": ctx.id_raca},
            )
            pred_columns = [col[0].lower() for col in cur.description]
            for pred_row in cur.fetchall():
                pred_dict = dict(zip(pred_columns, pred_row))
                ctx.predisposicoes.append(
                    {k: str(v) if v is not None else "" for k, v in pred_dict.items()}
                )

        logger.info(
            "%d predisposição(ões) genética(s) encontrada(s) para a raça/espécie.",
            len(ctx.predisposicoes),
        )
    else:
        logger.warning("ID_ESPECIE é NULL — pulando query de predisposições.")

    return ctx


def _safe_float(value: Any) -> float | None:
    """Converte valor para float; retorna None em falha."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None