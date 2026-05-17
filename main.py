"""
main.py
=======
Ponto de entrada do Motor de Inteligência Clínica Veterinária ArkIve.

Uso:
    python main.py <ID_CONSULTA>

Exemplo:
    python main.py 42

Saída:
    JSON formatado com o diagnóstico clínico estruturado, impresso no stdout.
    Erros e logs de execução são direcionados ao stdout (nível INFO por padrão).
"""

from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    """
    Ponto de entrada principal. Retorna 0 em sucesso, 1 em erro.
    O exit code pode ser capturado por orquestradores (Java, Docker, CI/CD).
    """

    # ── Parsing do argumento de entrada ──────────────────────────────────────
    if len(sys.argv) < 2:
        print(
            "Uso: python main.py <ID_CONSULTA>\n"
            "Exemplo: python main.py 42",
            file=sys.stderr,
        )
        return 1

    try:
        id_consulta = int(sys.argv[1])
        if id_consulta <= 0:
            raise ValueError("ID deve ser um inteiro positivo.")
    except ValueError as exc:
        print(
            f"Erro: ID_CONSULTA inválido — '{sys.argv[1]}'. {exc}",
            file=sys.stderr,
        )
        return 1

    # ── Validação de configuração (fail-fast) ────────────────────────────────
    try:
        # O import de config configura o logging antes de qualquer outro módulo.
        from config import validate_config
        validate_config()
    except RuntimeError as exc:
        print(f"Erro de configuração: {exc}", file=sys.stderr)
        return 1

    # ── Execução do Motor de Inteligência ────────────────────────────────────
    try:
        from agents.clinical_agent import ClinicalIntelligenceEngine

        engine = ClinicalIntelligenceEngine()
        result: dict = engine.analyze(id_consulta=id_consulta)

    except ValueError as exc:
        # Consulta não encontrada no banco
        logger.error("Consulta não encontrada: %s", exc)
        error_payload = {
            "error": "CONSULTA_NAO_ENCONTRADA",
            "message": str(exc),
            "id_consulta": id_consulta,
        }
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    except RuntimeError as exc:
        # Falha irrecuperável na LLM
        logger.error("Falha no motor de IA: %s", exc)
        error_payload = {
            "error": "FALHA_MOTOR_IA",
            "message": str(exc),
            "id_consulta": id_consulta,
        }
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    except Exception as exc:
        # Erro inesperado — logar com traceback completo
        logger.exception("Erro inesperado durante a análise clínica: %s", exc)
        error_payload = {
            "error": "ERRO_INESPERADO",
            "message": str(exc),
            "id_consulta": id_consulta,
        }
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    # ── Saída do Resultado ───────────────────────────────────────────────────
    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("── Análise clínica concluída com sucesso. ──")
    return 0


if __name__ == "__main__":
    sys.exit(main())
