"""
config.py
=========
Centraliza o carregamento e a validação de todas as configurações do
Motor de Inteligência Clínica Veterinária ArkIve.

Lê variáveis do arquivo .env via python-dotenv; falha rapidamente
(fail-fast) se credenciais críticas estiverem ausentes.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env a partir do diretório do projeto (sobe um nível se necessário)
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# ── Logging ──────────────────────────────────────────────────────────────────

_LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL: int = getattr(logging, _LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# ── Oracle Database ───────────────────────────────────────────────────────────

ORACLE_DSN: str = os.getenv("ORACLE_DSN", "localhost:1521/XEPDB1")
ORACLE_USER: str = os.getenv("ORACLE_USER", "")
ORACLE_PASSWORD: str = os.getenv("ORACLE_PASSWORD", "")

# ── GROQ (Free Tier) ─────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE: float = 0.10

# ── Limiar de Ambiguidade ─────────────────────────────────────────────────────

#: Se a IA avaliar confiança local < este valor, a busca web é acionada.
AMBIGUITY_THRESHOLD: int = int(os.getenv("AMBIGUITY_THRESHOLD", "60"))

# ── Validação Fail-Fast ───────────────────────────────────────────────────────

def validate_config() -> None:
    missing: list[str] = []
    if not ORACLE_USER:
        missing.append("ORACLE_USER")
    if not ORACLE_PASSWORD:
        missing.append("ORACLE_PASSWORD")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError(
            f"Defina as variáveis no .env: {', '.join(missing)}"
        )
    logger.info(
        "Configuração validada | Oracle DSN: %s | Modelo: %s",
        ORACLE_DSN,
        GROQ_MODEL,
    )
