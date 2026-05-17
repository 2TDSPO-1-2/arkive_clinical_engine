"""
database/connection.py
======================
Fábrica de conexões Oracle em modo Thin (sem Oracle Instant Client).

Política de leitura:
  - O usuário DB deve possuir apenas privilégios SELECT (enforçado pelo DBA).
  - autocommit=False é definido explicitamente como barreira adicional.
  - Um rollback() é chamado no finally para garantir que nenhuma transação
    pendente acidental persista.
  - oracledb.defaults.fetch_lobs = False converte CLOBs automaticamente
    em str Python, evitando estouro de memória com LOBs grandes.

Referência: https://python-oracledb.readthedocs.io/en/latest/user_guide/lob_data.html
"""

import logging
from contextlib import contextmanager
from typing import Generator

import oracledb

from config import ORACLE_DSN, ORACLE_PASSWORD, ORACLE_USER

logger = logging.getLogger(__name__)

# ── Configuração Global de LOBs ───────────────────────────────────────────────
# Converte CLOB/BLOB para str/bytes diretamente no fetch, sem alocar objetos LOB.
# Isso é seguro para campos de tamanho razoável (< 1 GB).
# Deve ser chamado antes de qualquer conexão ser estabelecida.
oracledb.defaults.fetch_lobs = False


@contextmanager
def get_connection() -> Generator[oracledb.Connection, None, None]:
    """
    Context manager que fornece uma conexão Oracle Thin em modo leitura.

    Uso::

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
                rows = cur.fetchall()

    Raises:
        oracledb.DatabaseError: Se a conexão com o banco falhar.
    """
    conn: oracledb.Connection | None = None
    try:
        logger.debug("Iniciando conexão Oracle Thin | DSN: %s | User: %s", ORACLE_DSN, ORACLE_USER)

        conn = oracledb.connect(
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            dsn=ORACLE_DSN,
            # thin=True é o padrão em oracledb >= 2.0; explicitado por clareza.
        )

        # Barreira explícita: sem auto-commit, sem DML.
        conn.autocommit = False

        logger.debug("Conexão Oracle estabelecida com sucesso (READ-ONLY).")
        yield conn

    except oracledb.DatabaseError as exc:
        (error,) = exc.args
        logger.error(
            "Falha ao conectar ao Oracle | Código: %s | Mensagem: %s",
            getattr(error, "code", "N/A"),
            getattr(error, "message", str(exc)),
        )
        raise

    finally:
        if conn:
            try:
                # Desfaz qualquer transação pendente acidental (ex: SELECT FOR UPDATE).
                conn.rollback()
            except Exception:
                pass
            conn.close()
            logger.debug("Conexão Oracle encerrada.")
