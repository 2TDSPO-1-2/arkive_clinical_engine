"""
database/connection.py
======================
Fábrica de conexões Oracle em modo Thin (sem Oracle Instant Client).

Garante leitura segura via: autocommit=False, rollback() no finally, e
oracledb.defaults.fetch_lobs=False (converte CLOBs em str automaticamente).
"""

import logging
from contextlib import contextmanager
from typing import Generator

import oracledb

from config import ORACLE_DSN, ORACLE_PASSWORD, ORACLE_USER

logger = logging.getLogger(__name__)

# Converte CLOB/BLOB para str/bytes diretamente no fetch (deve ser chamado antes da primeira conexão)
oracledb.defaults.fetch_lobs = False


@contextmanager
def get_connection() -> Generator[oracledb.Connection, None, None]:
    """
    Context manager que fornece conexão Oracle Thin em modo leitura.

    Raises:
        oracledb.DatabaseError: Se a conexão falhar.
    """
    conn: oracledb.Connection | None = None
    try:
        logger.debug("Iniciando conexão Oracle Thin | DSN: %s | User: %s", ORACLE_DSN, ORACLE_USER)

        conn = oracledb.connect(
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            dsn=ORACLE_DSN,
        )

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
                conn.rollback()
            except Exception:
                pass
            conn.close()
            logger.debug("Conexão Oracle encerrada.")