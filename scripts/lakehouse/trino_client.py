"""
Client Trino réutilisable avec retry, logging et gestion d erreurs.
"""

import trino
import logging
import time
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class TrinoClient:
    def __init__(self):
        self.host    = os.getenv("TRINO_HOST", "localhost")
        self.port    = int(os.getenv("TRINO_PORT", "8080"))
        self.catalog = "iceberg"
        self.user    = "admin"

    def _get_connection(self, schema: str = None):
        return trino.dbapi.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            catalog=self.catalog,
            schema=schema,
            http_scheme="http",
        )

    def execute(
        self,
        sql: str,
        schema: str = None,
        fetch: bool = False,
        retries: int = 3,
    ) -> list[Any] | None:
        """
        Exécute une requête SQL.
        - fetch=True  → retourne les résultats
        - fetch=False → DDL/DML sans retour
        """
        for attempt in range(1, retries + 1):
            try:
                conn   = self._get_connection(schema)
                cursor = conn.cursor()
                cursor.execute(sql)
                if fetch:
                    return cursor.fetchall()
                return None
            except Exception as e:
                logger.warning(f"Tentative {attempt}/{retries} échouée : {e}")
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)  # backoff exponentiel

    def execute_many(self, statements: list[str], schema: str = None):
        """Exécute une liste de requêtes DDL dans l ordre."""
        for i, sql in enumerate(statements, 1):
            preview = sql.strip()[:60].replace("\n", " ")
            logger.info(f"[{i}/{len(statements)}] {preview}...")
            self.execute(sql, schema=schema)