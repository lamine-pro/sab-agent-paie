"""
Pipeline d'ingestion complet.
Coordonne : upload MinIO → ingestion_log → extraction → raw_text
"""

import uuid
from datetime import datetime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from storage import MinIOStorage
from trino_client import TrinoClient
import sys
import argparse
from pathlib import Path

load_dotenv()

BUCKET = "raw"


class IngestionPipeline:

    def __init__(self):
        self.storage = MinIOStorage()
        self.trino   = TrinoClient()

    # ── Point d'entrée principal ──────────────────────────────────
    def ingest_file(
        self,
        file_path:     str | Path,
        source_system: str = "manual",
    ) -> dict:
        """
        Ingère un fichier complet.
        Retourne un dict avec le résultat de l'opération.
        """
        file_path = Path(file_path)
        filename  = file_path.name
        file_type = file_path.suffix.lstrip(".").lower()

        logger.info(f"━━━ Ingestion : {filename} ━━━")

        # ── 2. Calcul du hash (déduplication) ─────────────────────
        file_hash = self.storage.compute_hash(file_path)
        logger.info(f"  Hash SHA-256 : {file_hash[:16]}...")

        if self._is_duplicate(file_hash):
            logger.warning(f"  Fichier déjà ingéré (hash connu) — skip")
            return {"status": "skipped", "reason": "duplicate", "filename": filename}

        # ── 3. Upload vers MinIO Bronze ───────────────────────────
        ingestion_id = str(uuid.uuid4())
        object_key   = self.storage.build_object_key(filename, file_type)
        s3_path      = self.storage.upload_file(file_path, BUCKET, object_key)
        file_size    = file_path.stat().st_size

        # ── 4. Enregistrement dans ingestion_log ──────────────────
        self._log_ingestion(
            ingestion_id=ingestion_id,
            filename=filename,
            file_path=s3_path,
            file_type=file_type,
            file_size=file_size,
            file_hash=file_hash,
            source_system=source_system,
            status="pending",
        )
        logger.info(f"  ingestion_log → id={ingestion_id[:8]}...")

 # ── Étapes internes ───────────────────────────────────────────

    def _is_duplicate(self, file_hash: str) -> bool:
        """Vérifie si un fichier avec ce hash existe déjà."""
        rows = self.trino.execute(
            f"""
            SELECT COUNT(*) FROM minio.raw.ingestion_log
            WHERE file_hash = '{file_hash}'
              AND status != 'failed'
            """,
            fetch=True,
        )
        return rows[0][0] > 0 if rows else False
    
    def _log_ingestion(
        self,
        ingestion_id: str,
        filename:     str,
        file_path:    str,
        file_type:    str,
        file_size:    int,
        file_hash:    str,
        source_system: str,
        status:       str,
    ):
        """Insère une ligne dans bronze.ingestion_log."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.trino.execute(f"""
            INSERT INTO iceberg.raw.ingestion_log VALUES (
                '{ingestion_id}',
                '{filename}',
                '{file_path}',
                '{file_type}',
                {file_size},
                '{file_hash}',
                '{source_system}',
                TIMESTAMP '{now}',
                '{status}'
            )
        """)

    
def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(
        "logs/ingestion_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )

def ingest_path(path: Path, source: str, pipeline: IngestionPipeline):
    """Ingère un fichier ou tous les fichiers d'un dossier."""
    supported = {".pdf", ".docx", ".doc", ".csv"}

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [f for f in path.rglob("*") if f.suffix.lower() in supported]
        logger.info(f"{len(files)} fichier(s) trouvé(s) dans {path}")
    else:
        logger.error(f"Chemin introuvable : {path}")
        sys.exit(1)

    results = {"success": 0, "skipped": 0, "failed": 0}

    for file in files:
        try:
            result = pipeline.ingest_file(file, source_system=source)
            results[result["status"]] = results.get(result["status"], 0) + 1
        except Exception as e:
            logger.error(f"Échec sur {file.name} : {e}")
            results["failed"] += 1

    logger.info(
        f"\nRésultat : "
        f"{results['success']} OK | "
        f"{results['skipped']} skippés | "
        f"{results['failed']} échoués"
    )

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Pipeline d'ingestion RAG")
    parser.add_argument("path",   help="Fichier ou dossier à ingérer")
    parser.add_argument("--source", default="manual", help="Système source")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    ingest_path(Path(args.path), args.source, pipeline)

if __name__ == "__main__":
    main()