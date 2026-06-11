"""
Point d'entrée du pipeline de processing.
Usage : python run_processing.py [--batch 10]
"""

import sys
import argparse
from loguru import logger
from dotenv import load_dotenv

from ingestion.storage import MinIOStorage
from lakehouse.trino_client import TrinoClient
from processing.document_processor import ProcessingPipeline

#load_dotenv()


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(
        "logs/processing_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Pipeline de processing RAG")
    parser.add_argument("--batch", type=int, default=10, help="Nb fichiers par run")
    args = parser.parse_args()

    pipeline = ProcessingPipeline(
        storage=MinIOStorage(),
        trino=TrinoClient(),
    )
    pipeline.run(batch_size=args.batch)


if __name__ == "__main__":
    main()