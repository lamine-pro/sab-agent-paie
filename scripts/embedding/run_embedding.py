"""
Point d'entrée du pipeline d'embedding.
Usage : python run_embedding.py [--batch 10]
"""

import sys
import argparse
from loguru import logger
from dotenv import load_dotenv

from lakehouse.trino_client import TrinoClient
from embedding.embedding import EmbeddingPipeline

def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(
        "logs/embedding_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Pipeline d'embedding RAG")
    parser.add_argument("--batch", type=int, default=10, help="Nb documents par run")
    args = parser.parse_args()

    pipeline = EmbeddingPipeline(trino=TrinoClient())
    pipeline.run(batch_size=args.batch)


if __name__ == "__main__":
    main()