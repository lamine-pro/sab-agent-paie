"""
Pipeline d'embedding et d'indexation Pinecone.
Étape 3 du RAG Lakehouse.

Flux :
  1. Lecture des documents 'pending' dans iceberg.processed.documents
  2. Chunking hybride (sections Markdown → chunks fixes si trop long)
  3. Embeddings via Mistral (mistral-embed)
  4. Upsert dans Pinecone
  5. UPDATE indexing_status → 'indexed' dans processed.documents

Prérequis .env :
  MISTRAL_API_KEY=...
  PINECONE_API_KEY=...
  PINECONE_INDEX_NAME=...
"""

import os
import re
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger
from dotenv import load_dotenv

# ── LangChain ─────────────────────────────────────────────────
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# ── Pinecone ──────────────────────────────────────────────────
from pinecone import Pinecone

load_dotenv(dotenv_path="/agent_paie/docker/.env")


# ─────────────────────────────────────────────────────────────
# Modèle de données
# ─────────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    """Résultat du chunking d'un document."""
    ingestion_id:  str
    filename:      str
    chunks:        list[Document]
    total_chars:   int
    chunk_count:   int


# ─────────────────────────────────────────────────────────────
# Chunker hybride : Markdown sections → RecursiveCharacter
# ─────────────────────────────────────────────────────────────

class HybridChunker:
    """
    Stratégie hybride :
      1. Découpe sur les titres Markdown (#, ##, ###)
      2. Si un chunk dépasse MAX_CHUNK_SIZE → sous-découpe avec
         RecursiveCharacterTextSplitter
    """

    MAX_CHUNK_SIZE = 1000   # chars par chunk final
    CHUNK_OVERLAP  = 150    # overlap entre chunks fixes

    def __init__(self):
        # Splitter niveau 1 : découpe par titres Markdown
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#",   "section"),
                ("##",  "subsection"),
                ("###", "subsubsection"),
            ],
            strip_headers=False,
        )

        # Splitter niveau 2 : chunks fixes pour les sections trop longues
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.MAX_CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, clean_text: str, base_metadata: dict) -> list[Document]:
        """
        Retourne une liste de Documents LangChain avec métadonnées enrichies.
        """
        logger.info(f"  [Chunker] Découpe Markdown du texte ({len(clean_text)} chars)...")

        # ── Étape 1 : découpe par sections Markdown ───────────
        md_chunks = self.md_splitter.split_text(clean_text)
        logger.info(f"  [Chunker] {len(md_chunks)} section(s) Markdown identifiée(s)")

        # ── Étape 2 : sous-découpe des sections trop longues ──
        final_chunks = []
        oversized    = 0

        for md_chunk in md_chunks:
            content = md_chunk.page_content if hasattr(md_chunk, "page_content") else str(md_chunk)

            if len(content) > self.MAX_CHUNK_SIZE:
                oversized += 1
                sub_chunks = self.char_splitter.split_text(content)
                for sub in sub_chunks:
                    final_chunks.append(sub)
            else:
                final_chunks.append(content)

        if oversized > 0:
            logger.info(f"  [Chunker] {oversized} section(s) sous-découpée(s) (> {self.MAX_CHUNK_SIZE} chars)")

        # ── Étape 3 : construction des Documents LangChain ────
        documents = []
        for i, chunk_text in enumerate(final_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # ID déterministe basé sur le contenu
            chunk_id = self._make_chunk_id(base_metadata.get("ingestion_id", ""), i, chunk_text)

            doc = Document(
                page_content=chunk_text,
                metadata={
                    **base_metadata,
                    "chunk_index":  i,
                    "chunk_count":  len(final_chunks),
                    "chunk_id":     chunk_id,
                    "char_count":   len(chunk_text),
                },
            )
            documents.append(doc)

        logger.info(
            f"  [Chunker] Chunking OK → {len(documents)} chunks "
            f"(moy. {sum(len(d.page_content) for d in documents) // max(len(documents), 1)} chars/chunk)"
        )
        return documents

    @staticmethod
    def _make_chunk_id(ingestion_id: str, index: int, text: str) -> str:
        """Génère un ID stable et unique pour chaque chunk."""
        raw = f"{ingestion_id}_{index}_{text[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# Embedding + Indexation Pinecone
# ─────────────────────────────────────────────────────────────

class PineconeIndexer:
    """
    Calcule les embeddings (mistral-embed) et upsert dans Pinecone.
    """

    EMBEDDING_MODEL = "mistral-embed"
    BATCH_SIZE      = 50   # nb de chunks par appel embed + upsert

    def __init__(self):
        # Embeddings Mistral
        self.embeddings = MistralAIEmbeddings(
            model=self.EMBEDDING_MODEL,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )

        # Connexion Pinecone
        api_key    = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")

        if not api_key or not index_name:
            raise ValueError("PINECONE_API_KEY et PINECONE_INDEX_NAME requis dans .env")

        logger.info(f"  [Pinecone] Connexion à l'index '{index_name}'...")
        pc          = Pinecone(api_key=api_key)
        self.index  = pc.Index(index_name)
        self.index_name = index_name

        # Stats index
        stats = self.index.describe_index_stats()
        logger.info(
            f"  [Pinecone] Index OK → "
            f"{stats.get('total_vector_count', 0)} vecteurs existants | "
            f"dimension={stats.get('dimension', '?')}"
        )

    def index_documents(self, documents: list[Document]) -> int:
        """
        Embed + upsert les documents dans Pinecone.
        Retourne le nombre de chunks indexés.
        """
        if not documents:
            logger.warning("  [Pinecone] Aucun document à indexer")
            return 0

        logger.info(
            f"  [Pinecone] Début indexation : {len(documents)} chunks "
            f"en batches de {self.BATCH_SIZE}..."
        )

        total_indexed = 0
        batches       = [
            documents[i:i + self.BATCH_SIZE]
            for i in range(0, len(documents), self.BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(batches, 1):
            logger.info(
                f"  [Pinecone] Batch {batch_num}/{len(batches)} "
                f"({len(batch)} chunks)..."
            )

            # ── Calcul embeddings ──────────────────────────────
            texts = [doc.page_content for doc in batch]
            logger.debug(f"  [Pinecone] Calcul embeddings (mistral-embed)...")
            vectors = self.embeddings.embed_documents(texts)
            logger.debug(f"  [Pinecone] {len(vectors)} vecteurs calculés (dim={len(vectors[0])})")

            # ── Préparation des vecteurs Pinecone ──────────────
            pinecone_vectors = []
            for doc, vector in zip(batch, vectors):
                # Pinecone n'accepte que des valeurs scalaires en metadata
                meta = self._sanitize_metadata(doc.metadata)
                meta["text"] = doc.page_content  # texte stocké pour la récupération

                pinecone_vectors.append({
                    "id":     doc.metadata["chunk_id"],
                    "values": vector,
                    "metadata": meta,
                })

            # ── Upsert ────────────────────────────────────────
            self.index.upsert(vectors=pinecone_vectors)
            total_indexed += len(batch)
            logger.info(f"  [Pinecone] Batch {batch_num} upsert OK ({total_indexed}/{len(documents)} chunks)")

        logger.info(f"  [Pinecone] Indexation terminée → {total_indexed} chunks indexés")
        return total_indexed

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """
        Pinecone n'accepte que str, int, float, bool en metadata.
        Convertit les listes et autres types.
        """
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = json.dumps(v, ensure_ascii=False)
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean


# ─────────────────────────────────────────────────────────────
# Embedding Pipeline — orchestrateur principal
# ─────────────────────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Orchestre :
      processed.documents (pending) → chunking → embedding → Pinecone → indexed
    """

    PENDING_STATUS = "pending"
    INDEXED_STATUS = "indexed"
    FAILED_STATUS  = "failed"

    def __init__(self, trino):
        self.trino   = trino
        self.chunker = HybridChunker()
        self.indexer = PineconeIndexer()

    # ── Point d'entrée ────────────────────────────────────────

    def run(self, batch_size: int = 10) -> dict:
        logger.info(f"━━━ Démarrage Embedding Pipeline (batch={batch_size}) ━━━")

        docs = self._fetch_pending(batch_size)

        if not docs:
            logger.info("  Aucun document pending trouvé — pipeline terminé.")
            return {"success": 0, "failed": 0, "total_chunks": 0}

        logger.info(f"  {len(docs)} document(s) à indexer")
        results = {"success": 0, "failed": 0, "total_chunks": 0}

        for i, row in enumerate(docs, 1):
            ingestion_id = row[0]
            filename     = row[1]
            clean_text   = row[2]
            metadata     = self._parse_row_metadata(row)

            logger.info(f"\n  [{i}/{len(docs)}] ─── {filename} ───")

            try:
                chunk_count = self._process_one(
                    ingestion_id=ingestion_id,
                    filename=filename,
                    clean_text=clean_text,
                    metadata=metadata,
                )
                results["success"]      += 1
                results["total_chunks"] += chunk_count

            except Exception as e:
                logger.error(f"  ✗ Échec {filename} : {e}")
                self._update_indexing_status(ingestion_id, self.FAILED_STATUS)
                results["failed"] += 1

        logger.info(
            f"\n━━━ Résultat final : "
            f"{results['success']} indexé(s) | "
            f"{results['failed']} échoué(s) | "
            f"{results['total_chunks']} chunks total ━━━"
        )
        return results

    # ── Étapes internes ───────────────────────────────────────

    def _fetch_pending(self, limit: int) -> list:
        """Récupère les documents non encore indexés."""
        logger.info(f"  Recherche des documents à indexer (indexing_status='{self.PENDING_STATUS}')...")
        rows = self.trino.execute(
            f"""
            SELECT
                ingestion_id, filename, clean_text,
                title, summary, domain, doc_type,
                language, tags, key_entities,
                page_count, file_type
            FROM iceberg.process.documents
            WHERE indexing_status = '{self.PENDING_STATUS}'
            ORDER BY processed_at ASC
            LIMIT {limit}
            """,
            fetch=True,
        )
        count = len(rows) if rows else 0
        logger.info(f"  {count} document(s) trouvé(s) avec indexing_status='{self.PENDING_STATUS}'")
        return rows or []

    def _parse_row_metadata(self, row: tuple) -> dict:
        """Construit le dict de métadonnées depuis une ligne Trino."""
        return {
            "ingestion_id": row[0],
            "filename":     row[1],
            "title":        row[3]  or "",
            "summary":      row[4]  or "",
            "domain":       row[5]  or "other",
            "doc_type":     row[6]  or "other",
            "language":     row[7]  or "fr",
            "tags":         row[8]  or "[]",
            "key_entities": row[9]  or "[]",
            "page_count":   row[10] or 0,
            "file_type":    row[11] or "",
        }

    def _process_one(
        self,
        ingestion_id: str,
        filename:     str,
        clean_text:   str,
        metadata:     dict,
    ) -> int:
        """Traite un document : chunking → embedding → Pinecone → update statut."""

        # ── 1. Chunking ────────────────────────────────────────
        chunks = self.chunker.chunk(clean_text, metadata)

        if not chunks:
            logger.warning(f"  [Chunker] Aucun chunk produit pour {filename} — skip")
            self._update_indexing_status(ingestion_id, self.FAILED_STATUS)
            return 0

        # ── 2. Embedding + Upsert Pinecone ─────────────────────
        indexed_count = self.indexer.index_documents(chunks)

        # ── 3. UPDATE indexing_status → indexed ────────────────
        logger.info(f"  [Trino] Mise à jour indexing_status → {self.INDEXED_STATUS}")
        self._update_indexing_status(ingestion_id, self.INDEXED_STATUS, chunk_count=indexed_count)
        logger.info(f"  ✓ {filename} indexé avec succès ({indexed_count} chunks)")

        return indexed_count

    def _update_indexing_status(
        self,
        ingestion_id:  str,
        status:        str,
        chunk_count:   int = 0,
    ):
        """Met à jour indexing_status et indexed_chunks_count dans processed.documents."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000000")
        self.trino.execute(f"""
            UPDATE iceberg.process.documents
            SET
                indexing_status       = CAST('{status}' AS VARCHAR),
                indexed_chunks_count  = CAST({chunk_count} AS INTEGER),
                indexed_at            = CAST('{now}' AS TIMESTAMP)
            WHERE ingestion_id = '{ingestion_id}'
        """)
        logger.info(
            f"  [Trino] indexing_status → {status} | "
            f"chunks={chunk_count} | id={ingestion_id[:8]}..."
        )