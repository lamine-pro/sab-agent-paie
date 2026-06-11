"""
Pipeline de processing des documents — compatible mistralai SDK v2.x
Étape 2 du RAG Lakehouse : extraction OCR, nettoyage, enrichissement métadonnées.

Flux :
  1. Lecture des fichiers 'pending' dans iceberg.raw.ingestion_log
  2. Téléchargement depuis MinIO
  3. Mistral OCR → cleaning → metadata enrichment
  4. INSERT dans iceberg.processed.documents
  5. UPDATE statut ingestion_log : pending → processed

Compatibilité SDK :
  mistralai >= 2.0  →  from mistralai.client.sdk import Mistral
                        client.files.upload(file=models.File(...))
                        client.ocr.process(model=..., document=...)
"""

import re
import os
import json
import unicodedata
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from loguru import logger
from dotenv import load_dotenv

# ── SDK Mistral v2 ────────────────────────────────────────────
from mistralai.client.sdk import Mistral
from mistralai.client import models as mistral_models

# ── LangChain + Mistral pour l'enrichissement ─────────────────
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="/agent_paie/docker/.env")


# ─────────────────────────────────────────────────────────────
# Modèles de données
# ─────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    title:        str       = Field(description="Titre du document")
    summary:      str       = Field(description="Résumé factuel en 2-3 phrases")
    domain:       str       = Field(description="Domaine : legal, hr, finance, technical, other")
    tags:         list[str] = Field(description="Tags thématiques (5 max)")
    language:     str       = Field(description="Langue détectée : fr, en, other")
    doc_type:     str       = Field(description="Type : contract, report, law, policy, manual, other")
    key_entities: list[str] = Field(description="Entités clés : organisations, lois, dates importantes")


@dataclass
class ProcessedDocument:
    ingestion_id:      str
    filename:          str
    file_type:         str
    raw_markdown:      str
    clean_text:        str
    metadata:          dict
    page_count:        int
    char_count:        int
    processing_status: str = "success"
    error_message:     Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Mistral OCR Extractor — SDK v2
# ─────────────────────────────────────────────────────────────

class MistralOCRExtractor:
    """
    Extraction via Mistral OCR API (SDK v2.x).
    Upload → Signed URL → OCR → Suppression fichier distant.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(self):
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def extract(self, file_path: Path) -> tuple[str, int]:
        """Retourne (markdown, nb_pages)."""
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Format non supporté par Mistral OCR : {ext}")

        file_size_kb = file_path.stat().st_size // 1024
        logger.info(f"  [OCR] Début extraction : {file_path.name} ({file_size_kb} KB)")

        # ── 1. Upload ──────────────────────────────────────────
        logger.info(f"  [OCR] Upload vers Mistral...")
        print(os.getenv("MISTRAL_API_KEY"))
        with open(file_path, "rb") as f:
            upload_response = self.client.files.upload(
                file=mistral_models.File(
                    file_name=file_path.name,
                    content=f,
                ),
                purpose='ocr',
            )
        file_id = upload_response.id
        logger.info(f"  [OCR] Fichier uploadé → file_id={file_id}")

        # ── 2. Signed URL ──────────────────────────────────────
        logger.info(f"  [OCR] Récupération de l'URL signée...")
        signed = self.client.files.get_signed_url(file_id=file_id)
        logger.debug(f"  [OCR] Signed URL OK")

        # ── 3. Traitement OCR ──────────────────────────────────
        logger.info(f"  [OCR] Traitement OCR en cours (mistral-ocr-latest)...")
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document=mistral_models.DocumentURLChunk(
                document_url=signed.url,
            ),
        )

        # ── 4. Assemblage Markdown ─────────────────────────────
        pages      = ocr_response.pages if hasattr(ocr_response, "pages") else []
        markdown   = "\n\n---\n\n".join(
            p.markdown for p in pages if hasattr(p, "markdown") and p.markdown
        )
        page_count = len(pages)
        logger.info(f"  [OCR] Extraction OK → {page_count} pages | {len(markdown)} chars")

        # ── 5. Suppression du fichier distant ─────────────────
        try:
            self.client.files.delete(file_id=file_id)
            logger.info(f"  [OCR] Fichier distant supprimé : {file_id}")
        except Exception as e:
            logger.warning(f"  [OCR] Suppression fichier distant ignorée : {e}")

        return markdown, page_count


# ─────────────────────────────────────────────────────────────
# Text Cleaner
# ─────────────────────────────────────────────────────────────

class TextCleaner:
    """Nettoyage déterministe du Markdown issu de Mistral OCR."""

    BOILERPLATE_PATTERNS = [
        r"Page\s+\d+\s+(?:sur|of|/)\s+\d+",
        r"(?m)^\s*\d+\s*$",
        r"(?i)confidentiel\s*[-–]\s*usage\s+interne",
        r"(?i)(tous droits réservés|all rights reserved)",
        r"\f",
    ]

    def clean(self, markdown_text: str) -> str:
        logger.info(f"  [Cleaner] Nettoyage du texte ({len(markdown_text)} chars bruts)...")
        text = unicodedata.normalize("NFKC", markdown_text)

        # Suppression boilerplate
        removed = 0
        for pattern in self.BOILERPLATE_PATTERNS:
            before = len(text)
            text   = re.sub(pattern, "", text, flags=re.MULTILINE)
            removed += before - len(text)

        if removed > 0:
            logger.debug(f"  [Cleaner] Boilerplate supprimé : {removed} chars")

        # Caractères de contrôle + typographie + espaces
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        for orig, repl in {
            "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--", "\u00a0": " ",
        }.items():
            text = text.replace(orig, repl)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+([,;:!?.])", r"\1", text)
        text = text.strip()

        logger.info(f"  [Cleaner] Nettoyage OK → {len(text)} chars (gain : {len(markdown_text) - len(text)} chars supprimés)")
        return text


# ─────────────────────────────────────────────────────────────
# Metadata Enricher
# ─────────────────────────────────────────────────────────────

class MetadataEnricher:
    """Enrichissement via mistral-small (LangChain)."""

    SYSTEM_PROMPT = """Tu es un expert en analyse documentaire.
Analyse le texte fourni et retourne UNIQUEMENT un objet JSON valide avec ces champs :
{{
  "title": "titre du document",
  "summary": "résumé factuel en 2-3 phrases",
  "domain": "legal | hr | finance | technical | other",
  "tags": ["tag1", "tag2"],
  "language": "fr | en | other",
  "doc_type": "contract | report | law | policy | manual | other",
  "key_entities": ["entité1", "entité2"]
}}
Réponds UNIQUEMENT avec le JSON, sans markdown ni explication."""

    def __init__(self):
        self.llm    = ChatMistralAI(model="mistral-small-latest", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=DocumentMetadata)

    def enrich(self, text: str, filename: str) -> dict:
        sample_size = min(len(text), 3000)
        logger.info(f"  [Enricher] Extraction des métadonnées (échantillon : {sample_size} chars)...")

        sample = text[:1500] + "\n...\n" + text[-1500:] if len(text) > 3000 else text
        prompt = ChatPromptTemplate.from_messages([
                ("system", """Tu es un expert en analyse documentaire. {format_instructions} 
                 Réponds uniquement avec le JSON valide."""),
                ("human", "Fichier: {filename}\n\nTexte:\n{text}")
            ]).partial(
                format_instructions=self.parser.get_format_instructions()
            )
        try:
            result = (prompt | self.llm | self.parser).invoke(
                {"filename": filename, "text": sample}
            )
            logger.info(
                f"  [Enricher] Métadonnées extraites → "
                f"domain={result.get('domain')} | "
                f"type={result.get('doc_type')} | "
                f"lang={result.get('language')} | "
                f"tags={result.get('tags', [])}"
            )
            return result
        except Exception as e:
            logger.warning(f"  [Enricher] Fallback métadonnées minimales : {e}")
            return {
                "title": Path(filename).stem, "summary": "", "domain": "other",
                "tags": [], "language": "fr", "doc_type": "other", "key_entities": [],
            }


# ─────────────────────────────────────────────────────────────
# Processing Pipeline — orchestrateur principal
# ─────────────────────────────────────────────────────────────

class ProcessingPipeline:
    """
    Orchestre :
      pending (ingestion_log) → OCR → clean → enrich → processed.documents → processed
    """

    PENDING_STATUS   = "pending"
    PROCESSED_STATUS = "processed"
    FAILED_STATUS    = "failed"

    def __init__(self, storage, trino):
        self.storage  = storage
        self.trino    = trino
        self.ocr      = MistralOCRExtractor()
        self.cleaner  = TextCleaner()
        self.enricher = MetadataEnricher()

    # ── Point d'entrée ────────────────────────────────────────

    def run(self, batch_size: int = 10) -> dict:
        logger.info(f"━━━ Démarrage Processing Pipeline (batch={batch_size}) ━━━")

        pending = self._fetch_pending(batch_size)

        if not pending:
            logger.info("  Aucun fichier pending trouvé — pipeline terminé.")
            return {"success": 0, "failed": 0}

        logger.info(f"  {len(pending)} fichier(s) pending à traiter")
        results = {"success": 0, "failed": 0}

        for i, row in enumerate(pending, 1):
            ingestion_id, filename, s3_path, file_type = row[0], row[1], row[2], row[3]
            logger.info(f"\n  [{i}/{len(pending)}] ─── {filename} ───")
            try:
                self._process_one(ingestion_id, filename, s3_path, file_type)
                results["success"] += 1
            except Exception as e:
                logger.error(f"  ✗ Échec sur {filename} : {e}")
                self._update_status(ingestion_id, self.FAILED_STATUS)
                results["failed"] += 1

        logger.info(
            f"\n━━━ Résultat final : "
            f"{results['success']} traité(s) | "
            f"{results['failed']} échoué(s) ━━━"
        )
        return results

    # ── Étapes internes ───────────────────────────────────────

    def _fetch_pending(self, limit: int) -> list:
        logger.info(f"  Recherche des fichiers pending dans ingestion_log (limit={limit})...")
        rows = self.trino.execute(
            f"""
            SELECT ingestion_id, filename, file_path, file_type
            FROM iceberg.raw.ingestion_log
            WHERE status ='{self.PENDING_STATUS}' OR status ='{self.FAILED_STATUS}'
            ORDER BY created_at ASC
            LIMIT {limit}
            """,
            fetch=True,
        )
        count = len(rows) if rows else 0
        logger.info(f"  {count} fichier(s) trouvé(s) avec statut '{self.PENDING_STATUS}'")
        return rows or []

    def _process_one(self, ingestion_id: str, filename: str, s3_path: str, file_type: str):

        # ── 1. Téléchargement MinIO → temp ─────────────────────
        parts      = s3_path.replace("s3a://", "").split("/", 1)
        bucket     = parts[0]
        object_key = parts[1]
        logger.info(f"  [MinIO] Téléchargement : s3a://{bucket}/{object_key}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / filename
            self.storage.download_file(bucket, object_key, local_path)
            file_size_kb = local_path.stat().st_size // 1024
            logger.info(f"  [MinIO] Téléchargement OK → {local_path.name} ({file_size_kb} KB)")

            # ── 2. OCR ─────────────────────────────────────────
            raw_markdown, page_count = self.ocr.extract(local_path)

        # ── 3. Cleaning ────────────────────────────────────────
        clean_text = self.cleaner.clean(raw_markdown)

        # ── 4. Enrichissement ──────────────────────────────────
        ai_meta = self.enricher.enrich(clean_text, filename)

        # ── 5. INSERT dans processed.documents ─────────────────
        logger.info(f"  [Trino] INSERT dans iceberg.process.documents...")
        self._insert_processed(
            ingestion_id=ingestion_id,
            filename=filename,
            file_type=file_type,
            clean_text=clean_text,
            page_count=page_count,
            char_count=len(clean_text),
            metadata=ai_meta,
        )

        # ── 6. UPDATE statut ───────────────────────────────────
        logger.info(f"  [Trino] Mise à jour statut : pending → {self.PROCESSED_STATUS}")
        self._update_status(ingestion_id, self.PROCESSED_STATUS)
        logger.info(f"  ✓ {filename} traité avec succès")

    def _insert_processed(
        self,
        ingestion_id: str,
        filename:     str,
        file_type:    str,
        clean_text:   str,
        page_count:   int,
        char_count:   int,
        metadata:     dict,
    ):
        now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000000")
        tags_json = json.dumps(metadata.get("tags", []),         ensure_ascii=False)
        ents_json = json.dumps(metadata.get("key_entities", []), ensure_ascii=False)

        def esc(s: str) -> str:
            return str(s).replace("'", "''")

        self.trino.execute(f"""
            INSERT INTO iceberg.process.documents (ingestion_id, filename, file_type, clean_text, page_count, 
                           char_count, title, summary, domain, doc_type, language, tags, key_entities, processed_at, indexing_status)
                VALUES (
                CAST('{esc(ingestion_id)}'                     AS VARCHAR),
                CAST('{esc(filename)}'                         AS VARCHAR),
                CAST('{esc(file_type)}'                        AS VARCHAR),
                CAST('{esc(clean_text[:65000])}'               AS VARCHAR),
                CAST({page_count}                              AS INTEGER),
                CAST({char_count}                              AS INTEGER),
                CAST('{esc(metadata.get("title", ""))}'        AS VARCHAR),
                CAST('{esc(metadata.get("summary", ""))}'      AS VARCHAR),
                CAST('{esc(metadata.get("domain", "other"))}'  AS VARCHAR),
                CAST('{esc(metadata.get("doc_type", "other"))}' AS VARCHAR),
                CAST('{esc(metadata.get("language", "fr"))}'   AS VARCHAR),
                CAST('{esc(tags_json)}'                        AS VARCHAR),
                CAST('{esc(ents_json)}'                        AS VARCHAR),
                CAST('{now}'                                   AS TIMESTAMP),
                CAST('pending'                                 AS VARCHAR)
            )
        """)
        logger.info(f"  [Trino] INSERT OK → {page_count} pages | {char_count} chars | title='{metadata.get('title', '')[:50]}'")

    def _update_status(self, ingestion_id: str, status: str):
        self.trino.execute(f"""
            UPDATE iceberg.raw.ingestion_log
            SET status = CAST('{status}' AS VARCHAR)
            WHERE ingestion_id = '{ingestion_id}'
        """)
        logger.info(f"  [Trino] Statut mis à jour → {status} (id={ingestion_id[:8]}...)")