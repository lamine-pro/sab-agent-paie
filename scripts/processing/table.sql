-- Création de la table iceberg.processed.documents
-- À exécuter une seule fois via Trino

CREATE SCHEMA IF NOT EXISTS iceberg.processed
WITH (location = 's3a://p/');

CREATE TABLE IF NOT EXISTS iceberg.processed.documents (
    ingestion_id  VARCHAR,
    filename      VARCHAR,
    file_type     VARCHAR,
    clean_text    VARCHAR,
    page_count    INTEGER,
    char_count    INTEGER,
    -- Métadonnées enrichies
    title         VARCHAR,
    summary       VARCHAR,
    domain        VARCHAR,
    doc_type      VARCHAR,
    language      VARCHAR,
    tags          VARCHAR,   -- JSON array sérialisé
    key_entities  VARCHAR,   -- JSON array sérialisé
    processed_at  TIMESTAMP
)
WITH (
    format         = 'PARQUET',
    location       = 's3a://processed-data/documents/',
    partitioning   = ARRAY['domain', 'file_type']
);

-- Migration : ajout des colonnes d'indexation dans processed.documents
-- À exécuter une seule fois via Trino

ALTER TABLE iceberg.process.documents
    ADD COLUMN indexing_status      VARCHAR;

ALTER TABLE iceberg.process.documents
    ADD COLUMN indexed_chunks_count INTEGER;

ALTER TABLE iceberg.process.documents
    ADD COLUMN indexed_at           TIMESTAMP;

-- Initialiser les documents existants à 'pending'
UPDATE iceberg.process.documents
SET indexing_status = 'pending'
WHERE indexing_status IS NULL;