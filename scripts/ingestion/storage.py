"""
Client MinIO pour la couche Bronze.
Gère l'upload des fichiers bruts et la vérification d'existence.
Cette classe permet de créer une connexion et de se connecter à MinIO pour charger des fichiers
"""

import os
import hashlib
from pathlib import Path
from loguru import logger

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


class MinIOStorage:
    def __init__(self):
        self.endpoint  = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        self.access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
        self.secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")
        self.client = self._build_client()

    def _build_client(self):
        'cette méthode permet de initier une instance de conexion à MinIO'
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    # ── Upload : chargement des fichiers dans minio ────────────────────────────────────────────────────

    def upload_file(
        self,
        local_path: str | Path,
        bucket: str,
        object_key: str,
    ) -> str:
        """
        Uploader un fichier vers MinIO.
        Retourne le chemin S3 complet : s3a://bucket/object_key
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {local_path}")

        self.client.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=object_key,
        )
        s3_path = f"s3a://{bucket}/{object_key}"
        logger.info(f"Upload OK → {s3_path}")
        return s3_path

    def upload_bytes(
        self,
        data: bytes,
        bucket: str,
        object_key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload depuis des bytes en mémoire (utile pour les tests)."""
        self.client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3a://{bucket}/{object_key}"

    # ── Vérifications ─────────────────────────────────────────────

    def file_exists(self, bucket: str, object_key: str) -> bool:
        """Vérifie si un objet existe dans MinIO."""
        try:
            self.client.head_object(Bucket=bucket, Key=object_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_file_size(self, bucket: str, object_key: str) -> int:
        """Retourne la taille en bytes d'un objet MinIO."""
        response = self.client.head_object(Bucket=bucket, Key=object_key)
        return response["ContentLength"]

    # ── Download ──────────────────────────────────────────────────

    def download_file(
        self,
        bucket: str,
        object_key: str,
        local_path: str | Path,
    ) -> Path:
        """Télécharge un fichier depuis MinIO vers le disque local."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(
            Bucket=bucket,
            Key=object_key,
            Filename=str(local_path),
        )
        return local_path

    def download_bytes(self, bucket: str, object_key: str) -> bytes:
        """Télécharge un fichier depuis MinIO en mémoire."""
        response = self.client.get_object(Bucket=bucket, Key=object_key)
        return response["Body"].read()

    # ── Utilitaires ───────────────────────────────────────────────

    @staticmethod
    def compute_hash(file_path: str | Path) -> str:
        """Calcule le SHA-256 d'un fichier local (pour déduplication)."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def build_object_key(self, filename: str, file_type: str) -> str:
        """
        Construit le chemin de stockage dans Bronze.
        Convention : {file_type}/{filename}
        Ex : pdf/rapport_annuel_2023.pdf
        """
        return f"{file_type}/{filename}"

    def list_files(self, bucket: str, prefix: str = "") -> list[dict]:
        """Liste les fichiers d'un bucket avec leurs métadonnées."""
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [
            {
                "key":           obj["Key"],
                "size":          obj["Size"],
                "last_modified": obj["LastModified"],
            }
            for obj in response["Contents"]
        ]