"""
Crée les buckets MinIO et vérifie leur existence.
À exécuter UNE FOIS après le premier démarrage de la stack.
"""

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")

BUCKETS = "warehouse"  # Nom du bucket à créer

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",  # Valeur arbitraire, requise par boto3
    )

def create_bucket(client, bucket_name: str) -> bool:
    try:
        client.head_bucket(Bucket=bucket_name)
        print(f"  [SKIP] Bucket '{bucket_name}' existe déjà.")
        return False
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            client.create_bucket(Bucket=bucket_name)
            print(f"  [OK]   Bucket '{bucket_name}' créé.")
            return True
        raise

def set_bucket_policy_public_read(client, bucket_name: str):
    """Optionnel : politique de lecture publique pour le développement."""
    policy = f"""{{
        "Version": "2012-10-17",
        "Statement": [{{
            "Effect": "Allow",
            "Principal": {{"AWS": ["*"]}},
            "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::{bucket_name}/*"]
        }}]
    }}"""
    client.put_bucket_policy(Bucket=bucket_name, Policy=policy)

def main():
    print("=== Initialisation des buckets MinIO ===\n")
    client = get_s3_client()

    create_bucket(client, BUCKETS)

    print("\n=== Vérification ===")
    response = client.list_buckets()
    existing = [b["Name"] for b in response["Buckets"]]

    print("\nInitialisation terminée.")

if __name__ == "__main__":
    main()