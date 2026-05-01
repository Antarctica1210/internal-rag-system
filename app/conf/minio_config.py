# Import core dependencies: dataclass, environment variable loading, path handling
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


# Define MinIO object storage config (consistent with LLMConfig style, fields map to .env entries)
@dataclass
class MinIOConfig:
    endpoint: str       # MinIO service address (including http/https and port)
    access_key: str     # MinIO access key (maps to MINIO_ACCESS_KEY)
    secret_key: str     # MinIO secret key (maps to MINIO_SECRET_KEY)
    bucket_name: str    # Default MinIO bucket name (dedicated to knowledge base files)
    minio_img_dir: str  # MinIO folder for storing images
    minio_secure: bool  # Whether to use SSL encryption (http vs https)


# Instantiate MinIO config object, automatically reads and binds values from .env
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)
