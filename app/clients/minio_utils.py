# Python standard library imports
import os
import json
# MinIO official Python SDK
from minio import Minio
# Internal project config and logger
from app.conf.minio_config import minio_config
from app.core.logger import logger

# Global MinIO client instance, available project-wide after initialization
_minio_client = None


def _init_minio_client() -> Minio | None:
    """
    Initialize the MinIO client and ensure the configured bucket exists
    with a public read-only policy applied.
    :return: Initialized Minio instance, or None if initialization fails
    """
    try:
        client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=False  # Use HTTP for local/intranet deployments; set True with SSL for public deployments
        )
        bucket_name = minio_config.bucket_name

        # Check if the bucket exists; create it if not
        if not client.bucket_exists(bucket_name):
            logger.info(f"MinIO bucket [{bucket_name}] does not exist, creating now")
            client.make_bucket(bucket_name)
            logger.info(f"MinIO bucket [{bucket_name}] created successfully")
        else:
            logger.info(f"MinIO bucket [{bucket_name}] already exists, skipping creation")

        # Apply a public read-only policy: allows anonymous users to access files via URL
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},  # * means all anonymous users (S3-compatible identifier)
                "Action": ["s3:GetObject"],   # Grant read/access permission only
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
            }]
        }
        client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
        logger.info(f"MinIO bucket [{bucket_name}] public read-only policy applied, anonymous URL access enabled")

        return client

    except Exception as e:
        # Log the error and return None so callers can handle the missing client
        logger.error(f"MinIO client initialization failed: {str(e)}", exc_info=True)
        return None


_minio_client = _init_minio_client()


def get_minio_client() -> Minio | None:
    """
    Return the globally initialized MinIO client instance.
    :return: Initialized Minio instance, or None if initialization failed
    """
    return _minio_client
