import hashlib
from contextlib import asynccontextmanager

from aiobotocore.session import get_session
from botocore.config import Config
from botocore.exceptions import ClientError

from app.settings import Settings

MAX_CONTENT = 1_048_576


class Objects:
    def __init__(self, settings: Settings):
        self.settings = settings

    @asynccontextmanager
    async def client(self):
        s = self.settings
        if not s.object_endpoint:
            raise RuntimeError("Object storage is not configured")
        async with get_session().create_client(
            "s3",
            endpoint_url=s.object_endpoint,
            region_name="us-east-1",
            aws_access_key_id=s.object_access_key,
            aws_secret_access_key=s.object_secret_key.get_secret_value(),
            config=Config(
                connect_timeout=3,
                read_timeout=10,
                retries={"max_attempts": 2},
                s3={"addressing_style": "path"},
            ),
        ) as client:
            yield client

    async def ensure_bucket(self):
        async with self.client() as client:
            try:
                await client.head_bucket(Bucket=self.settings.object_bucket)
            except ClientError as exc:
                if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                    raise
                try:
                    await client.create_bucket(Bucket=self.settings.object_bucket)
                except ClientError as create_error:
                    if create_error.response["Error"]["Code"] not in (
                        "BucketAlreadyOwnedByYou",
                        "BucketAlreadyExists",
                    ):
                        raise

    async def put(
        self, key: str, content: str, content_type: str = "text/plain"
    ) -> tuple[int, str]:
        body = content.encode()
        if len(body) > MAX_CONTENT:
            raise ValueError("Content exceeds the 1 MiB limit")
        async with self.client() as client:
            await client.put_object(
                Bucket=self.settings.object_bucket, Key=key, Body=body, ContentType=content_type
            )
        return len(body), hashlib.sha256(body).hexdigest()

    async def get(self, key: str) -> str:
        async with self.client() as client:
            result = await client.get_object(Bucket=self.settings.object_bucket, Key=key)
            async with result["Body"] as stream:
                body = await stream.read(MAX_CONTENT + 1)
            if len(body) > MAX_CONTENT:
                raise ValueError("Stored content exceeds the limit")
            return body.decode()
