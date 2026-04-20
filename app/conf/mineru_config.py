from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


# MinerU config
@dataclass
class MineruConfig:
    local_base_url: str   # Local self-hosted mineru-api base URL (e.g. http://localhost:8000)
    cloud_base_url: str   # MinerU cloud API base URL (e.g. https://mineru.net/api/v4)
    api_key: str          # API token — only required for cloud API
    use_local: bool       # True = local self-hosted API, False = MinerU cloud API

    @property
    def base_url(self) -> str:
        """Return the active base URL based on the current mode."""
        return self.local_base_url if self.use_local else self.cloud_base_url


mineru_config = MineruConfig(
    local_base_url=os.getenv("MINERU_LOCAL_BASE_URL", "http://localhost:8000"),
    cloud_base_url=os.getenv("MINERU_CLOUD_BASE_URL", "https://mineru.net/api/v4"),
    api_key=os.getenv("MINERU_API_TOKEN"),
    use_local=os.getenv("MINERU_USE_LOCAL", "false").lower() == "true"
)
