"""Server configuration: which MCP servers DevPilot connects to, and how."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Secrets and defaults, from the environment or the repo-root .env (never servers.json)."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    github_token: str = ""
    github_repo: str = ""  # default "owner/repo" when a tool call omits one
    # Any OpenAI-compatible endpoint: OpenRouter (default), OpenAI, a local Ollama, ...
    llm_api_key: str = Field("", validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY"))
    llm_base_url: str = "https://openrouter.ai/api/v1"
    devpilot_model: str = "openai/gpt-4o-mini"  # must support tool calling
    devpilot_db: str = str(ROOT / "devpilot.db")  # SQLite file for the project server
    # Project knowledge: what the filesystem and docs servers can see, and the vector index.
    project_root: str = str(ROOT)
    chroma_dir: str = str(ROOT / ".chroma")
    embedding_model: str = "openai/text-embedding-3-small"
    embedder: str = "api"  # "api" (real semantic embeddings) or "hash" (offline, for tests)


class StdioServerConfig(BaseModel):
    """A local server: the client spawns it as a subprocess and talks over stdin/stdout."""

    transport: Literal["stdio"]
    command: str
    args: list[str] = []
    env: dict[str, str] = {}
    cwd: str | None = None
    connect_timeout: float = 15.0


class HttpServerConfig(BaseModel):
    """A remote server: already running somewhere, reached over streamable HTTP."""

    transport: Literal["streamable-http"]
    url: str
    headers: dict[str, str] = {}
    connect_timeout: float = 15.0


ServerConfig = Annotated[
    StdioServerConfig | HttpServerConfig, Field(discriminator="transport")
]


class AppConfig(BaseModel):
    servers: dict[str, ServerConfig]


def load_config(path: str | Path = "servers.json") -> AppConfig:
    return AppConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
