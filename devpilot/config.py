"""Server configuration: which MCP servers DevPilot connects to, and how."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


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
