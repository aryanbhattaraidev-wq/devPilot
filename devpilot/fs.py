"""Safe access to the project tree, shared by the filesystem and docs servers.

Two rules protect the user: nothing outside the project root is reachable (including via
``..`` or symlinks), and secrets/build junk are never listed, read, searched or indexed.
"""

from collections.abc import Iterator
from pathlib import Path

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".chroma",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build", ".idea", ".vscode",
}
TEXT_EXTENSIONS = {
    ".md", ".rst", ".txt", ".py", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini",
    ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".sh", ".sql", ".html", ".css",
}
MAX_FILE_BYTES = 200_000


class PathError(ValueError):
    pass


def is_blocked(path: Path) -> bool:
    """Secrets and generated state: refuse even if the model asks for them by name."""
    name = path.name.lower()
    return (
        name.startswith(".env")
        or name.endswith((".db", ".sqlite", ".sqlite3", ".pem", ".key", ".lock"))
        or any(part in SKIP_DIRS for part in path.parts)
    )


def resolve_in_root(root: Path, rel: str = ".") -> Path:
    root = root.resolve()
    target = (root / rel).resolve()  # resolves symlinks and '..'
    if not target.is_relative_to(root):
        raise PathError(f"path escapes the project root: {rel!r}")
    if is_blocked(target.relative_to(root)):
        raise PathError(f"access to {rel!r} is not allowed")
    return target


def iter_project_files(root: Path) -> Iterator[Path]:
    """Every readable text file under root, as absolute paths, in stable order."""
    root = root.resolve()
    stack = [root]
    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir(), reverse=True):
            rel = entry.relative_to(root)
            if is_blocked(rel):
                continue
            if entry.is_symlink() and not entry.resolve().is_relative_to(root):
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.suffix.lower() in TEXT_EXTENSIONS and entry.stat().st_size <= MAX_FILE_BYTES:
                yield entry


def rel_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
