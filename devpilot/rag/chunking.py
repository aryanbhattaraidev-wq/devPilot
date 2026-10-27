"""Split files into chunks that embed well and can be cited precisely.

Why chunk at all? An embedding is one vector per input, so a whole file would blur many
topics into one point, and retrieval would return far more text than the model needs.
Why not fixed-size slices? They cut sentences and functions in half. So:

* Markdown is split on headings first, so each chunk is about one topic and carries its
  heading path ("Architecture > Transports") as context.
* Anything bigger than ``MAX_CHARS`` is packed line by line, with a small overlap so a
  sentence on a boundary appears whole in at least one chunk.
* Every chunk records its 1-based line range, which is what makes a precise citation
  (``docs/architecture.md:12-30``) possible later.
"""

from dataclasses import dataclass

MAX_CHARS = 1200
OVERLAP_CHARS = 150


@dataclass(frozen=True)
class Chunk:
    text: str
    start_line: int  # 1-based, inclusive
    end_line: int
    heading: str = ""

    def embed_text(self, source: str) -> str:
        """What we actually embed: the text plus where it came from (helps short chunks)."""
        context = f"{source} > {self.heading}" if self.heading else source
        return f"{context}\n{self.text}"


def _pack(lines: list[tuple[int, str]], heading: str) -> list[Chunk]:
    """Greedily pack (line_no, text) pairs into chunks of <= MAX_CHARS with tail overlap."""
    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    size = 0

    def flush() -> None:
        # Trim blank edges so the recorded line range is exactly the text (accurate citations).
        kept = [(n, t) for n, t in current if t.strip()]
        if kept:
            first, last = kept[0][0], kept[-1][0]
            body = "\n".join(t for n, t in current if first <= n <= last)
            chunks.append(Chunk(body, first, last, heading))

    for line_no, text in lines:
        # A single enormous line (minified code, long paragraph): split it by characters.
        pieces = [text[i : i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)] or [""]
        for piece in pieces:
            if current and size + len(piece) + 1 > MAX_CHARS:
                flush()
                tail: list[tuple[int, str]] = []
                kept = 0
                for item in reversed(current):  # carry the last ~OVERLAP_CHARS into the next chunk
                    if kept + len(item[1]) > OVERLAP_CHARS:
                        break
                    tail.insert(0, item)
                    kept += len(item[1]) + 1
                current, size = tail, kept
            current.append((line_no, piece))
            size += len(piece) + 1
    # Skip a trailing chunk that is nothing but carried-over overlap.
    if current and (not chunks or current[-1][0] > chunks[-1].end_line):
        flush()
    return chunks


def _markdown_sections(lines: list[str]) -> list[tuple[str, list[tuple[int, str]]]]:
    sections: list[tuple[str, list[tuple[int, str]]]] = []
    stack: list[tuple[int, str]] = []
    body: list[tuple[int, str]] = []
    heading = ""
    in_fence = False

    def close() -> None:
        if any(t.strip() for _, t in body):
            sections.append((heading, list(body)))

    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        level = len(line) - len(line.lstrip("#"))
        if not in_fence and 1 <= level <= 6 and line[level : level + 1] == " ":
            close()
            body = []
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, line[level:].strip()))
            heading = " > ".join(h for _, h in stack)
        body.append((n, line))
    close()
    return sections


def chunk_text(text: str, filename: str) -> list[Chunk]:
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return []
    if filename.lower().endswith((".md", ".rst", ".txt")):
        return [c for heading, body in _markdown_sections(lines) for c in _pack(body, heading)]
    return _pack(list(enumerate(lines, 1)), "")
