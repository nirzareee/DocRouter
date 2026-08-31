"""Backend registry.

Backends declare themselves here. `available_backends()` filters to the ones
whose dependencies are actually installed, so the same eval command works on a
laptop with nothing installed and on a GPU box with everything.
"""

from __future__ import annotations

from .base import Backend, ParseResult, doc_id_for
from .local import NaiveBackend, PyLibBackend
from .docling_backend import DoclingBackend

__all__ = [
    "Backend",
    "ParseResult",
    "doc_id_for",
    "REGISTRY",
    "get_backend",
    "available_backends",
]

REGISTRY: dict[str, type[Backend]] = {
    NaiveBackend.name: NaiveBackend,
    PyLibBackend.name: PyLibBackend,
    DoclingBackend.name: DoclingBackend,
}


def get_backend(name: str) -> Backend:
    if name not in REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def available_backends() -> list[Backend]:
    """Instantiate every backend whose dependencies are importable."""
    out = []
    for cls in REGISTRY.values():
        backend = cls()
        if backend.is_available():
            out.append(backend)
    return out
