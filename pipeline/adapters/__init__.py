"""Dataset adapters — one per survey programme, the only place programme-specific logic lives."""

from pipeline.adapters.base import DatasetAdapter
from pipeline.adapters.arkansas import ArkansasAdapter
from pipeline.adapters.mdc import MdcAdapter

# Adding a programme means adding a module here and one line below; nothing above the
# common dataset interface changes.
ADAPTERS = {"mdc": MdcAdapter, "arkansas": ArkansasAdapter}


def get(source: str, **kwargs) -> DatasetAdapter:
    """Adapter for a source name, e.g. get("mdc") or get("arkansas", root=...)."""
    try:
        return ADAPTERS[source](**kwargs)
    except KeyError:
        raise KeyError(f"unknown source {source!r} (known: {', '.join(sorted(ADAPTERS))})")


__all__ = ["DatasetAdapter", "MdcAdapter", "ArkansasAdapter", "ADAPTERS", "get"]
