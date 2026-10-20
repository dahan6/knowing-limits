"""Adapter registry: manage trained adapters with metadata and compatibility."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import config


class AdapterRegistry:
    """Manage trained LoRA adapters, their metadata, and hot-swap compatibility."""

    def __init__(self, registry_path: str = None):
        self.registry_path = Path(
            registry_path or
            Path(config.QLORA_OUTPUT_DIR) / "registry.json"
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"adapters": {}, "conflict_matrix": {}}

    def _save(self):
        """Save registry to disk."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def register(self, name: str, metadata: Dict):
        """Register a new adapter."""
        self._data["adapters"][name] = {
            **metadata,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        self._save()

    def get_adapter(self, name: str) -> Optional[Dict]:
        """Get adapter metadata by name."""
        return self._data["adapters"].get(name)

    def list_adapters(self, filter_by: Dict = None) -> List[Dict]:
        """List all adapters, optionally filtered."""
        adapters = []
        for name, meta in self._data["adapters"].items():
            if filter_by:
                match = all(meta.get(k) == v for k, v in filter_by.items())
                if not match:
                    continue
            adapters.append({"name": name, **meta})
        return adapters

    def unregister(self, name: str):
        """Remove an adapter from registry."""
        if name in self._data["adapters"]:
            del self._data["adapters"][name]
            self._save()

    def set_status(self, name: str, status: str):
        """Set adapter status (active/inactive/archived)."""
        if name in self._data["adapters"]:
            self._data["adapters"][name]["status"] = status
            self._save()

    def add_conflict(self, name_a: str, name_b: str, reason: str = ""):
        """Mark two adapters as conflicting."""
        if name_a not in self._data["conflict_matrix"]:
            self._data["conflict_matrix"][name_a] = {}
        if name_b not in self._data["conflict_matrix"]:
            self._data["conflict_matrix"][name_b] = {}
        self._data["conflict_matrix"][name_a][name_b] = reason
        self._data["conflict_matrix"][name_b][name_a] = reason
        self._save()

    def validate_compatibility(self, adapter_names: List[str]) -> bool:
        """Check if a set of adapters can be mounted together."""
        for i, name_a in enumerate(adapter_names):
            for name_b in adapter_names[i + 1:]:
                conflicts = self._data["conflict_matrix"].get(name_a, {})
                if name_b in conflicts:
                    return False
        return True

    def get_active_adapters(self) -> List[str]:
        """Get list of active adapter names."""
        return [
            name for name, meta in self._data["adapters"].items()
            if meta.get("status") == "active"
        ]


# Global registry instance
_registry = None


def get_registry() -> AdapterRegistry:
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry
