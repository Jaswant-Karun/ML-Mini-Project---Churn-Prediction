from __future__ import annotations

from importlib import import_module
import os
from typing import Any, Dict, List, Optional


class SupabasePredictionStore:
    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL", "https://vfcpqrtpcdhqqnlyypom.supabase.co").strip()
        self.key = os.getenv("SUPABASE_KEY", "sb_publishable_BNs-j-Uf1fIWIHquiNPHvg_CA9x5Jwf").strip()
        self.table = os.getenv("SUPABASE_TABLE", "prediction_logs").strip()
        self.client = None

        if self.url and self.key:
            try:
                supabase_module = import_module("supabase")
                self.client = supabase_module.create_client(self.url, self.key)
            except Exception:
                self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def log_prediction(self, payload: Dict[str, Any]) -> None:
        if not self.client:
            return
        try:
            self.client.table(self.table).insert(payload).execute()
        except Exception:
            # Keep predictions available even if logging fails.
            pass

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        try:
            res = (
                self.client.table(self.table)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception:
            return []
