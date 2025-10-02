from __future__ import annotations

import http.client
import json
import logging
from typing import Any, Dict, List, Optional


class WebSearchService:
    """Thin wrapper around the Serper search API."""

    def __init__(
        self,
        api_key: str,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = bool(self.api_key)

        if not self.enabled:
            self.logger.warning("SERPER_API_KEY is not configured; web search is disabled")

    def search_and_summarize(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> Optional[str]:
        """Return a plain-text summary of the top search results for a prompt."""

        if not self.enabled:
            return None

        if not query:
            raise ValueError("Search query must not be empty")

        try:
            raw_results = self._search(query, max_results=max_results)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Serper search failed: %s", exc)
            return None

        if not raw_results:
            return None

        lines: List[str] = [
            "Web search summary (top results):",
        ]

        for index, result in enumerate(raw_results, start=1):
            title = result.get("title") or result.get("name") or "Untitled result"
            snippet = result.get("snippet") or result.get("description") or ""
            url = result.get("link") or result.get("url") or ""
            entry = f"{index}. {title}"
            if snippet:
                entry += f" — {snippet.strip()}"
            if url:
                entry += f" ({url})"
            lines.append(entry)

        return "\n".join(lines)

    def _search(
        self,
        query: str,
        *,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": query, "num": max_results})
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            conn.request("POST", "/search", payload, headers)
            response = conn.getresponse()

            if response.status != 200:
                raise RuntimeError(f"Serper API returned status {response.status}")

            data = response.read().decode("utf-8")
        finally:
            conn.close()

        payload_data = json.loads(data)
        organic_results = payload_data.get("organic")
        if not organic_results:
            return []

        return organic_results[:max_results]
