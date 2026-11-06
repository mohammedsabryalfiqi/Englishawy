"""Shared application services (v5 patched).

Adds `ingestor` (GithubRepoIngestor) so the Gradio web UI and the Telegram bot
share the SAME background GitHub import jobs and the same owner font catalog.
Drop-in replacement for the original services.py.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from font_catalog import FontCatalog
from font_store import FontStore
from github_ingestor import GithubRepoIngestor
from owner_auth import OwnerAuth


class ImageSessions:
    def __init__(self, capacity: int = 32, ttl: int = 1800):
        self.capacity, self.ttl = capacity, ttl
        self._items: OrderedDict[tuple[int, int], tuple[bytes, float, bool]] = OrderedDict()
        self._lock = threading.RLock()

    def put(self, key: tuple[int, int], image: bytes) -> None:
        with self._lock:
            now = time.monotonic()
            self._items[key] = (image, now + self.ttl, False)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def get(self, key: tuple[int, int]) -> bytes | None:
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            if item[1] <= time.monotonic():
                del self._items[key]
                return None
            return item[0]

    def awaiting(self, key: tuple[int, int], value: bool) -> None:
        with self._lock:
            item = self.get(key)
            if item is not None:
                self._items[key] = (item, self._items[key][1], value)

    def is_awaiting(self, key: tuple[int, int]) -> bool:
        with self._lock:
            return self.get(key) is not None and self._items[key][2]


@dataclass
class Services:
    store: FontStore
    catalog: FontCatalog
    auth: OwnerAuth = field(default_factory=OwnerAuth)
    sessions: ImageSessions = field(default_factory=ImageSessions)
    analysis_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))
    ingestor: GithubRepoIngestor | None = None
    notify_callback: Callable[[str], None] | None = None  # e.g. Telegram owner notifier

    @classmethod
    def create(cls, root: str | Path | None = None,
               notify_callback: Callable[[str], None] | None = None) -> "Services":
        store = FontStore(root)
        store.cleanup_temp()
        svc = cls(store=store, catalog=FontCatalog(store), notify_callback=notify_callback)
        svc.ingestor = GithubRepoIngestor(store, notify_callback=notify_callback)
        return svc
