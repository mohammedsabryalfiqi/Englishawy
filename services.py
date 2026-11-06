"""Shared application services (v5 patched).

Adds `ingestor` (GithubRepoIngestor) so the Gradio web UI and the Telegram bot
share the SAME background GitHub import jobs and the same owner font catalog.
Drop-in replacement for the original services.py.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from font_catalog import FontCatalog
from font_store import FontStore
from github_ingestor import GithubRepoIngestor
from owner_auth import OwnerAuth


@dataclass
class ImageSession:
    image: bytes
    expires: float
    token: str
    text: str = ""
    owner_only: bool = False
    busy: bool = False


class ImageSessions:
    """Bounded, thread-safe image state; callback tokens reject stale cards."""

    def __init__(self, capacity: int = 32, ttl: int = 1800):
        self.capacity, self.ttl = capacity, ttl
        self._items: OrderedDict[tuple[int, int], ImageSession] = OrderedDict()
        self._lock = threading.RLock()

    def _live(self, key: tuple[int, int]) -> ImageSession | None:
        item = self._items.get(key)
        if item is not None and item.expires <= time.monotonic():
            del self._items[key]
            return None
        return item

    def put(self, key: tuple[int, int], image: bytes) -> str:
        with self._lock:
            token = uuid.uuid4().hex[:12]
            self._items[key] = ImageSession(image, time.monotonic() + self.ttl, token)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return token

    def get(self, key: tuple[int, int]) -> bytes | None:
        with self._lock:
            item = self._live(key)
            return item.image if item else None

    def current(self, key: tuple[int, int]) -> ImageSession | None:
        with self._lock:
            return self._live(key)

    def state(self, key: tuple[int, int], token: str) -> ImageSession | None:
        with self._lock:
            item = self._live(key)
            return item if item is not None and item.token == token else None

    def begin(self, key: tuple[int, int], token: str) -> bool:
        with self._lock:
            item = self.state(key, token)
            if item is None or item.busy:
                return False
            item.busy = True
            return True

    def finish(self, key: tuple[int, int], token: str, text: str = "", owner_only: bool = False) -> None:
        with self._lock:
            item = self.state(key, token)
            if item is not None:
                item.busy = False
                item.text = text or item.text
                item.owner_only = owner_only

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
