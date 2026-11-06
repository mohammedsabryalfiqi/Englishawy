"""One-use Telegram-issued codes exchange for opaque, expiring browser sessions.

The browser never submits a Telegram user ID. Only a webhook-authenticated Telegram
update from the configured owner in a private chat can issue a login code.
Codes/tokens are kept in memory as hashes and are intentionally invalidated on restart.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict

from config import is_owner


class OwnerAuth:
    def __init__(self, code_ttl: int = 300, session_ttl: int = 8 * 3600, clock=time.monotonic):
        self.code_ttl = code_ttl
        self.session_ttl = session_ttl
        self._clock = clock
        self._codes: OrderedDict[str, float] = OrderedDict()
        self._sessions: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.RLock()
        self._failures: OrderedDict[str, list[float]] = OrderedDict()

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _prune(self) -> None:
        now = self._clock()
        for mapping in (self._codes, self._sessions):
            for key, expiry in list(mapping.items()):
                if expiry <= now:
                    del mapping[key]
        for key, attempts in list(self._failures.items()):
            attempts[:] = [t for t in attempts if now - t < 60]
            if not attempts:
                del self._failures[key]

    def issue_code(self, user_id: int, chat_id: int, chat_type: str = "private") -> str:
        if not is_owner(user_id, chat_id, chat_type):
            raise PermissionError("إدارة الخطوط متاحة للمالك فقط في محادثة خاصة.")
        with self._lock:
            self._prune()
            # A fresh code invalidates previous unused codes for this single-owner app.
            self._codes.clear()
            code = secrets.token_urlsafe(12)
            self._codes[self._digest(code)] = self._clock() + self.code_ttl
            return code

    def exchange(self, code: str, client_key: str = "default") -> str:
        """Returns an opaque session token; raises PermissionError without leaking validity."""
        denied = PermissionError("الكود غير صحيح أو منتهي. اطلب /login من البوت مرة أخرى.")
        if not isinstance(code, str) or len(code) > 128:
            raise denied
        with self._lock:
            self._prune()
            key = hashlib.sha256(client_key.encode()).hexdigest()[:32]
            attempts = self._failures.setdefault(key, [])
            if len(attempts) >= 8:
                raise PermissionError("محاولات كثيرة. انتظر دقيقة ثم اطلب كوداً جديداً.")
            digest = self._digest(code.strip())
            matched = next((k for k in self._codes if hmac.compare_digest(k, digest)), None)
            if matched is None:
                attempts.append(self._clock())
                if len(self._failures) > 1024:
                    self._failures.popitem(last=False)
                raise denied
            del self._codes[matched]
            self._failures.pop(key, None)
            token = secrets.token_urlsafe(32)
            self._sessions[self._digest(token)] = self._clock() + self.session_ttl
            # Bound memory even if an attacker creates many sessions with valid codes.
            while len(self._sessions) > 128:
                self._sessions.popitem(last=False)
            return token

    def verify(self, token: str | None) -> bool:
        if not isinstance(token, str) or not 20 <= len(token) <= 128:
            return False
        with self._lock:
            self._prune()
            digest = self._digest(token)
            return any(hmac.compare_digest(k, digest) for k in self._sessions)

    def require(self, token: str | None) -> None:
        if not self.verify(token):
            raise PermissionError("سجل دخول المالك أولاً بكود /login من تليجرام.")

    def logout(self, token: str | None) -> None:
        if isinstance(token, str):
            with self._lock:
                self._sessions.pop(self._digest(token), None)
