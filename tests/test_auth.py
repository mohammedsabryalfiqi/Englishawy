import pytest

from config import is_owner, owner_telegram_id
from owner_auth import OwnerAuth


class Clock:
    def __init__(self): self.now = 100.0
    def __call__(self): return self.now


def test_default_owner_and_private_chat_required(monkeypatch):
    monkeypatch.delenv("OWNER_TELEGRAM_ID", raising=False)
    assert owner_telegram_id() == 5231453921
    assert is_owner(5231453921, 5231453921, "private")
    assert not is_owner(5231453921, -100, "supergroup")
    assert not is_owner(123, 123, "private")


def test_invalid_owner_override_fails_closed(monkeypatch):
    monkeypatch.setenv("OWNER_TELEGRAM_ID", "not-an-id")
    with pytest.raises(ValueError):
        owner_telegram_id()


def test_code_is_single_use_and_session_expires():
    clock = Clock()
    auth = OwnerAuth(code_ttl=5, session_ttl=10, clock=clock)
    with pytest.raises(PermissionError):
        auth.issue_code(999, 999)
    code = auth.issue_code(5231453921, 5231453921)
    token = auth.exchange(code)
    assert auth.verify(token)
    with pytest.raises(PermissionError):
        auth.exchange(code)
    clock.now += 11
    assert not auth.verify(token)


def test_code_expiry_and_logout():
    clock = Clock()
    auth = OwnerAuth(code_ttl=5, clock=clock)
    code = auth.issue_code(5231453921, 5231453921)
    clock.now += 6
    with pytest.raises(PermissionError):
        auth.exchange(code)
    token = auth.exchange(auth.issue_code(5231453921, 5231453921))
    auth.logout(token)
    assert not auth.verify(token)
    assert not auth.verify(None)
    assert not auth.verify(str(5231453921))
