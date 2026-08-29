"""Small, dependency-free helpers for deployment security boundaries."""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Callable
from urllib.parse import urlsplit


_INSECURE_PLACEHOLDERS = frozenset(
    {
        "change-me",
        "changeme",
        "worker-secret-key-change-in-production",
        "your-secret-key-change-in-production",
        "replace-with-a-long-random-value",
        "replace-with-a-different-long-random-value",
        "replace-with-a-third-long-random-value",
    }
)

INTERNAL_SERVER_ERROR = "Internal server error"
OAUTH_STATE_COOKIE_PREFIX = "worlddistill_oauth_state_"


def require_secret_env(name: str) -> str:
    """Return a configured secret or fail before a service starts insecurely."""

    value = os.environ.get(name)
    normalized = value.strip() if value is not None else ""
    if not normalized or normalized.lower() in _INSECURE_PLACEHOLDERS:
        raise RuntimeError(
            f"{name} must be set to a non-placeholder secret before starting "
            "the deployment service"
        )
    assert value is not None
    return value


def optional_env_group(feature: str, names: Iterable[str]) -> dict[str, str] | None:
    """Load an optional feature's variables, rejecting partial configuration."""

    variables = tuple(names)
    values = {name: os.environ.get(name, "").strip() for name in variables}
    if not any(values.values()):
        return None

    missing = [name for name, value in values.items() if not value]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"{feature} configuration is incomplete; missing: {missing_text}")
    return values


def bearer_token_from_headers(headers: Mapping[str, str]) -> str | None:
    """Extract a bearer token without ever accepting it from a URL."""

    authorization = ""
    for name, value in headers.items():
        if name.lower() == "authorization":
            authorization = value.strip()
            break

    scheme, separator, credentials = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and credentials.strip():
        return credentials.strip()
    return None


def safe_url_host(value: str | None) -> str | None:
    """Return only a URL's host for diagnostics, dropping credentials and paths."""

    if not value:
        return None
    candidate = value if "://" in value else f"//{value}"
    try:
        return urlsplit(candidate).hostname or "configured"
    except ValueError:
        return "configured"


def public_error_detail(detail: object, status_code: int) -> object:
    """Hide exception details from every server-error response."""

    return INTERNAL_SERVER_ERROR if int(status_code) >= 500 else detail


def cors_allowed_origins(value: str | None = None) -> list[str]:
    """Parse an explicit comma-separated CORS origin allowlist.

    An empty allowlist keeps the deployment same-origin. Wildcards and values
    that are not serialized HTTP(S) origins fail closed because the deployment
    serves credentialed requests.
    """

    raw_value = os.environ.get("LIGHTX2V_CORS_ALLOWED_ORIGINS", "") if value is None else value
    origins: list[str] = []
    seen: set[str] = set()
    for candidate in raw_value.split(","):
        origin = candidate.strip()
        if not origin:
            continue
        if origin == "*":
            raise RuntimeError("LIGHTX2V_CORS_ALLOWED_ORIGINS must not contain '*'")

        try:
            parsed = urlsplit(origin)
            # Accessing port also validates malformed values such as ':not-a-port'.
            parsed.port
        except ValueError as exc:
            raise RuntimeError(f"Invalid CORS origin: {origin!r}") from exc

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError(
                "LIGHTX2V_CORS_ALLOWED_ORIGINS entries must be exact HTTP(S) "
                f"origins without credentials, paths, queries, or fragments: {origin!r}"
            )
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return origins


class OAuthState:
    """Server-side record bound to one OAuth authorization transaction."""

    __slots__ = ("provider", "redirect_uri", "expires_at")

    def __init__(self, provider: str, redirect_uri: str, expires_at: float) -> None:
        self.provider = provider
        self.redirect_uri = redirect_uri
        self.expires_at = expires_at


class OAuthStateStore:
    """Bounded, process-local, one-time OAuth state store.

    The opaque state itself is never retained: only its SHA-256 digest is used
    as the lookup key. ``consume`` removes the record atomically before any
    token exchange can occur, so callbacks cannot be replayed.
    """

    def __init__(
        self,
        ttl_seconds: int = 600,
        max_entries: int = 2048,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError("OAuth state TTL must be a positive integer")
        if isinstance(max_entries, bool) or max_entries <= 0:
            raise ValueError("OAuth state capacity must be a positive integer")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._records: OrderedDict[str, OAuthState] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(state: str) -> str:
        return hashlib.sha256(state.encode("utf-8")).hexdigest()

    def _purge_expired_locked(self, now: float) -> None:
        expired = [key for key, record in self._records.items() if record.expires_at <= now]
        for key in expired:
            self._records.pop(key, None)

    def issue(self, provider: str, redirect_uri: str) -> str:
        if not provider or not redirect_uri:
            raise ValueError("OAuth provider and redirect URI are required")

        state = secrets.token_urlsafe(32)
        now = self._clock()
        record = OAuthState(provider=provider, redirect_uri=redirect_uri, expires_at=now + self.ttl_seconds)
        with self._lock:
            self._purge_expired_locked(now)
            while len(self._records) >= self.max_entries:
                self._records.popitem(last=False)
            self._records[self._key(state)] = record
        return state

    def consume(self, state: str | None, expected_provider: str | None = None) -> OAuthState | None:
        if not state:
            return None

        now = self._clock()
        with self._lock:
            record = self._records.pop(self._key(state), None)
        if record is None or record.expires_at <= now:
            return None
        if expected_provider is not None and record.provider != expected_provider:
            return None
        return record


def oauth_state_cookie_name(state: str) -> str:
    """Return a deterministic cookie name so simultaneous tabs do not collide."""

    digest = hashlib.sha256(state.encode("utf-8")).hexdigest()
    return f"{OAUTH_STATE_COOKIE_PREFIX}{digest[:24]}"


def consume_browser_bound_oauth_state(
    store: OAuthStateStore,
    state: str | None,
    cookie_state: str | None,
    expected_provider: str | None = None,
) -> OAuthState | None:
    """Validate browser binding, then atomically consume an OAuth state."""

    if not state or not cookie_state:
        return None
    try:
        cookie_matches = secrets.compare_digest(state, cookie_state)
    except TypeError:
        return None
    if not cookie_matches:
        return None
    return store.consume(state, expected_provider=expected_provider)
