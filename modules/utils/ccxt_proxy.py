"""
CCXT proxy helpers for exchange API access (downloads, live feed).

Host, port, username, and password are read from environment variables only.
Loads project-root ``.env`` on import if present (does not override existing env).
PROXY_ENABLED and PROXY_TYPE default from config.py but can be overridden via env.
"""

import os
from pathlib import Path
from urllib.parse import quote

try:
    import config
except ImportError:
    config = None


def _project_root():
    return Path(__file__).resolve().parents[2]


def _load_dotenv():
    """Load KEY=VALUE lines from project-root .env into os.environ."""
    env_path = _project_root() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def get_proxy_endpoint():
    """Return (host, port) from environment variables only."""
    host = os.environ.get("PROXY_HOST", "")
    port = os.environ.get("PROXY_PORT", "")
    return host, port


def get_proxy_credentials():
    """Return (username, password) from environment variables only."""
    return os.environ.get("PROXY_USERNAME", ""), os.environ.get("PROXY_PASSWORD", "")


def get_proxy_urls():
    """
    Return (http_proxy, https_proxy) URLs or (None, None) if proxy disabled.
    """
    enabled = _env_bool(
        "PROXY_ENABLED",
        getattr(config, "PROXY_ENABLED", False) if config else False,
    )
    if not enabled:
        return None, None

    host, port = get_proxy_endpoint()
    username, password = get_proxy_credentials()
    proxy_type = (
        os.environ.get("PROXY_TYPE")
        or getattr(config, "PROXY_TYPE", "http")
        or "http"
    ).lower()

    if not host or not port:
        env_file = _project_root() / ".env"
        raise ValueError(
            "PROXY_ENABLED but PROXY_HOST and PROXY_PORT are not set. "
            f"Export them or add them to {env_file} (see .env.example)"
        )

    if username and password:
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    elif username or password:
        raise ValueError(
            "Both PROXY_USERNAME and PROXY_PASSWORD must be set in the environment"
        )
    else:
        auth = ""

    scheme = "socks5" if proxy_type in ("socks5", "socks5h") else "http"
    url = f"{scheme}://{auth}{host}:{port}"
    return url, url


def merge_proxy_config(exchange_config):
    """Add ccxt ``proxies`` to an exchange constructor config dict."""
    http, https = get_proxy_urls()
    if http:
        exchange_config = dict(exchange_config)
        exchange_config["proxies"] = {"http": http, "https": https or http}
    return exchange_config


def apply_proxy_to_exchange(exchange):
    """Apply proxy settings to an existing ccxt exchange instance."""
    http, _ = get_proxy_urls()
    if http:
        exchange.http_proxy = http
    return exchange
