"""Credential resolution: explicit args > environment variables > ~/.o2org/config.ini.

Keeping this separate from client.py means adding more instances later (e.g. a
[staging] profile alongside [default]) is a config-file change, not a code change.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".o2org" / "config.ini"

ENV_BASE_URL = "O2ORG_BASE_URL"
ENV_EMAIL = "O2ORG_EMAIL"
ENV_PASSWORD = "O2ORG_PASSWORD"


def _read_config_file(path: Path) -> configparser.ConfigParser | None:
    if not path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(path)
    return parser


def resolve_credentials(
    base_url: str | None = None,
    email: str | None = None,
    password: str | None = None,
    profile: str = "default",
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[str, str | None, str | None]:
    """Fill in whatever wasn't passed explicitly, in order: args, env vars, config file.

    `password` (and even `email`) may legitimately end up None here — that's fine
    when the caller is going to authenticate with a token instead of login().
    Only `base_url` is required, since every request needs somewhere to go.

    Args:
        base_url: Explicit base URL, if the caller already has one.
        email: Explicit login email, if the caller already has one.
        password: Explicit login password, if the caller already has one.
        profile: Section name to read from `config_path` (`~/.o2org/config.ini`
            by default) when a value wasn't otherwise provided.
        config_path: Path to the config file. Overridable mainly for testing.

    Returns:
        A `(base_url, email, password)` tuple. `email`/`password` may be None.

    Raises:
        ValueError: `base_url` could not be resolved from any source.
    """
    base_url = base_url or os.environ.get(ENV_BASE_URL)
    email = email or os.environ.get(ENV_EMAIL)
    password = password or os.environ.get(ENV_PASSWORD)

    if not (base_url and email and password):
        parser = _read_config_file(config_path)
        if parser and parser.has_section(profile):
            section = parser[profile]
            base_url = base_url or section.get("base_url")
            email = email or section.get("email")
            password = password or section.get("password")

    if not base_url:
        raise ValueError(
            "No base_url found. Pass connect(base_url=...), set "
            f"{ENV_BASE_URL}, or add it to {config_path} under [{profile}]:\n\n"
            f"    [{profile}]\n"
            "    base_url = https://your-o2-org-dma-instance.example\n"
            "    email = you@example.com\n"
            "    password = your-password\n"
        )
    return base_url, email, password
