"""Shared exception type.

Lives outside `client.py` so that ingestion can raise `AssistantError` without
importing the Anthropic SDK.
"""


class AssistantError(Exception):
    """Unrecoverable failure. Raw SDK and parser exceptions are wrapped in this
    so they don't leak to the CLI."""
