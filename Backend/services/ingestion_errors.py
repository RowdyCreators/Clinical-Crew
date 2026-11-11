"""Errors and input validation shared by ingestion API and pipeline."""
import re


class IngestionError(ValueError):
    """An input or library state that prevents safe ingestion."""


def disease_name(value):
    value = value.strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise IngestionError('Use a category with letters, numbers, underscores, or hyphens.')
    return value
