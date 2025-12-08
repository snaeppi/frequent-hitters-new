"""Shared Rich console for interactive commands."""

from rich.console import Console

# Single shared console to keep progress bars and interactive output aligned.
console: Console = Console()
