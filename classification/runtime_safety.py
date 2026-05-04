from __future__ import annotations

from pathlib import Path


def env_flag(raw: str | None, default: bool = False) -> bool:
    """Parse a boolean-like environment value."""
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_years_spec(spec: str | None, default_years: list[int]) -> list[int]:
    """Parse a comma-separated year list like '2017,2019-2021'."""
    if spec is None or not spec.strip():
        return list(default_years)

    years: set[int] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid year range: {token}")
            years.update(range(start, end + 1))
        else:
            years.add(int(token))
    return sorted(years)


def year_cache_path(cache_dir: Path, year: int) -> Path:
    return cache_dir / f"{year}.json"


def pending_years(years: list[int], cache_dir: Path, force_rerun: bool = False) -> list[int]:
    """Return years still needing transfer processing."""
    if force_rerun:
        return list(years)
    return [year for year in years if not year_cache_path(cache_dir, year).exists()]
