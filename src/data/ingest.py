"""
src/data/ingest.py
------------------
Fetch raw match data from the Riot Games API and persist it locally.

Usage (CLI):
    python -m src.data.ingest --region na1 --tier DIAMOND --pages 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import requests

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = pathlib.Path(get("data.raw_dir", "data/raw"))

# Riot API base URL (v4 summoner / v5 match endpoints)
_BASE = "https://{region}.api.riotgames.com"
_REGIONAL = "https://{routing}.api.riotgames.com"

# Region → routing cluster mapping
_ROUTING = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
}


class RiotClient:
    """Thin wrapper around the Riot Games REST API.

    Args:
        api_key: Riot API key (``RGAPI-...``).
        region:  Platform routing value, e.g. ``"na1"``.
        rate_limit_pause: Seconds to sleep between requests to stay
                          within the default 20 req/s personal key limit.
    """

    def __init__(self, api_key: str, region: str = "na1", rate_limit_pause: float = 0.05) -> None:
        self.api_key = api_key
        self.region = region.lower()
        self.routing = _ROUTING.get(self.region, "americas")
        self._pause = rate_limit_pause
        self._session = requests.Session()
        self._session.headers.update({"X-Riot-Token": api_key})

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        time.sleep(self._pause)
        resp = self._session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _platform_url(self, path: str) -> str:
        return _BASE.format(region=self.region) + path

    def _regional_url(self, path: str) -> str:
        return _REGIONAL.format(routing=self.routing) + path

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_challenger_summoners(self, queue: str = "RANKED_SOLO_5x5") -> list[dict]:
        """Return summoner entries for the Challenger ladder."""
        url = self._platform_url(f"/lol/league/v4/challengerleagues/by-queue/{queue}")
        data = self._get(url)
        return data.get("entries", [])

    def get_puuid(self, summoner_id: str) -> str:
        """Resolve a summonerId to its PUUID."""
        url = self._platform_url(f"/lol/summoner/v4/summoners/{summoner_id}")
        data = self._get(url)
        return data["puuid"]

    def get_match_ids(self, puuid: str, count: int = 20, queue: int = 420) -> list[str]:
        """Return recent ranked match IDs for a player.

        Args:
            puuid: Player's PUUID.
            count: Number of matches to fetch (max 100).
            queue: Queue ID — 420 = Ranked Solo/Duo.
        """
        url = self._regional_url(f"/lol/match/v5/matches/by-puuid/{puuid}/ids")
        return self._get(url, params={"queue": queue, "count": count})

    def get_match(self, match_id: str) -> dict:
        """Fetch full match data by match ID."""
        url = self._regional_url(f"/lol/match/v5/matches/{match_id}")
        return self._get(url)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


def ingest_matches(
    api_key: str,
    region: str = "na1",
    pages: int = 1,
    matches_per_player: int = 10,
    output_dir: pathlib.Path = RAW_DIR,
) -> None:
    """End-to-end pipeline: fetch Challenger players → match IDs → match data.

    Args:
        api_key:             Riot API key.
        region:              Platform region code.
        pages:               Not used directly (kept for CLI parity); controls
                             how many challenger entries to process.
        matches_per_player:  Ranked matches to fetch per summoner.
        output_dir:          Directory to write raw JSON files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = RiotClient(api_key=api_key, region=region)

    logger.info("Fetching Challenger ladder for region=%s", region)
    entries = client.get_challenger_summoners()
    logger.info("Found %d Challenger entries", len(entries))

    seen_matches: set[str] = set()
    # Load already-downloaded match IDs to avoid duplicates
    for p in output_dir.glob("*.json"):
        seen_matches.add(p.stem)

    for entry in entries[: pages * 50]:
        summoner_id = entry["summonerId"]
        try:
            puuid = client.get_puuid(summoner_id)
            match_ids = client.get_match_ids(puuid, count=matches_per_player)
        except requests.HTTPError as exc:
            logger.warning("Skipping summoner %s: %s", summoner_id, exc)
            continue

        for mid in match_ids:
            if mid in seen_matches:
                continue
            try:
                match_data = client.get_match(mid)
                out_path = output_dir / f"{mid}.json"
                out_path.write_text(json.dumps(match_data, indent=2))
                seen_matches.add(mid)
                logger.debug("Saved %s", mid)
            except requests.HTTPError as exc:
                logger.warning("Could not fetch match %s: %s", mid, exc)

    logger.info("Ingestion complete. Total matches: %d", len(seen_matches))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest LoL match data from Riot API")
    parser.add_argument("--api-key", required=True, help="Riot API key (RGAPI-...)")
    parser.add_argument("--region", default="na1")
    parser.add_argument("--pages", type=int, default=1, help="Challenger pages to process (50 summoners each)")
    parser.add_argument("--matches-per-player", type=int, default=10)
    parser.add_argument("--output-dir", default=str(RAW_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest_matches(
        api_key=args.api_key,
        region=args.region,
        pages=args.pages,
        matches_per_player=args.matches_per_player,
        output_dir=pathlib.Path(args.output_dir),
    )
