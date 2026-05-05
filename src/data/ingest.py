"""
src/data/ingest.py
------------------
Fetch raw match data from the Riot Games API and persist it locally.

Collects per match:
  - match_id
  - champion picks (ordered by draft turn)
  - bans (ordered by pick turn)
  - roles
  - match outcome (win/loss)

Results are stored both as individual raw JSON files and as a combined
``matches_structured.json`` in the output directory.  Ingestion progress
is checkpointed to ``_progress.json`` so interrupted runs can resume.

Usage (CLI):
    python -m src.data.ingest --api-key RGAPI-... --region na1 --tier challenger --pages 5
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

# Standard draft pick turns (1-indexed global turn across both teams)
# Blue side picks on turns: 1, 4, 5, 8, 9
# Red  side picks on turns: 2, 3, 6, 7, 10
_BLUE_PICK_TURNS = [1, 4, 5, 8, 9]
_RED_PICK_TURNS = [2, 3, 6, 7, 10]

# Riot API teamPosition values → canonical role names used in this project
_POSITION_MAP: dict[str, str] = {
    "TOP": "TOP",
    "JUNGLE": "JUNGLE",
    "MIDDLE": "MID",
    "BOTTOM": "ADC",
    "UTILITY": "SUPPORT",
}

# Progress / structured-output file names (stems, without .json)
_PROGRESS_STEM = "_progress"
_STRUCTURED_STEM = "matches_structured"

# Sentinel sort key used when a pick's draft_turn is unknown.
# Using a value well beyond the max valid draft turn (10) so missing
# entries always sort last.
_MISSING_DRAFT_TURN = 9999

# Retry configuration
_MAX_RETRIES = 5
_RETRY_BACKOFF_BASE = 1.0  # seconds


class RateLimitError(Exception):
    """Raised when the API returns 429 and all retries have been exhausted."""


class RiotClient:
    """Thin wrapper around the Riot Games REST API with rate-limit handling.

    Args:
        api_key: Riot API key (``RGAPI-...``).
        region:  Platform routing value, e.g. ``"na1"``.
        rate_limit_pause: Seconds to sleep between every request to stay
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
        """Send a GET request with retry/back-off for rate limits and server errors.

        - **429 Too Many Requests**: honours the ``Retry-After`` response header
          and retries up to ``_MAX_RETRIES`` times with exponential back-off.
        - **5xx Server Error**: retries with exponential back-off.
        - Any other HTTP error is raised immediately.

        Raises:
            RateLimitError: When ``_MAX_RETRIES`` retries are exhausted.
            requests.HTTPError: For non-retryable HTTP errors.
        """
        time.sleep(self._pause)
        for attempt in range(_MAX_RETRIES):
            resp = self._session.get(url, params=params, timeout=10)

            if resp.status_code == 429:
                # Riot may send a numeric delay (seconds) or an HTTP-date
                # in the Retry-After header; fall back to exponential back-off
                # when the header is absent or cannot be parsed as a number.
                try:
                    retry_after = float(resp.headers["Retry-After"])
                except (KeyError, ValueError):
                    retry_after = _RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Rate limited (429). Sleeping %.1fs (attempt %d/%d).",
                    retry_after, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(retry_after)
                continue

            if resp.status_code >= 500:
                wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Server error %d. Retrying in %.1fs (attempt %d/%d).",
                    resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        raise RateLimitError(f"Exceeded {_MAX_RETRIES} retries for {url}")

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

    def get_grandmaster_summoners(self, queue: str = "RANKED_SOLO_5x5") -> list[dict]:
        """Return summoner entries for the Grandmaster ladder."""
        url = self._platform_url(f"/lol/league/v4/grandmasterleagues/by-queue/{queue}")
        data = self._get(url)
        return data.get("entries", [])

    def get_master_summoners(self, queue: str = "RANKED_SOLO_5x5") -> list[dict]:
        """Return summoner entries for the Master ladder."""
        url = self._platform_url(f"/lol/league/v4/masterleagues/by-queue/{queue}")
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

    def get_match_timeline(self, match_id: str) -> dict:
        """Fetch match timeline data, used to derive accurate draft-pick order.

        The timeline endpoint exposes ``CHAMPION_DRAFT`` events that record
        the global pick turn (``pickTurn``) for each champion selection.
        """
        url = self._regional_url(f"/lol/match/v5/matches/{match_id}/timeline")
        return self._get(url)


# ---------------------------------------------------------------------------
# Data parsing helpers
# ---------------------------------------------------------------------------


def _extract_draft_order_from_timeline(timeline: dict) -> dict[int, int]:
    """Parse ``CHAMPION_DRAFT`` events from a match timeline.

    Args:
        timeline: Raw match timeline JSON from the Riot v5 timeline endpoint.

    Returns:
        Mapping of ``champion_id`` → ``pick_turn`` (1-indexed global turn).
        Empty dict if no draft events are found.
    """
    draft_order: dict[int, int] = {}
    frames = timeline.get("info", {}).get("frames", [])
    for frame in frames:
        for event in frame.get("events", []):
            if event.get("type") == "CHAMPION_DRAFT":
                champ_id: int | None = event.get("championId")
                pick_turn: int | None = event.get("pickTurn")
                if champ_id is not None and pick_turn is not None:
                    draft_order[champ_id] = pick_turn
    return draft_order


def parse_match(match: dict, timeline: dict | None = None) -> dict:
    """Extract a structured draft record from a raw Riot match v5 JSON.

    Extracts:

    * ``match_id`` — unique match identifier.
    * **Champion picks** — one entry per participant, sorted by ``draft_turn``
      (accurate when *timeline* is provided; falls back to the standard
      interleave order: blue turns 1, 4, 5, 8, 9 / red turns 2, 3, 6, 7, 10).
    * **Bans** — sorted by ``pick_turn`` for each team.
    * **Roles** — ``TOP``, ``JUNGLE``, ``MID``, ``ADC``, ``SUPPORT`` from the
      Riot ``teamPosition`` field.
    * **Match outcome** — ``win`` boolean for each team.

    Args:
        match:    Raw Riot match v5 JSON.
        timeline: Optional timeline JSON for accurate draft-pick ordering.
                  Fetched via ``RiotClient.get_match_timeline()``.

    Returns:
        Dict with keys ``match_id``, ``game_duration``, ``blue``, ``red``.

    Raises:
        ValueError: If *match* does not contain exactly 10 participants.
    """
    metadata = match.get("metadata", {})
    info = match.get("info", {})
    match_id: str = metadata.get("matchId", "unknown")
    participants: list[dict] = info.get("participants", [])
    teams_raw: list[dict] = info.get("teams", [])
    teams: dict[int, dict] = {t["teamId"]: t for t in teams_raw}

    if len(participants) != 10:
        raise ValueError(f"Expected 10 participants, got {len(participants)} in {match_id}")

    blue_win: bool = teams.get(100, {}).get("win", False)
    game_duration: int = info.get("gameDuration", 0)

    # Build champion_id → draft_turn mapping from timeline when available
    draft_order_map: dict[int, int] = {}
    if timeline:
        draft_order_map = _extract_draft_order_from_timeline(timeline)

    # Build per-participant pick records
    blue_picks: list[dict] = []
    red_picks: list[dict] = []
    for p in participants:
        champ_id: int = p.get("championId", -1)
        role = _POSITION_MAP.get(p.get("teamPosition", "FILL"), "FILL")
        pick_record: dict[str, Any] = {
            "champion_id": champ_id,
            "champion_name": p.get("championName", ""),
            "role": role,
            "summoner_name": p.get("summonerName", p.get("riotIdGameName", "")),
            "draft_turn": draft_order_map.get(champ_id),
        }
        if p.get("teamId") == 100:
            blue_picks.append(pick_record)
        else:
            red_picks.append(pick_record)

    # Sort by draft turn.  When the timeline was available we have exact values;
    # otherwise fall back to the known standard interleave order.
    if draft_order_map:
        blue_picks.sort(key=lambda x: x["draft_turn"] or _MISSING_DRAFT_TURN)
        red_picks.sort(key=lambda x: x["draft_turn"] or _MISSING_DRAFT_TURN)
    else:
        for i, pick in enumerate(blue_picks):
            pick["draft_turn"] = _BLUE_PICK_TURNS[i] if i < len(_BLUE_PICK_TURNS) else _MISSING_DRAFT_TURN
        for i, pick in enumerate(red_picks):
            pick["draft_turn"] = _RED_PICK_TURNS[i] if i < len(_RED_PICK_TURNS) else _MISSING_DRAFT_TURN

    def _extract_bans(team_id: int) -> list[dict]:
        bans = teams.get(team_id, {}).get("bans", [])
        sorted_bans = sorted(bans, key=lambda b: b.get("pickTurn", 0))
        return [
            {"champion_id": b.get("championId", -1), "pick_turn": b.get("pickTurn")}
            for b in sorted_bans
        ]

    return {
        "match_id": match_id,
        "game_duration": game_duration,
        "blue": {
            "win": blue_win,
            "picks": blue_picks,
            "bans": _extract_bans(100),
        },
        "red": {
            "win": not blue_win,
            "picks": red_picks,
            "bans": _extract_bans(200),
        },
    }


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def _load_progress(output_dir: pathlib.Path) -> dict:
    """Load ingestion progress from ``_progress.json`` in *output_dir*.

    Returns a dict with ``processed_summoners`` and ``seen_matches`` as sets.
    Returns an empty progress dict when the file does not exist or is corrupt.
    """
    progress_path = output_dir / f"{_PROGRESS_STEM}.json"
    if progress_path.exists():
        try:
            data = json.loads(progress_path.read_text())
            data["processed_summoners"] = set(data.get("processed_summoners", []))
            data["seen_matches"] = set(data.get("seen_matches", []))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load progress file: %s — starting fresh.", exc)
    return {"processed_summoners": set(), "seen_matches": set()}


def _save_progress(output_dir: pathlib.Path, progress: dict) -> None:
    """Persist *progress* to ``_progress.json``, converting sets to sorted lists."""
    progress_path = output_dir / f"{_PROGRESS_STEM}.json"
    serializable = {
        "processed_summoners": sorted(progress["processed_summoners"]),
        "seen_matches": sorted(progress["seen_matches"]),
    }
    progress_path.write_text(json.dumps(serializable, indent=2))


def _load_structured(output_dir: pathlib.Path) -> list[dict]:
    """Load existing structured records from ``matches_structured.json``."""
    structured_path = output_dir / f"{_STRUCTURED_STEM}.json"
    if structured_path.exists():
        try:
            return json.loads(structured_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load structured file: %s — starting fresh.", exc)
    return []


def _save_structured(output_dir: pathlib.Path, records: list[dict]) -> None:
    """Flush *records* to ``matches_structured.json``."""
    structured_path = output_dir / f"{_STRUCTURED_STEM}.json"
    structured_path.write_text(json.dumps(records, indent=2))


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


def ingest_matches(
    api_key: str,
    region: str = "na1",
    pages: int = 1,
    matches_per_player: int = 10,
    output_dir: pathlib.Path = RAW_DIR,
    fetch_timeline: bool = False,
    batch_size: int = 50,
    tier: str = "challenger",
) -> None:
    """End-to-end pipeline: fetch high-elo players → match IDs → match data.

    Each raw match is saved as ``<match_id>.json`` in *output_dir*.  Parsed
    draft records are appended to ``matches_structured.json``.  The pipeline
    is fully resumable: ``_progress.json`` records which summoners and matches
    have already been processed so interrupted runs pick up where they left off.

    Args:
        api_key:             Riot API key (``RGAPI-...``).
        region:              Platform region code (e.g. ``"na1"``).
        pages:               Number of pages of summoners to process;
                             each page contains up to 50 entries.
        matches_per_player:  Ranked matches to fetch per summoner (max 100).
        output_dir:          Directory to write raw JSON files and state.
        fetch_timeline:      When ``True``, also fetch the match timeline to
                             obtain accurate draft-pick ordering.
        batch_size:          Save progress to disk every *batch_size* new matches.
        tier:                Ladder tier to pull summoners from.
                             One of ``"challenger"``, ``"grandmaster"``, ``"master"``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = RiotClient(api_key=api_key, region=region)

    # Load existing progress (resumable runs)
    progress = _load_progress(output_dir)
    seen_matches: set[str] = progress["seen_matches"]
    processed_summoners: set[str] = progress["processed_summoners"]

    # Also pick up any raw JSON files already on disk
    reserved_stems = {_PROGRESS_STEM, _STRUCTURED_STEM}
    for p in output_dir.glob("*.json"):
        if p.stem not in reserved_stems:
            seen_matches.add(p.stem)

    structured_records = _load_structured(output_dir)
    structured_ids = {r["match_id"] for r in structured_records}

    logger.info("Fetching %s ladder for region=%s", tier, region)
    tier_lower = tier.lower()
    if tier_lower == "grandmaster":
        entries = client.get_grandmaster_summoners()
    elif tier_lower == "master":
        entries = client.get_master_summoners()
    else:
        entries = client.get_challenger_summoners()
    logger.info("Found %d %s entries", len(entries), tier)

    entries_to_process = [
        e for e in entries[: pages * 50]
        if e["summonerId"] not in processed_summoners
    ]
    logger.info(
        "%d summoners to process (%d already done).",
        len(entries_to_process), len(processed_summoners),
    )

    new_count = 0
    for entry in entries_to_process:
        summoner_id = entry["summonerId"]
        try:
            puuid = client.get_puuid(summoner_id)
            match_ids = client.get_match_ids(puuid, count=matches_per_player)
        except (requests.HTTPError, RateLimitError) as exc:
            logger.warning("Skipping summoner %s: %s", summoner_id, exc)
            continue

        for mid in match_ids:
            if mid in seen_matches:
                continue
            try:
                match_data = client.get_match(mid)

                # Persist raw JSON
                out_path = output_dir / f"{mid}.json"
                out_path.write_text(json.dumps(match_data, indent=2))
                seen_matches.add(mid)
                logger.debug("Saved raw %s", mid)

                # Parse into structured record
                if mid not in structured_ids:
                    timeline: dict | None = None
                    if fetch_timeline:
                        try:
                            timeline = client.get_match_timeline(mid)
                        except (requests.HTTPError, RateLimitError) as exc:
                            logger.warning("Could not fetch timeline for %s: %s", mid, exc)
                    try:
                        record = parse_match(match_data, timeline=timeline)
                        structured_records.append(record)
                        structured_ids.add(mid)
                    except (ValueError, KeyError) as exc:
                        logger.warning("Could not parse match %s: %s", mid, exc)

                new_count += 1

            except (requests.HTTPError, RateLimitError) as exc:
                logger.warning("Could not fetch match %s: %s", mid, exc)

        processed_summoners.add(summoner_id)
        progress["processed_summoners"] = processed_summoners
        progress["seen_matches"] = seen_matches

        # Checkpoint every batch_size new matches
        if new_count > 0 and new_count % batch_size == 0:
            logger.info(
                "Batch checkpoint: %d new matches. Saving progress…", new_count
            )
            _save_progress(output_dir, progress)
            _save_structured(output_dir, structured_records)

    # Final flush
    _save_progress(output_dir, progress)
    _save_structured(output_dir, structured_records)
    logger.info(
        "Ingestion complete. New: %d | Total seen: %d | Structured records: %d",
        new_count, len(seen_matches), len(structured_records),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest LoL match data from Riot API")
    parser.add_argument("--api-key", required=True, help="Riot API key (RGAPI-...)")
    parser.add_argument("--region", default="na1", help="Platform region (e.g. na1, euw1, kr)")
    parser.add_argument(
        "--tier",
        default="challenger",
        choices=["challenger", "grandmaster", "master"],
        help="Ladder tier to pull summoners from",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Pages of summoners to process (50 entries per page)",
    )
    parser.add_argument("--matches-per-player", type=int, default=10)
    parser.add_argument("--output-dir", default=str(RAW_DIR))
    parser.add_argument(
        "--fetch-timeline",
        action="store_true",
        help="Also fetch match timelines for accurate draft-pick ordering",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Save progress to disk every N new matches",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest_matches(
        api_key=args.api_key,
        region=args.region,
        pages=args.pages,
        matches_per_player=args.matches_per_player,
        output_dir=pathlib.Path(args.output_dir),
        fetch_timeline=args.fetch_timeline,
        batch_size=args.batch_size,
        tier=args.tier,
    )
