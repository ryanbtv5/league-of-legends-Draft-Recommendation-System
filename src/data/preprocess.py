"""
src/data/preprocess.py
----------------------
Parse raw Riot match JSON files into a clean tabular DataFrame of draft states.

Each row in the output represents ONE pick event during the draft, containing:
  - match_id
  - team (blue / red)
  - role
  - champion_id picked
  - all prior picks and bans (draft state)
  - match outcome (did the picking team win?)

Usage (CLI):
    python -m src.data.preprocess --input data/raw --output data/processed/drafts.parquet
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = pathlib.Path(get("data.raw_dir", "data/raw"))
PROCESSED_DIR = pathlib.Path(get("data.processed_dir", "data/processed"))

ROLES = get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])
ROLE_MAP = {r: i for i, r in enumerate(ROLES)}

# Riot API team IDs
BLUE_TEAM = 100
RED_TEAM = 200


def _extract_draft(match: dict[str, Any]) -> list[dict]:
    """Extract draft rows from a single Riot match v5 JSON.

    Returns a list of dicts, one per pick in the game.
    """
    info = match.get("info", {})
    participants = info.get("participants", [])
    teams = {t["teamId"]: t for t in info.get("teams", [])}

    if len(participants) != 10:
        return []

    blue_win = teams.get(BLUE_TEAM, {}).get("win", False)

    # Build team pick/ban lists
    blue_picks: list[int] = []
    red_picks: list[int] = []
    blue_bans: list[int] = []
    red_bans: list[int] = []

    for p in participants:
        champ_id = p.get("championId", -1)
        team_id = p.get("teamId")
        if team_id == BLUE_TEAM:
            blue_picks.append(champ_id)
        else:
            red_picks.append(champ_id)

    for team in teams.values():
        bans = [b.get("championId", -1) for b in team.get("bans", [])]
        if team["teamId"] == BLUE_TEAM:
            blue_bans = bans
        else:
            red_bans = bans

    rows: list[dict] = []
    match_id = match.get("metadata", {}).get("matchId", "unknown")

    for pick_idx, champ_id in enumerate(blue_picks):
        rows.append(
            {
                "match_id": match_id,
                "team": "blue",
                "pick_order": pick_idx,
                "champion_id": champ_id,
                "role": ROLES[pick_idx] if pick_idx < len(ROLES) else "FILL",
                "blue_picks_so_far": blue_picks[:pick_idx],
                "red_picks_so_far": red_picks[:pick_idx],
                "blue_bans": blue_bans,
                "red_bans": red_bans,
                "blue_win": blue_win,
                "team_win": blue_win,
            }
        )

    for pick_idx, champ_id in enumerate(red_picks):
        rows.append(
            {
                "match_id": match_id,
                "team": "red",
                "pick_order": pick_idx,
                "champion_id": champ_id,
                "role": ROLES[pick_idx] if pick_idx < len(ROLES) else "FILL",
                "blue_picks_so_far": blue_picks[:pick_idx],
                "red_picks_so_far": red_picks[:pick_idx],
                "blue_bans": blue_bans,
                "red_bans": red_bans,
                "blue_win": blue_win,
                "team_win": not blue_win,
            }
        )

    return rows


def preprocess(
    input_dir: pathlib.Path = RAW_DIR,
    output_path: pathlib.Path = PROCESSED_DIR / "drafts.parquet",
) -> pd.DataFrame:
    """Parse all JSON files in *input_dir* and write a clean Parquet file.

    Args:
        input_dir:   Directory of raw Riot match JSON files.
        output_path: Destination Parquet file.

    Returns:
        Processed :class:`pandas.DataFrame`.
    """
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", input_dir)
        return pd.DataFrame()

    all_rows: list[dict] = []
    errors = 0
    for fp in tqdm(json_files, desc="Parsing matches"):
        try:
            match = json.loads(fp.read_text())
            all_rows.extend(_extract_draft(match))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error parsing %s: %s", fp.name, exc)
            errors += 1

    if errors:
        logger.warning("%d files could not be parsed", errors)

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return df


def load_processed(path: pathlib.Path = PROCESSED_DIR / "drafts.parquet") -> pd.DataFrame:
    """Load the processed Parquet file.

    Args:
        path: Path to the Parquet file produced by :func:`preprocess`.

    Returns:
        :class:`pandas.DataFrame` of draft events.
    """
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw LoL match JSON into draft events")
    parser.add_argument("--input", default=str(RAW_DIR), help="Directory of raw JSON files")
    parser.add_argument("--output", default=str(PROCESSED_DIR / "drafts.parquet"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(
        input_dir=pathlib.Path(args.input),
        output_path=pathlib.Path(args.output),
    )
