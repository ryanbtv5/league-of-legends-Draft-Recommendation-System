"""
src/data/preprocess.py
----------------------
Parse raw Riot match JSON files into an ML-ready DataFrame of draft states.

Each row in the output represents ONE intermediate draft state — the partial
information visible at a specific global pick turn.  A single 10-pick game
therefore contributes **10 rows** (one per global pick turn 1–10), simulating
every decision point in the draft.

Output schema
~~~~~~~~~~~~~
``match_id``        str     — unique match identifier
``draft_step``      int     — global pick turn (1–10)
``picking_team``    int     — 0 = blue side is picking, 1 = red side
``blue_pick_{1-5}`` int     — champion IDs in blue's pick order; 0 = not yet revealed
``red_pick_{1-5}``  int     — champion IDs in red's pick order; 0 = not yet revealed
``blue_ban_{1-5}``  int     — blue side ban champion IDs; 0 = no ban
``red_ban_{1-5}``   int     — red side ban champion IDs; 0 = no ban
``blue_win``        bool    — label: True if blue side won

Standard draft interleave used for pick ordering within each team
(picks sorted by role: TOP, JUNGLE, MID, ADC, SUPPORT):

  Turn  1 → Blue pick 1   Turn  2 → Red pick 1   Turn  3 → Red pick 2
  Turn  4 → Blue pick 2   Turn  5 → Blue pick 3   Turn  6 → Red pick 3
  Turn  7 → Red pick 4    Turn  8 → Blue pick 4   Turn  9 → Blue pick 5
  Turn 10 → Red pick 5

Note: exact draft-turn ordering requires timeline data (``--fetch-timeline``
in ``src/data/ingest.py``).  When only raw match JSON is available, picks
within each team are ordered by role (TOP → JUNGLE → MID → ADC → SUPPORT)
and mapped to the standard interleave above.

Usage (CLI):
    python -m src.data.preprocess --input data/raw --output data/processed/drafts.parquet
    python -m src.data.preprocess --input data/raw/match_data.jsonl
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

# Riot API teamPosition → canonical role
_POSITION_TO_ROLE: dict[str, str] = {
    "TOP": "TOP",
    "JUNGLE": "JUNGLE",
    "MIDDLE": "MID",
    "BOTTOM": "ADC",
    "UTILITY": "SUPPORT",
}

# Global pick turns assigned to each side.
# Blue side picks at turns: 1, 4, 5, 8, 9
# Red  side picks at turns: 2, 3, 6, 7, 10
_BLUE_PICK_TURNS = [1, 4, 5, 8, 9]
_RED_PICK_TURNS = [2, 3, 6, 7, 10]

# Ordered draft sequence: list of (global_turn, team, team_pick_index).
# team_pick_index is the 0-based index into that team's sorted pick list.
_DRAFT_SEQUENCE: list[tuple[int, str, int]] = [
    (1,  "blue", 0),
    (2,  "red",  0),
    (3,  "red",  1),
    (4,  "blue", 1),
    (5,  "blue", 2),
    (6,  "red",  2),
    (7,  "red",  3),
    (8,  "blue", 3),
    (9,  "blue", 4),
    (10, "red",  4),
]

# Number of picks / bans per side (used for column generation)
_PICKS_PER_TEAM = get("data.picks_per_team", 5)
_BANS_PER_TEAM = get("data.bans_per_team", 5)

# Flat column names produced for each row
_PICK_COLS = (
    [f"blue_pick_{i}" for i in range(1, _PICKS_PER_TEAM + 1)]
    + [f"red_pick_{i}"  for i in range(1, _PICKS_PER_TEAM + 1)]
)
_BAN_COLS = (
    [f"blue_ban_{i}" for i in range(1, _BANS_PER_TEAM + 1)]
    + [f"red_ban_{i}"  for i in range(1, _BANS_PER_TEAM + 1)]
)
ALL_FEATURE_COLS = _PICK_COLS + _BAN_COLS


def _sort_picks_by_role(participants: list[dict], team_id: int) -> list[int]:
    """Return champion IDs for *team_id*, sorted by canonical role order.

    The ordering TOP → JUNGLE → MID → ADC → SUPPORT provides a stable,
    reproducible sequence that can be mapped onto the standard draft
    interleave when exact timeline data is unavailable.

    Participants with unknown/missing positions are appended at the end
    in their original list order.

    Args:
        participants: All 10 participant dicts from ``info.participants``.
        team_id:      Riot team ID (100 = blue, 200 = red).

    Returns:
        List of up to 5 champion IDs in role order.
    """
    team_participants = [p for p in participants if p.get("teamId") == team_id]
    known: list[tuple[int, int]] = []  # (role_order, champion_id)
    unknown: list[int] = []
    for p in team_participants:
        raw_pos = p.get("teamPosition", "")
        role = _POSITION_TO_ROLE.get(raw_pos)
        champ_id = p.get("championId", 0)
        if role and role in ROLE_MAP:
            known.append((ROLE_MAP[role], champ_id))
        else:
            unknown.append(champ_id)
    known.sort(key=lambda x: x[0])
    return [c for _, c in known] + unknown


def _extract_draft_states(match: dict[str, Any]) -> list[dict]:
    """Build intermediate draft-state rows from a single Riot match v5 JSON.

    Generates one row per global pick turn (1–10), each representing the
    partial draft visible at that point.  Pick columns contain the champion
    ID if the pick has already been made at the given turn, or 0 otherwise.
    Ban columns are treated as fully known at every state (standard
    simplification — Phase 2 bans are assumed visible throughout).

    Args:
        match: Raw Riot match v5 JSON dict.

    Returns:
        List of 10 dicts (one per draft turn), or empty list if the match
        does not have exactly 10 participants.
    """
    if "info" not in match and isinstance(match.get("root"), dict):
        match = match["root"]

    info = match.get("info", {})
    participants = info.get("participants", [])
    teams = {t["teamId"]: t for t in info.get("teams", [])}

    if len(participants) != 10:
        return []

    match_id: str = match.get("metadata", {}).get("matchId", "unknown")
    blue_win: bool = bool(teams.get(BLUE_TEAM, {}).get("win", False))

    # Ordered pick lists (role-sorted within each team)
    blue_picks = _sort_picks_by_role(participants, BLUE_TEAM)
    red_picks  = _sort_picks_by_role(participants, RED_TEAM)

    # Pad to 5 elements in case of <5 participants per team
    while len(blue_picks) < _PICKS_PER_TEAM:
        blue_picks.append(0)
    while len(red_picks) < _PICKS_PER_TEAM:
        red_picks.append(0)

    # Extract bans, sorted by pickTurn, padded to 5 elements
    def _get_bans(team_id: int) -> list[int]:
        raw_bans = teams.get(team_id, {}).get("bans", [])
        sorted_bans = sorted(raw_bans, key=lambda b: b.get("pickTurn", 0))
        ids = []
        for ban in sorted_bans:
            champ_id = ban.get("championId", 0)
            if champ_id is None or champ_id < 0:
                champ_id = 0
            ids.append(champ_id)
        while len(ids) < _BANS_PER_TEAM:
            ids.append(0)
        return ids[:_BANS_PER_TEAM]

    blue_bans = _get_bans(BLUE_TEAM)
    red_bans  = _get_bans(RED_TEAM)

    rows: list[dict] = []
    for global_turn, picking_team, team_pick_idx in _DRAFT_SEQUENCE:
        # Reveal only picks whose global turn is <= the current step.
        # The bounds check guards against _PICKS_PER_TEAM being configured
        # to a value larger than the length of the turn-order lists.
        revealed_blue = [
            blue_picks[i] if i < len(_BLUE_PICK_TURNS) and _BLUE_PICK_TURNS[i] <= global_turn else 0
            for i in range(_PICKS_PER_TEAM)
        ]
        revealed_red = [
            red_picks[i] if i < len(_RED_PICK_TURNS) and _RED_PICK_TURNS[i] <= global_turn else 0
            for i in range(_PICKS_PER_TEAM)
        ]

        row: dict[str, Any] = {
            "match_id":     match_id,
            "draft_step":   global_turn,
            "picking_team": 0 if picking_team == "blue" else 1,
        }
        for i, col in enumerate([f"blue_pick_{j}" for j in range(1, _PICKS_PER_TEAM + 1)]):
            row[col] = revealed_blue[i]
        for i, col in enumerate([f"red_pick_{j}" for j in range(1, _PICKS_PER_TEAM + 1)]):
            row[col] = revealed_red[i]
        for i, col in enumerate([f"blue_ban_{j}" for j in range(1, _BANS_PER_TEAM + 1)]):
            row[col] = blue_bans[i]
        for i, col in enumerate([f"red_ban_{j}" for j in range(1, _BANS_PER_TEAM + 1)]):
            row[col] = red_bans[i]
        row["blue_win"] = blue_win
        rows.append(row)

    return rows


def preprocess(
    input_dir: pathlib.Path = RAW_DIR,
    output_path: pathlib.Path = PROCESSED_DIR / "drafts.parquet",
) -> pd.DataFrame:
    """Parse raw match JSON or JSONL data into an ML-ready draft-states DataFrame.

    Each match contributes **10 rows** — one per global pick turn — so that
    models can learn from every intermediate draft state, not just the final
    composition.  All pick and ban columns are flat integers (champion IDs),
    with 0 representing "not yet picked/banned".

    Skips files named ``_progress.json`` and ``matches_structured.json``
    (pipeline state files written by ``src/data/ingest``).

    Args:
        input_dir:   Directory of raw Riot match JSON files or a .json/.jsonl file.
        output_path: Destination Parquet file.

    Returns:
        Processed :class:`pandas.DataFrame` with schema described in the
        module docstring.
    """
    all_rows: list[dict] = []
    errors = 0

    if input_dir.is_dir():
        # Exclude pipeline-state files written by ingest.py
        _skip_stems = {"_progress", "matches_structured"}
        json_files = [
            p for p in input_dir.glob("*.json") if p.stem not in _skip_stems
        ]
        if not json_files:
            logger.warning("No JSON files found in %s", input_dir)
            return pd.DataFrame()

        for fp in tqdm(json_files, desc="Parsing matches"):
            try:
                match = json.loads(fp.read_text())
                rows = _extract_draft_states(match)
                if not rows:
                    logger.debug("Skipped %s (unexpected participant count)", fp.name)
                all_rows.extend(rows)
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Error parsing %s: %s", fp.name, exc)
                errors += 1
    elif input_dir.is_file():
        if input_dir.suffix.lower() == ".jsonl":
            with input_dir.open("r", encoding="utf-8") as handle:
                for line_no, line in tqdm(
                    enumerate(handle, start=1),
                    desc=f"Parsing {input_dir.name}",
                ):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        match = json.loads(line)
                        rows = _extract_draft_states(match)
                        if not rows:
                            logger.debug(
                                "Skipped %s line %d (unexpected participant count)",
                                input_dir.name,
                                line_no,
                            )
                        all_rows.extend(rows)
                    except json.JSONDecodeError as exc:
                        logger.debug(
                            "Error parsing %s line %d: %s",
                            input_dir.name,
                            line_no,
                            exc,
                        )
                        errors += 1
        else:
            try:
                match = json.loads(input_dir.read_text())
                rows = _extract_draft_states(match)
                if not rows:
                    logger.debug(
                        "Skipped %s (unexpected participant count)",
                        input_dir.name,
                    )
                all_rows.extend(rows)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Error parsing %s: %s", input_dir, exc)
                return pd.DataFrame()
    else:
        logger.warning("Input path does not exist: %s", input_dir)
        return pd.DataFrame()

    if errors:
        logger.warning("%d files could not be parsed", errors)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # Ensure consistent dtypes for all pick/ban columns
        for col in _PICK_COLS + _BAN_COLS:
            if col in df.columns:
                df[col] = df[col].astype("int32")
        df["draft_step"]   = df["draft_step"].astype("int8")
        df["picking_team"] = df["picking_team"].astype("int8")
        df["blue_win"]     = df["blue_win"].astype(bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Wrote %d rows (%d matches × 10 states) to %s",
        len(df), len(df) // 10 if not df.empty else 0, output_path,
    )
    return df


def load_processed(path: pathlib.Path = PROCESSED_DIR / "drafts.parquet") -> pd.DataFrame:
    """Load the processed Parquet file produced by :func:`preprocess`.

    Args:
        path: Path to the Parquet file.

    Returns:
        :class:`pandas.DataFrame` of draft states with the schema described
        in the module docstring.
    """
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw LoL match JSON into ML-ready draft states. "
            "Produces 10 rows per match (one per global pick turn)."
        )
    )
    parser.add_argument("--input",  default=str(RAW_DIR),  help="Directory of raw JSON files")
    parser.add_argument("--output", default=str(PROCESSED_DIR / "drafts.parquet"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(
        input_dir=pathlib.Path(args.input),
        output_path=pathlib.Path(args.output),
    )
