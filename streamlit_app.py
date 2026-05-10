"""
streamlit_app.py
----------------
Interactive Streamlit UI for draft recommendations and win-probability signals.

**Optimization Strategy:**
- Lazy model loading: Models are loaded only when "Recommend" is clicked
- Session state caching: Loaded models/encoders persist across reruns
- Champion names cached: First load scans JSONL (cached), subsequent runs instant
- ThreadPool limits: PyTorch threads reduced to minimize OpenMP overhead
- TorchScript: Models compiled to TorchScript for faster reload when available
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Iterable
import math

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Reduce PyTorch thread overhead on machines with many OpenMP threads; helps
# Streamlit startup latency on macOS / low-core environments.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from src.features.champion_encoder import ChampionEncoder, DraftStateEncoder
from src.models import baseline as bm
from src.models import neural as nm
from src.models import transformer as tm
from src.models.win_prediction import DraftWinPredictor
from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = pathlib.Path(get("training.model_save_dir", "models"))
DEFAULT_DATA_PATH = pathlib.Path(get("data.processed_dir", "data/processed")) / "drafts.parquet"
DEFAULT_RAW_JSONL = pathlib.Path(get("data.raw_dir", "data/raw")) / "match_data.jsonl"
PROCESSED_DIR = pathlib.Path(get("data.processed_dir", "data/processed"))
NUM_CHAMPIONS: int = get("data.num_champions", 165)
ROLES: list[str] = get("data.roles", ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])

BLUE_PICK_COLS = [f"blue_pick_{i}" for i in range(1, 6)]
RED_PICK_COLS = [f"red_pick_{i}" for i in range(1, 6)]
BLUE_BAN_COLS = [f"blue_ban_{i}" for i in range(1, 6)]
RED_BAN_COLS = [f"red_ban_{i}" for i in range(1, 6)]
ALL_DRAFT_COLS = BLUE_PICK_COLS + RED_PICK_COLS + BLUE_BAN_COLS + RED_BAN_COLS

TRANSFORMER_DRAFT_ORDER: list[tuple[str, str, int]] = [
    ("ban", "blue", 0),
    ("ban", "red", 0),
    ("ban", "blue", 1),
    ("ban", "red", 1),
    ("ban", "blue", 2),
    ("ban", "red", 2),
    ("pick", "blue", 0),
    ("pick", "red", 0),
    ("pick", "red", 1),
    ("pick", "blue", 1),
    ("pick", "blue", 2),
    ("pick", "red", 2),
    ("ban", "red", 3),
    ("ban", "blue", 3),
    ("ban", "red", 4),
    ("ban", "blue", 4),
    ("pick", "red", 3),
    ("pick", "blue", 3),
    ("pick", "blue", 4),
    ("pick", "red", 4),
]

CHAMPION_CACHE_DIR = PROCESSED_DIR / "streamlit_cache"


def _cache_key(path: pathlib.Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _champion_ids_cache_path(data_path: pathlib.Path) -> pathlib.Path:
    return CHAMPION_CACHE_DIR / f"champion_ids_{_cache_key(data_path)}.json"


def _champion_names_cache_path(raw_jsonl_path: pathlib.Path) -> pathlib.Path:
    return CHAMPION_CACHE_DIR / f"champion_names_{_cache_key(raw_jsonl_path)}.json"


@st.cache_data(show_spinner=False)
def load_champion_ids(data_path: pathlib.Path) -> tuple[list[int], bool]:
    cache_path = _champion_ids_cache_path(data_path)
    if cache_path.exists():
        try:
            cached_ids = json.loads(cache_path.read_text(encoding="utf-8"))
            champion_ids = sorted({int(cid) for cid in cached_ids if int(cid) > 0})
            if champion_ids:
                return champion_ids, False
        except Exception:
            pass

    if data_path.exists():
        try:
            df = pd.read_parquet(data_path, columns=ALL_DRAFT_COLS)
            values = df.to_numpy(dtype=np.int64, copy=False).ravel()
            champion_ids = sorted({int(cid) for cid in values if int(cid) > 0})
            if champion_ids:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(champion_ids), encoding="utf-8")
                return champion_ids, False
        except Exception as exc:
            logger.warning("Could not read champion IDs from %s: %s", data_path, exc)
    return list(range(NUM_CHAMPIONS)), True


@st.cache_data(show_spinner=False)
def load_champion_name_map(
    raw_jsonl_path: pathlib.Path,
    required_ids: tuple[int, ...],
    max_lines: int = 20000,
) -> dict[int, str]:
    cache_path = _champion_names_cache_path(raw_jsonl_path)
    name_map: dict[int, str] = {}
    if cache_path.exists():
        try:
            cached_map = json.loads(cache_path.read_text(encoding="utf-8"))
            name_map = {
                int(champ_id): str(champ_name)
                for champ_id, champ_name in cached_map.items()
                if int(champ_id) > 0 and isinstance(champ_name, str) and champ_name
            }
        except Exception:
            name_map = {}

    if not raw_jsonl_path.exists():
        return name_map

    remaining = set(required_ids)
    if remaining and name_map:
        remaining.difference_update(name_map.keys())
        if not remaining:
            return name_map

    if not remaining:
        return name_map

    try:
        with raw_jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line_no > max_lines or not remaining:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "info" not in record and isinstance(record.get("root"), dict):
                    record = record["root"]
                participants = record.get("info", {}).get("participants", [])
                for participant in participants:
                    champ_id = participant.get("championId")
                    champ_name = participant.get("championName")
                    if isinstance(champ_id, int) and champ_id > 0 and isinstance(champ_name, str) and champ_name:
                        name_map.setdefault(champ_id, champ_name)
                        if champ_id in remaining:
                            remaining.remove(champ_id)
    except OSError as exc:
        logger.warning("Could not read champion names from %s: %s", raw_jsonl_path, exc)

    if name_map:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(name_map), encoding="utf-8")
        except Exception:
            pass

    return name_map


@st.cache_data(show_spinner=False)
def load_role_priors(data_path: pathlib.Path, _champ_enc: ChampionEncoder) -> np.ndarray:
    """Compute per-champion prior for each role from processed drafts.parquet.

    Returns a matrix (num_champions, num_roles) of frequencies normalized per-role.
    """
    champ_enc = _champ_enc
    path = pathlib.Path(data_path)
    if not path.exists():
        return np.ones((champ_enc.num_champions, len(ROLES)), dtype=np.float32) / len(ROLES)

    try:
        df = pd.read_parquet(path)
    except Exception:
        return np.ones((champ_enc.num_champions, len(ROLES)), dtype=np.float32) / len(ROLES)

    # Mapping of pick position -> role index (assumes picks are ordered TOP,JUNGLE,MID,ADC,SUPPORT)
    pick_cols = [f"blue_pick_{i}" for i in range(1, 6)]
    role_counts = np.zeros((champ_enc.num_champions, len(ROLES)), dtype=np.float32)
    for i, col in enumerate(pick_cols):
        role_idx = min(i, len(ROLES) - 1)
        vals = df[col].to_numpy(dtype=np.int64, copy=False)
        for cid in vals:
            if int(cid) == 0:
                continue
            idx = champ_enc.encode(int(cid))
            role_counts[idx, role_idx] += 1.0

    # Smooth and normalize per-role
    role_counts += 1e-3
    role_priors = role_counts / role_counts.sum(axis=0, keepdims=True)
    return role_priors


@st.cache_data(show_spinner=False)
def available_models() -> list[str]:
    """Return list of available trained models. Skips any with access issues."""
    names: list[str] = []
    
    if (MODEL_DIR / "rf_recommender.pkl").exists():
        names.append("Random Forest")
    if (MODEL_DIR / "mlp_recommender_best.pt").exists():
        names.append("MLP")
    if (MODEL_DIR / "transformer_recommender_best.pt").exists():
        names.append("Transformer")
    
    # Prefer faster models first (MLP, RF, then Transformer as fallback)
    preferred_order = ["MLP", "Random Forest", "Transformer"]
    names = sorted(names, key=lambda n: preferred_order.index(n) if n in preferred_order else 999)
    return names


@st.cache_resource(show_spinner=False)
def load_model(
    model_name: str,
    champion_ids: tuple[int, ...],
) -> tuple[ChampionEncoder, DraftStateEncoder, object]:
    """Load a model and its encoders. Raises an exception if model cannot be loaded."""
    champ_enc = ChampionEncoder(champion_ids)
    state_enc = DraftStateEncoder(champ_enc)

    if model_name == "Random Forest":
        model = bm.RandomForestRecommender.load(MODEL_DIR / "rf_recommender.pkl")
    elif model_name == "MLP":
        model = nm.load_model(MODEL_DIR / "mlp_recommender_best.pt", torch.device("cpu"))
    elif model_name == "Transformer":
        model = tm.load_model(MODEL_DIR / "transformer_recommender_best.pt", torch.device("cpu"))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return champ_enc, state_enc, model


@st.cache_resource(show_spinner=False)
def load_win_predictor(_champ_enc: ChampionEncoder) -> DraftWinPredictor | None:
    """Load win probability predictor if available and compatible with `_champ_enc`, else return None.

    The leading underscore avoids Streamlit cache hashing issues for unhashable objects.
    """
    champ_enc = _champ_enc
    predictor_path = MODEL_DIR / "win_predictor_best.pt"
    if not predictor_path.exists():
        return None
    # Prefer a scripted (TorchScript) artifact if present for faster loads
    script_path = MODEL_DIR / "win_predictor_script.pt"
    if script_path.exists():
        try:
            scripted = torch.jit.load(script_path, map_location="cpu")
            logger.info("Loaded scripted win predictor from %s", script_path)
            return scripted
        except Exception:
            # Fall back to state_dict loading below
            pass

    try:
        predictor = DraftWinPredictor(num_champions=champ_enc.num_champions)
        predictor.load_state_dict(torch.load(predictor_path, map_location="cpu"))
        predictor.eval()

        # Try to create a scripted copy for faster subsequent loads (best-effort)
        try:
            scripted = torch.jit.script(predictor)
            torch.jit.save(scripted, script_path)
            logger.info("Saved scripted win predictor to %s", script_path)
        except Exception:
            pass

        return predictor
    except Exception as exc:
        logger.warning("Could not load win predictor: %s", exc)
        return None


def _compute_win_probability(
    predictor: DraftWinPredictor,
    champ_enc: ChampionEncoder,
    blue_picks: list[int],
    red_picks: list[int],
    blue_bans: list[int],
    red_bans: list[int],
    debug: bool = False,
) -> tuple[float | None, dict] | float | None:
    """Compute blue team win probability given draft state. 
    
    Matches training format exactly: iterate through draft_order_cols and collect
    champions at each position (blue_ban_1, red_ban_1, ..., red_pick_5).
    
    If debug=True, returns (win_prob, debug_info) dict.
    """
    debug_info = {}
    try:
        with torch.inference_mode():
            # Map from draft position to champion. Training iterates through these columns in order.
            draft_order_positions = [
                ("blue_ban", 0),   # blue_ban_1
                ("red_ban", 0),    # red_ban_1
                ("blue_ban", 1),   # blue_ban_2
                ("red_ban", 1),    # red_ban_2
                ("blue_ban", 2),   # blue_ban_3
                ("red_ban", 2),    # red_ban_3
                ("blue_pick", 0),  # blue_pick_1
                ("red_pick", 0),   # red_pick_1
                ("red_pick", 1),   # red_pick_2
                ("blue_pick", 1),  # blue_pick_2
                ("blue_pick", 2),  # blue_pick_3
                ("red_pick", 2),   # red_pick_3
                ("red_ban", 3),    # red_ban_4
                ("blue_ban", 3),   # blue_ban_4
                ("red_ban", 4),    # red_ban_5
                ("blue_ban", 4),   # blue_ban_5
                ("red_pick", 3),   # red_pick_4
                ("blue_pick", 3),  # blue_pick_4
                ("blue_pick", 4),  # blue_pick_5
                ("red_pick", 4),   # red_pick_5
            ]
            
            # Collect champions in draft order, matching training format
            seq = []
            collected_champs = []
            
            for pos_type, idx in draft_order_positions:
                champ_id = None
                if pos_type == "blue_pick" and idx < len(blue_picks):
                    champ_id = blue_picks[idx]
                elif pos_type == "red_pick" and idx < len(red_picks):
                    champ_id = red_picks[idx]
                elif pos_type == "blue_ban" and idx < len(blue_bans):
                    champ_id = blue_bans[idx]
                elif pos_type == "red_ban" and idx < len(red_bans):
                    champ_id = red_bans[idx]
                
                if champ_id is not None and champ_id != 0:
                    # Encode the champion and add +1 offset (matching training)
                    encoded_idx = champ_enc.encode(int(champ_id)) + 1
                    seq.append(encoded_idx)
                    collected_champs.append((pos_type, idx, champ_id, encoded_idx))
            
            # Truncate to 20 and pad (matching training)
            seq = seq[:20]
            seq += [0] * (20 - len(seq))
            
            debug_info["draft_order_sequence"] = seq
            debug_info["collected"] = collected_champs
            logger.info("Draft sequence (training format): %s", seq)

            # Support both nn.Module with `predict_proba` and TorchScript modules
            draft_seq = torch.tensor([seq], dtype=torch.long)
            if hasattr(predictor, "predict_proba"):
                win_prob = predictor.predict_proba(draft_seq)
                debug_info["model_type"] = "nn.Module"
            else:
                # Scripted modules expose `forward`; call and sigmoid manually
                logits = predictor(draft_seq)
                debug_info["raw_logits"] = float(logits.view(-1)[0].item()) if logits.numel() > 0 else None
                debug_info["model_type"] = "TorchScript"
                logger.info("Raw logits: %s", debug_info.get("raw_logits"))
                win_prob = torch.sigmoid(logits)

            # Normalise the returned tensor/array to a scalar float robustly.
            if isinstance(win_prob, torch.Tensor):
                if win_prob.numel() == 0:
                    return (None, debug_info) if debug else None
                # take the first element (batch dim)
                val = float(win_prob.view(-1)[0].item())
            else:
                try:
                    val = float(win_prob)
                except Exception:
                    return (None, debug_info) if debug else None

            if math.isnan(val):
                return (None, debug_info) if debug else None
            
            debug_info["win_prob"] = val
            return (val, debug_info) if debug else val
    except Exception as exc:
        logger.warning("Error computing win probability: %s", exc)
        return (None, debug_info) if debug else None



def _predict_probabilities(model_name: str, model: object, features: np.ndarray) -> np.ndarray:
    if model_name == "MLP":
        with torch.inference_mode():
            model.eval()
            device = next(model.parameters()).device
            inputs = torch.from_numpy(features).float().to(device)
            logits = model.net(inputs)
            return torch.softmax(logits, dim=-1).cpu().numpy()
    return model.predict_proba(features)


def _predict_mlp_embedding(model: nm.DraftMLP, champ_enc: ChampionEncoder, blue_picks, red_picks, blue_bans, red_bans, team) -> np.ndarray:
    """Use embedding-forward path of DraftMLP with team-aware probabilities."""
    with torch.inference_mode():
        model.eval()
        device = next(model.parameters()).device

        def _pad(ids: list[int], pad_to: int = 5):
            ids_enc = [champ_enc.encode(int(cid)) for cid in ids]
            if len(ids_enc) >= pad_to:
                ids_enc = ids_enc[:pad_to]
            else:
                ids_enc = ids_enc + [0] * (pad_to - len(ids_enc))
            return torch.tensor([ids_enc], dtype=torch.long).to(device)

        bp = _pad(blue_picks)
        rp = _pad(red_picks)
        bb = _pad(blue_bans)
        rb = _pad(red_bans)

        role_idx = min(len(blue_picks) if team == "blue" else len(red_picks), len(ROLES) - 1)
        role_vec = torch.zeros((1, len(ROLES)), dtype=torch.float32, device=device)
        role_vec[0, role_idx] = 1.0
        team_vec = torch.tensor([[1.0, 0.0]] if team == "blue" else [[0.0, 1.0]], dtype=torch.float32, device=device)

        probs = model.predict_proba(bp, rp, bb, rb, role_vec, team_vec)
        return probs[0]


def _build_transformer_tokens(
    blue_picks: list[int],
    red_picks: list[int],
    blue_bans: list[int],
    red_bans: list[int],
    champ_enc: ChampionEncoder,
) -> torch.Tensor:
    sources = {
        ("pick", "blue"): blue_picks,
        ("pick", "red"): red_picks,
        ("ban", "blue"): blue_bans,
        ("ban", "red"): red_bans,
    }

    observed_tokens: list[int] = []
    for kind, team, idx in TRANSFORMER_DRAFT_ORDER:
        values = sources[(kind, team)]
        if idx < len(values):
            observed_tokens.append(champ_enc.encode(int(values[idx])) + 1)

    observed_tokens = observed_tokens[:19]
    observed_tokens.append(0)
    return torch.tensor(observed_tokens, dtype=torch.long)


def _predict_transformer_probabilities(
    model: object,
    tokens: torch.Tensor,
) -> np.ndarray:
    with torch.inference_mode():
        model.eval()
        device = next(model.parameters()).device
        tokens = tokens.unsqueeze(0).to(device)
        logits = model(tokens)[:, -1, :]
        logits_adj = logits[:, 1:]
        return torch.softmax(logits_adj, dim=-1).cpu().numpy()


def _find_duplicates(groups: Iterable[Iterable[int]]) -> list[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for group in groups:
        for champ in group:
            if champ in seen:
                duplicates.add(champ)
            else:
                seen.add(champ)
    return sorted(duplicates)


def _mask_scores(scores: np.ndarray, unavailable: list[int]) -> np.ndarray:
    masked = scores.copy()
    if unavailable:
        masked[unavailable] = 0.0
    return masked


def main() -> None:
    st.set_page_config(page_title="LoL Draft Assistant", layout="wide")
    st.title("LoL Draft Recommendation Assistant")
    st.caption("Select the current draft state to get top-5 picks and evaluation metrics.")

    # Initialize session state for lazy loading
    if "champ_enc" not in st.session_state:
        st.session_state.champ_enc = None
    if "state_enc" not in st.session_state:
        st.session_state.state_enc = None
    if "loaded_models" not in st.session_state:
        st.session_state.loaded_models = {}
    if "win_predictor" not in st.session_state:
        st.session_state.win_predictor = None
    if "role_priors" not in st.session_state:
        pass

    st.sidebar.header("Draft Settings")
    data_path = pathlib.Path(
        st.sidebar.text_input("Champion source (drafts.parquet)", str(DEFAULT_DATA_PATH))
    )
    champion_ids, used_fallback_ids = load_champion_ids(data_path)
    # Load champion names with timeout fallback (lazy-load on-demand)
    champion_name_map = load_champion_name_map(DEFAULT_RAW_JSONL, tuple(champion_ids))

    def _champion_label(champ_id: int) -> str:
        name = champion_name_map.get(champ_id, f"Champion {champ_id}")
        return f"{name} ({champ_id})"

    if used_fallback_ids:
        st.sidebar.info("Using default champion IDs (0..N-1). Provide drafts.parquet for dataset IDs.")
    model_order = available_models()
    if not model_order:
        st.error(f"No trained models found in {MODEL_DIR}. Train a model before running the app.")
        st.stop()

    models_available = model_order
    model_name = st.sidebar.selectbox("Recommendation model (primary)", models_available)
    compare_all_models = st.sidebar.checkbox("Compare all models", value=False)
    team = st.sidebar.radio("Picking team", ["blue", "red"], horizontal=True)
    top_k = st.sidebar.slider("Top-K recommendations", min_value=1, max_value=10, value=5, step=1)

    col_blue, col_red = st.columns(2)
    with col_blue:
        st.subheader("Blue team")
        blue_picks = st.multiselect(
            "Blue picks",
            options=champion_ids,
            format_func=_champion_label,
            default=[],
            max_selections=5,
        )
        blue_bans = st.multiselect(
            "Blue bans",
            options=champion_ids,
            format_func=_champion_label,
            default=[],
            max_selections=5,
        )
    with col_red:
        st.subheader("Red team")
        red_picks = st.multiselect(
            "Red picks",
            options=champion_ids,
            format_func=_champion_label,
            default=[],
            max_selections=5,
        )
        red_bans = st.multiselect(
            "Red bans",
            options=champion_ids,
            format_func=_champion_label,
            default=[],
            max_selections=5,
        )

    duplicates = _find_duplicates([blue_picks, red_picks, blue_bans, red_bans])
    if duplicates:
        duplicate_labels = [_champion_label(champ_id) for champ_id in duplicates]
        st.warning(f"Duplicate champions detected across picks/bans: {duplicate_labels}")

    if st.button("Recommend"):
        # Lazy-load models and encoders on first recommendation request
        if st.session_state.champ_enc is None:
            with st.spinner("Loading models and encoders..."):
                try:
                    champ_enc, state_enc, selected_model = load_model(model_name, tuple(champion_ids))
                    st.session_state.champ_enc = champ_enc
                    st.session_state.state_enc = state_enc
                    st.session_state.loaded_models[model_name] = selected_model
                    if compare_all_models:
                        for name in models_available:
                            if name == model_name:
                                continue
                            try:
                                _, _, m = load_model(name, tuple(champion_ids))
                                st.session_state.loaded_models[name] = m
                            except Exception as e:
                                logger.warning("Could not load %s: %s", name, e)
                    # Load win predictor
                    st.session_state.win_predictor = load_win_predictor(champ_enc)
                except Exception as e:
                    st.error(f"Failed to load {model_name}. Try selecting a different model or check the models/ directory.")
                    logger.error("Model load error: %s", e, exc_info=True)
                    return
        elif model_name not in st.session_state.loaded_models:
            with st.spinner(f"Loading {model_name}..."):
                try:
                    _, _, selected_model = load_model(model_name, tuple(champion_ids))
                    st.session_state.loaded_models[model_name] = selected_model
                except Exception as e:
                    st.error(f"Failed to load {model_name}. Try selecting a different model.")
                    logger.error("Model load error: %s", e, exc_info=True)
                    return

        champ_enc = st.session_state.champ_enc
        state_enc = st.session_state.state_enc
        loaded_models = st.session_state.loaded_models
        win_predictor = st.session_state.win_predictor

        # Build recommendations for the requested model(s) and display side-by-side
        model_results: dict[str, list[dict]] = {}
        if compare_all_models:
            names_to_run = models_available
        else:
            names_to_run = [model_name] if model_name in loaded_models else []
        if not names_to_run:
            st.error("No models loaded. Please try again.")
            return
        
        for name in names_to_run:
            try:
                # Load model if not already loaded
                if name not in loaded_models:
                    _, _, m = load_model(name, tuple(champion_ids))
                    loaded_models[name] = m
                else:
                    m = loaded_models[name]
                
                if name == "Transformer":
                    tokens = _build_transformer_tokens(blue_picks, red_picks, blue_bans, red_bans, champ_enc)
                    scores = _predict_transformer_probabilities(m, tokens)[0]
                elif name == "MLP":
                    # use embedding-forward to respect role
                    try:
                        probs = _predict_mlp_embedding(m, champ_enc, blue_picks, red_picks, blue_bans, red_bans, team)
                        scores = probs
                    except Exception:
                        # Fall back to flat features path
                        row = {
                            "blue_picks_so_far": blue_picks,
                            "red_picks_so_far": red_picks,
                            "blue_bans": blue_bans,
                            "red_bans": red_bans,
                            "team": team,
                        }
                        features = state_enc.encode_batch([row])
                        scores = _predict_probabilities(name, m, features)[0]
                else:  # Random Forest
                    row = {
                        "blue_picks_so_far": blue_picks,
                        "red_picks_so_far": red_picks,
                        "blue_bans": blue_bans,
                        "red_bans": red_bans,
                        "team": team,
                    }
                    features = state_enc.encode_batch([row])
                    scores = _predict_probabilities(name, m, features)[0]

                # Mask unavailable champions
                unavailable_ids = set(blue_picks + red_picks + blue_bans + red_bans)
                unavailable_idx = [champ_enc.encode(cid) for cid in sorted(unavailable_ids)]
                masked_scores = _mask_scores(scores, unavailable_idx)

                top_idx = np.argsort(masked_scores)[::-1][:top_k]
                recs = [
                    {
                        "Champion": champion_name_map.get(champ_enc.decode(int(idx)), f"Champion {champ_enc.decode(int(idx))}"),
                        "Champion ID": champ_enc.decode(int(idx)),
                        "Recommendation Confidence": float(masked_scores[idx]),
                        "Rank": rank + 1,
                    }
                    for rank, idx in enumerate(top_idx)
                ]
                model_results[name] = recs
            except Exception as e:
                logger.warning("Error predicting with %s: %s", name, e)
                st.warning(f"Could not generate recommendations with {name}")

        # Compute win probability if predictor is available
        win_prob = None
        win_debug_info = {}
        if win_predictor is not None:
            result = _compute_win_probability(
                win_predictor,
                champ_enc,
                blue_picks,
                red_picks,
                blue_bans,
                red_bans,
                debug=True,
            )
            if isinstance(result, tuple):
                win_prob, win_debug_info = result
            else:
                win_prob = result

        if model_results:
            # Display primary model metric + win prob
            primary_recs = model_results.get(model_name, [])
            metric_cols = st.columns(2)
            with metric_cols[0]:
                if primary_recs:
                    st.metric(
                        "Recommendation Confidence (top pick)",
                        f"{primary_recs[0]['Recommendation Confidence']:.1%}",
                    )
            with metric_cols[1]:
                if win_predictor is not None:
                    if win_prob is None:
                        st.error("Could not compute win probability")
                    else:
                        if team == "blue":
                            prob = win_prob
                            label = "Blue Win Probability"
                        else:
                            prob = 1.0 - win_prob
                            label = "Red Win Probability"
                        st.metric(label, f"{prob:.1%}")
                else:
                    st.info("Win predictor not trained. Train a win prediction model for this metric.")



            st.subheader("Top recommendations by model")
            cols = st.columns(len(model_results))
            for (mname, recs), col in zip(model_results.items(), cols):
                with col:
                    st.markdown(f"**{mname}**")
                    if recs:
                        st.dataframe(pd.DataFrame(recs), hide_index=True, width="stretch")
                    else:
                        st.info("No recommendations available.")
        else:
            st.info("No recommendations available. Check your draft inputs.")


if __name__ == "__main__":
    main()
