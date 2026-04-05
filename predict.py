#!/usr/bin/env python3
"""
Tennis match winner predictor using a Random Forest classifier.

Trains on historical ATP match data (1991–2025), engineering rolling
ELO ratings, head-to-head records, and serve statistics as features,
then predicts the probability each player wins a given match-up.

Usage
-----
    python predict.py "Player Name 1" "Player Name 2"
    python predict.py "Player Name 1" "Player Name 2" --surface clay
    python predict.py "Player Name 1" "Player Name 2" --surface grass

Surfaces: hard (default), clay, grass
"""

import argparse
import os
import sys
from collections import defaultdict, deque
from datetime import date
from difflib import get_close_matches

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Match data year range
MATCH_YEAR_START = 1991
MATCH_YEAR_END = 2025

# Required columns — rows missing any of these are dropped
REQUIRED_COLUMNS = [
    "winner_id", "loser_id",
    "winner_ht", "loser_ht",
    "winner_age", "loser_age",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
    "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank_points", "loser_rank_points",
    "winner_rank", "loser_rank",
    "surface",
]

# Rolling-window sizes for win-rate features
WIN_RATE_WINDOWS = [3, 5, 10, 25, 50, 100]

# Rolling-window sizes for serve-stat features
SERVE_STAT_WINDOWS = [3, 5, 10, 20, 50, 100, 200, 300, 2000]

# Rolling-window sizes for ELO-gradient features
ELO_GRAD_WINDOWS = [5, 10, 20, 35, 50, 100, 250]

# ELO starting rating and K-factor
ELO_DEFAULT = 1500.0
ELO_K = 24

# Mapping from internal stat key to feature-name prefix
SERVE_STAT_PREFIX = {
    "p_ace": "P_ACE_LAST",
    "p_df": "P_DF_LAST",
    "p_1stIn": "P_1ST_IN_LAST",
    "p_1stWon": "P_1ST_WON_LAST",
    "p_2ndWon": "P_2ND_WON_LAST",
    "p_bpSaved": "P_BP_SAVED_LAST",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_match_data():
    """Load and concatenate all ATP match CSVs, dropping incomplete rows."""
    print("Loading match data...", end=" ", flush=True)

    frames = []
    for year in range(MATCH_YEAR_START, MATCH_YEAR_END):
        path = os.path.join(DATA_DIR, "all", f"atp_matches_{year}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, low_memory=False))

    aus_path = os.path.join(DATA_DIR, "all", "aus_open_2025.csv")
    if os.path.exists(aus_path):
        frames.append(pd.read_csv(aus_path, low_memory=False))

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    print(f"{len(df):,} matches loaded.")
    return df


def load_players():
    """Return players DataFrame with a combined ``full_name`` column."""
    path = os.path.join(DATA_DIR, "atp_players.csv")
    df = pd.read_csv(path, low_memory=False)
    df["full_name"] = (
        df["name_first"].fillna("") + " " + df["name_last"].fillna("")
    ).str.strip()
    return df


def load_rankings():
    """Return ``{player_id: (rank, points)}`` from the most recent ranking date."""
    path = os.path.join(DATA_DIR, "rankings", "atp_rankings_current.csv")
    df = pd.read_csv(path)
    latest_date = df["ranking_date"].max()
    df = df[df["ranking_date"] == latest_date]
    return {
        int(row["player"]): (int(row["rank"]), int(row["points"]))
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Player name resolution
# ---------------------------------------------------------------------------

def find_player(name, players_df):
    """Return ``(player_id, canonical_name)`` for the closest matching name.

    Tries an exact case-insensitive match first, then falls back to fuzzy
    matching via :func:`difflib.get_close_matches`.

    Raises
    ------
    ValueError
        If no sufficiently close match is found.
    """
    name_lower = name.lower()
    players_df = players_df.copy()
    players_df["full_name_lower"] = players_df["full_name"].str.lower()

    exact = players_df[players_df["full_name_lower"] == name_lower]
    if not exact.empty:
        row = exact.iloc[0]
        return int(row["player_id"]), row["full_name"]

    all_names = players_df["full_name_lower"].tolist()
    close = get_close_matches(name_lower, all_names, n=3, cutoff=0.6)
    if close:
        row = players_df[players_df["full_name_lower"] == close[0]].iloc[0]
        return int(row["player_id"]), row["full_name"]

    raise ValueError(f"Could not find player: '{name}'. Check spelling.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mean(arr):
    """Return the mean of *arr*, or 0.5 if it is empty."""
    if len(arr) == 0:
        return 0.5
    return sum(arr) / len(arr)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_h2h_features(df, n_rows):
    """Compute head-to-head win-count differences (overall and per surface)."""
    h2h = defaultdict(int)
    h2h_surf = defaultdict(lambda: defaultdict(int))
    h2h_diff = np.zeros(n_rows)
    h2h_surf_diff = np.zeros(n_rows)

    for i, (winner, loser, surface) in enumerate(tqdm(
        zip(df["winner_id"], df["loser_id"], df["surface"]),
        total=n_rows,
        desc="H2H",
    )):
        h2h_diff[i] = h2h[(winner, loser)] - h2h[(loser, winner)]
        h2h_surf_diff[i] = (
            h2h_surf[surface][(winner, loser)]
            - h2h_surf[surface][(loser, winner)]
        )
        h2h[(winner, loser)] += 1
        h2h_surf[surface][(winner, loser)] += 1

    return h2h_diff, h2h_surf_diff, h2h, h2h_surf


def _compute_matches_played(df, n_rows):
    """Compute difference in career matches played at the time of each match."""
    matches_played = defaultdict(int)
    diff = np.zeros(n_rows)

    for i, (winner, loser) in enumerate(tqdm(
        zip(df["winner_id"], df["loser_id"]),
        total=n_rows,
        desc="Matches played",
    )):
        diff[i] = matches_played[winner] - matches_played[loser]
        matches_played[winner] += 1
        matches_played[loser] += 1

    return diff, matches_played


def _compute_win_rate_features(df, n_rows):
    """Compute rolling win-rate differences for each window in WIN_RATE_WINDOWS."""
    state = {}
    columns = {}

    for k in WIN_RATE_WINDOWS:
        history = defaultdict(lambda: deque(maxlen=k))
        col = np.zeros(n_rows)

        for i, (winner, loser) in enumerate(tqdm(
            zip(df["winner_id"], df["loser_id"]),
            total=n_rows,
            desc=f"Win rate last {k}",
        )):
            wh = history[winner]
            lh = history[loser]
            col[i] = (_mean(wh) if wh else 0) - (_mean(lh) if lh else 0)
            history[winner].append(1)
            history[loser].append(0)

        columns[f"WIN_LAST_{k}_DIFF"] = col
        state[k] = history

    return columns, state


def _compute_serve_stat_features(df, n_rows):
    """Compute rolling serve-statistic differences for each window in SERVE_STAT_WINDOWS."""
    state = {}
    columns = {}

    for k in SERVE_STAT_WINDOWS:
        history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=k)))
        stat_arrays = {stat: np.zeros(n_rows) for stat in SERVE_STAT_PREFIX}

        for i, row in enumerate(tqdm(
            df.itertuples(index=False),
            total=n_rows,
            desc=f"Serve stats last {k}",
        )):
            winner = row.winner_id
            loser = row.loser_id

            for stat in SERVE_STAT_PREFIX:
                stat_arrays[stat][i] = (
                    _mean(history[winner][stat])
                    - _mean(history[loser][stat])
                )

            w_svpt = row.w_svpt
            l_svpt = row.l_svpt
            w_1st = row.w_1stIn
            l_1st = row.l_1stIn

            # Winner serve stats (requires at least one second-serve attempt)
            if w_svpt != 0 and w_svpt != w_1st:
                history[winner]["p_ace"].append(100 * row.w_ace / w_svpt)
                history[winner]["p_df"].append(100 * row.w_df / w_svpt)
                history[winner]["p_1stIn"].append(100 * w_1st / w_svpt)
                history[winner]["p_2ndWon"].append(
                    100 * row.w_2ndWon / (w_svpt - w_1st)
                )
            if w_1st != 0:
                history[winner]["p_1stWon"].append(
                    100 * row.w_1stWon / w_1st
                )
            if row.w_bpFaced != 0:
                history[winner]["p_bpSaved"].append(
                    100 * row.w_bpSaved / row.w_bpFaced
                )

            # Loser serve stats
            if l_svpt != 0 and l_svpt != l_1st:
                history[loser]["p_ace"].append(100 * row.l_ace / l_svpt)
                history[loser]["p_df"].append(100 * row.l_df / l_svpt)
                history[loser]["p_1stIn"].append(100 * l_1st / l_svpt)
                history[loser]["p_2ndWon"].append(
                    100 * row.l_2ndWon / (l_svpt - l_1st)
                )
            if l_1st != 0:
                history[loser]["p_1stWon"].append(
                    100 * row.l_1stWon / l_1st
                )
            if row.l_bpFaced != 0:
                history[loser]["p_bpSaved"].append(
                    100 * row.l_bpSaved / row.l_bpFaced
                )

        for stat, prefix in SERVE_STAT_PREFIX.items():
            columns[f"{prefix}_{k}_DIFF"] = stat_arrays[stat]
        state[k] = history

    return columns, state


def _compute_elo_features(df, n_rows):
    """Compute overall and per-surface ELO rating differences."""
    elo = defaultdict(lambda: ELO_DEFAULT)
    elo_surf = defaultdict(lambda: defaultdict(lambda: ELO_DEFAULT))
    elo_diff = np.zeros(n_rows)
    elo_surf_diff = np.zeros(n_rows)

    for i, (winner, loser) in enumerate(tqdm(
        zip(df["winner_id"], df["loser_id"]),
        total=n_rows,
        desc="ELO",
    )):
        ew, el = elo[winner], elo[loser]
        expected_winner = 1 / (1 + 10 ** ((el - ew) / 400))
        elo_diff[i] = ew - el
        elo[winner] = ew + ELO_K * (1 - expected_winner)
        elo[loser] = el + ELO_K * (0 - (1 - expected_winner))

    for i, (winner, loser, surface) in enumerate(tqdm(
        zip(df["winner_id"], df["loser_id"], df["surface"]),
        total=n_rows,
        desc="ELO surface",
    )):
        ew, el = elo_surf[surface][winner], elo_surf[surface][loser]
        expected_winner = 1 / (1 + 10 ** ((el - ew) / 400))
        elo_surf_diff[i] = ew - el
        elo_surf[surface][winner] = ew + ELO_K * (1 - expected_winner)
        elo_surf[surface][loser] = el + ELO_K * (0 - (1 - expected_winner))

    return elo_diff, elo_surf_diff, elo, elo_surf


def _compute_elo_gradient_features(df, n_rows):
    """Compute ELO slope differences over rolling windows (ELO_GRAD_WINDOWS).

    A positive value means player 1's ELO has been rising faster than
    player 2's over the last *n* matches.
    """
    state = {}
    columns = {}

    for n in ELO_GRAD_WINDOWS:
        history = defaultdict(lambda: deque(maxlen=n))
        col = np.zeros(n_rows)

        for i, (winner, loser) in enumerate(tqdm(
            zip(df["winner_id"], df["loser_id"]),
            total=n_rows,
            desc=f"ELO grad {n}",
        )):
            wh = history[winner] if winner in history else deque([ELO_DEFAULT], maxlen=n)
            lh = history[loser] if loser in history else deque([ELO_DEFAULT], maxlen=n)
            ew, el = wh[-1], lh[-1]
            expected_winner = 1 / (1 + 10 ** ((el - ew) / 400))
            new_ew = ew + ELO_K * (1 - expected_winner)
            new_el = el + ELO_K * (0 - (1 - expected_winner))

            if len(wh) >= n and len(lh) >= n:
                slope_w = np.polyfit(np.arange(len(wh)), np.array(wh), 1)[0]
                slope_l = np.polyfit(np.arange(len(lh)), np.array(lh), 1)[0]
                col[i] = slope_w - slope_l

            history[winner].append(new_ew)
            history[loser].append(new_el)

        columns[f"ELO_GRAD_{n}_DIFF"] = col
        state[n] = history

    return columns, state


def _build_last_known_rankings(df):
    """Scan match data to record each player's most recently seen rank/points."""
    last_rank = {}
    last_rank_pts = {}
    for _, row in df.iterrows():
        last_rank[int(row["winner_id"])] = int(row["winner_rank"])
        last_rank[int(row["loser_id"])] = int(row["loser_rank"])
        last_rank_pts[int(row["winner_id"])] = int(row["winner_rank_points"])
        last_rank_pts[int(row["loser_id"])] = int(row["loser_rank_points"])
    return last_rank, last_rank_pts


def build_features_and_state(df):
    """Build the full feature matrix and capture per-player state.

    Returns
    -------
    feature_df : pd.DataFrame
        One row per match with all engineered features plus a ``RESULT``
        column (+1 = player-1 wins, -1 = player-2 wins).  Half the rows
        are randomly flipped so the model sees both orderings.
    player_state : dict
        Current state for every player after the full history — used when
        constructing a prediction vector for a new match.
    """
    print("Engineering features (this may take a minute)...")
    n_rows = len(df)

    # ------------------------------------------------------------------
    # Static per-match differences
    # ------------------------------------------------------------------
    fd = {
        "ATP_POINT_DIFF": (df["winner_rank_points"] - df["loser_rank_points"]).values,
        "ATP_RANK_DIFF": (df["winner_rank"] - df["loser_rank"]).values,
        "AGE_DIFF": (df["winner_age"] - df["loser_age"]).values,
        "HEIGHT_DIFF": (df["winner_ht"] - df["loser_ht"]).values,
        "BEST_OF": df["best_of"].values,
        "DRAW_SIZE": df["draw_size"].values,
    }

    # ------------------------------------------------------------------
    # Rolling / state-based features
    # ------------------------------------------------------------------
    h2h_diff, h2h_surf_diff, h2h, h2h_surf = _compute_h2h_features(df, n_rows)
    fd["H2H_DIFF"] = h2h_diff
    fd["H2H_SURFACE_DIFF"] = h2h_surf_diff

    diff_n_games, matches_played = _compute_matches_played(df, n_rows)
    fd["DIFF_N_GAMES"] = diff_n_games

    win_rate_cols, wins_last_k = _compute_win_rate_features(df, n_rows)
    fd.update(win_rate_cols)

    serve_cols, serve_last_k = _compute_serve_stat_features(df, n_rows)
    fd.update(serve_cols)

    elo_diff, elo_surf_diff, elo, elo_surf = _compute_elo_features(df, n_rows)
    fd["ELO_DIFF"] = elo_diff
    fd["ELO_SURFACE_DIFF"] = elo_surf_diff

    elo_grad_cols, elo_grad = _compute_elo_gradient_features(df, n_rows)
    fd.update(elo_grad_cols)

    # ------------------------------------------------------------------
    # Result column and row-flipping
    # ------------------------------------------------------------------
    fd["RESULT"] = np.ones(n_rows)
    feature_df = pd.DataFrame(fd)

    # Randomly flip half the rows so the model learns both player orderings
    flip_mask = np.random.rand(n_rows) < 0.5
    diff_cols = [c for c in feature_df.columns if "DIFF" in c] + ["RESULT"]
    feature_df.loc[flip_mask, diff_cols] *= -1

    # ------------------------------------------------------------------
    # Player state snapshot (used when building prediction vectors)
    # ------------------------------------------------------------------
    last_rank, last_rank_pts = _build_last_known_rankings(df)

    player_state = {
        "elo": elo,
        "elo_surf": elo_surf,
        "elo_grad": elo_grad,
        "h2h": h2h,
        "h2h_surf": h2h_surf,
        "matches_played": matches_played,
        "wins_last_k": wins_last_k,
        "serve_last_k": serve_last_k,
        "last_rank": last_rank,
        "last_rank_pts": last_rank_pts,
    }

    return feature_df, player_state


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(feature_df):
    """Train a Random Forest on the feature matrix and report test accuracy.

    Returns
    -------
    model : RandomForestClassifier
    feature_names : list[str]
        Column names in the order the model expects them.
    """
    print("Training model...", end=" ", flush=True)

    data = feature_df.to_numpy(dtype=float)
    np.random.shuffle(data)

    X = data[:, :-1]
    y = np.where(data[:, -1] > 0, "Player 1 Wins", "Player 2 Wins")

    split = int(0.85 * len(data))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=9,
        max_features="log2",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"done. Test accuracy: {accuracy:.1%}")
    return model, list(feature_df.columns[:-1])


# ---------------------------------------------------------------------------
# Prediction vector construction
# ---------------------------------------------------------------------------

def _player_rank_and_points(pid, rankings, player_state):
    """Return ``(rank, points)`` for *pid*, preferring live rankings."""
    if pid in rankings:
        return rankings[pid]
    rank = player_state["last_rank"].get(pid, 500)
    points = player_state["last_rank_pts"].get(pid, 0)
    return rank, points


def _player_height_and_age(pid, players_df):
    """Return ``(height_cm, age_years)`` for *pid* from the players CSV."""
    row = players_df[players_df["player_id"] == pid]
    if row.empty:
        return 180.0, 25.0

    height = (
        float(row.iloc[0]["height"])
        if pd.notna(row.iloc[0]["height"])
        else 180.0
    )

    dob = row.iloc[0]["dob"]
    if pd.notna(dob):
        try:
            dob_str = str(int(dob))
            birth = date(int(dob_str[:4]), int(dob_str[4:6]), int(dob_str[6:8]))
            age = (date.today() - birth).days / 365.25
        except (ValueError, IndexError):
            age = 25.0
    else:
        age = 25.0

    return height, age


def build_prediction_vector(
    p1_id, p2_id, surface, player_state, players_df, rankings, feature_names
):
    """Build a single feature vector for a hypothetical match between p1 and p2.

    All values are expressed as ``player1 - player2`` differences, matching
    the convention used during training.

    Parameters
    ----------
    p1_id, p2_id : int
        ATP player IDs.
    surface : str
        One of ``"hard"``, ``"clay"``, or ``"grass"`` (case-insensitive).
    player_state : dict
        Output of :func:`build_features_and_state`.
    players_df : pd.DataFrame
        Output of :func:`load_players`.
    rankings : dict
        Output of :func:`load_rankings`.
    feature_names : list[str]
        Ordered list of feature names from :func:`train_model`.

    Returns
    -------
    np.ndarray
        1-D float array aligned with *feature_names*.
    """
    surf = surface.capitalize()
    st = player_state

    rank1, pts1 = _player_rank_and_points(p1_id, rankings, st)
    rank2, pts2 = _player_rank_and_points(p2_id, rankings, st)
    height1, age1 = _player_height_and_age(p1_id, players_df)
    height2, age2 = _player_height_and_age(p2_id, players_df)

    vec = {}

    # Static differences
    vec["ATP_POINT_DIFF"] = pts1 - pts2
    vec["ATP_RANK_DIFF"] = rank1 - rank2
    vec["AGE_DIFF"] = age1 - age2
    vec["HEIGHT_DIFF"] = height1 - height2
    vec["BEST_OF"] = 3
    vec["DRAW_SIZE"] = 128

    # Head-to-head
    vec["H2H_DIFF"] = (
        st["h2h"].get((p1_id, p2_id), 0)
        - st["h2h"].get((p2_id, p1_id), 0)
    )
    vec["H2H_SURFACE_DIFF"] = (
        st["h2h_surf"][surf].get((p1_id, p2_id), 0)
        - st["h2h_surf"][surf].get((p2_id, p1_id), 0)
    )
    vec["DIFF_N_GAMES"] = (
        st["matches_played"].get(p1_id, 0)
        - st["matches_played"].get(p2_id, 0)
    )

    # Rolling win rates
    for k in WIN_RATE_WINDOWS:
        dk = st["wins_last_k"][k]
        h1 = dk.get(p1_id, deque())
        h2 = dk.get(p2_id, deque())
        vec[f"WIN_LAST_{k}_DIFF"] = (
            (_mean(h1) if h1 else 0) - (_mean(h2) if h2 else 0)
        )

    # Rolling serve stats
    for k in SERVE_STAT_WINDOWS:
        sk = st["serve_last_k"][k]
        for stat, prefix in SERVE_STAT_PREFIX.items():
            v1 = _mean(sk[p1_id][stat]) if p1_id in sk else 0.5
            v2 = _mean(sk[p2_id][stat]) if p2_id in sk else 0.5
            vec[f"{prefix}_{k}_DIFF"] = v1 - v2

    # ELO
    vec["ELO_DIFF"] = (
        st["elo"].get(p1_id, ELO_DEFAULT)
        - st["elo"].get(p2_id, ELO_DEFAULT)
    )
    vec["ELO_SURFACE_DIFF"] = (
        st["elo_surf"][surf].get(p1_id, ELO_DEFAULT)
        - st["elo_surf"][surf].get(p2_id, ELO_DEFAULT)
    )

    # ELO gradients
    default_history = np.array([ELO_DEFAULT, ELO_DEFAULT])
    for n in ELO_GRAD_WINDOWS:
        gstate = st["elo_grad"][n]
        h1 = (
            np.array(gstate[p1_id])
            if p1_id in gstate and len(gstate[p1_id]) >= 2
            else default_history
        )
        h2 = (
            np.array(gstate[p2_id])
            if p2_id in gstate and len(gstate[p2_id]) >= 2
            else default_history
        )
        slope1 = np.polyfit(np.arange(len(h1)), h1, 1)[0] if len(h1) >= 2 else 0
        slope2 = np.polyfit(np.arange(len(h2)), h2, 1)[0] if len(h2) >= 2 else 0
        vec[f"ELO_GRAD_{n}_DIFF"] = slope1 - slope2

    return np.array([vec[f] for f in feature_names], dtype=float)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict the winner of an ATP tennis match."
    )
    parser.add_argument("player1", help="Full name of the first player")
    parser.add_argument("player2", help="Full name of the second player")
    parser.add_argument(
        "--surface",
        default="hard",
        choices=["hard", "clay", "grass"],
        help="Court surface (default: hard)",
    )
    args = parser.parse_args()

    players_df = load_players()
    rankings = load_rankings()

    try:
        p1_id, p1_name = find_player(args.player1, players_df)
        p2_id, p2_name = find_player(args.player2, players_df)
    except ValueError as err:
        print(f"Error: {err}")
        sys.exit(1)

    surface_label = args.surface.capitalize()
    print(f"\nPredicting: {p1_name}  vs  {p2_name}  (surface: {surface_label})\n")

    df = load_match_data()
    feature_df, player_state = build_features_and_state(df)
    model, feature_names = train_model(feature_df)

    x = build_prediction_vector(
        p1_id, p2_id, args.surface,
        player_state, players_df, rankings, feature_names,
    )

    proba = model.predict_proba(x.reshape(1, -1))[0]
    classes = list(model.classes_)
    p1_prob = proba[classes.index("Player 1 Wins")]
    p2_prob = 1 - p1_prob

    winner = p1_name if p1_prob >= 0.5 else p2_name
    confidence = max(p1_prob, p2_prob)

    print("\n" + "=" * 50)
    print(f"  Expected winner: {winner}")
    print(f"  Confidence:      {confidence:.1%}")
    print(f"  {p1_name}: {p1_prob:.1%}  |  {p2_name}: {p2_prob:.1%}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
