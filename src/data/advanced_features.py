import numpy as np
import pandas as pd

ADVANCED_STAT_COLS: list[str] = [
    "POSS", "PACE", "EFG_PCT", "TOV_PCT", "OREB_PCT",
    "FT_RATE", "ORTG", "DRTG", "NET_RATING"
]

def compute_four_factors_and_pace(team_df: pd.DataFrame) -> pd.DataFrame:
    df = team_df.copy()
    
    poss = df["FGA"] + 0.44 * df["FTM"] - df["OREB"] + df["TOV"]
    poss = poss.clip(lower=60.0, upper=140.0)
    
    df["POSS"] = poss
    df["PACE"] = poss
    df["EFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / (df["FGA"] + 1e-6)
    df["TOV_PCT"] = df["TOV"] / (poss + 1e-6)
    df["OREB_PCT"] = df["OREB"] / (df["OREB"] + df["DREB"] + 1e-6)
    df["FT_RATE"] = df["FTM"] / (df["FGA"] + 1e-6)
    df["ORTG"] = 100.0 * df["PTS"] / (poss + 1e-6)
    df["DRTG"] = 100.0 * df["PTS_ALLOWED"] / (poss + 1e-6)
    df["NET_RATING"] = df["ORTG"] - df["DRTG"]

    for col in ADVANCED_STAT_COLS:
        df[col] = df[col].fillna(0.0)

    return df

def compute_elo_ratings(matchups_df: pd.DataFrame) -> pd.DataFrame:
    df = matchups_df.sort_values(["GAME_DATE", "GAME_ID"]).copy().reset_index(drop=True)
    
    team_elos: dict[int, float] = {}
    current_season: str | None = None
    
    elo_home_pre = np.zeros(len(df), dtype=float)
    elo_away_pre = np.zeros(len(df), dtype=float)
    elo_diff_pre = np.zeros(len(df), dtype=float)
    elo_prob_pre = np.zeros(len(df), dtype=float)

    for i, row in df.iterrows():
        season = row["SEASON"]
        if season != current_season:
            current_season = season
            for t_id in team_elos:
                team_elos[t_id] = 0.75 * team_elos[t_id] + 0.25 * 1500.0

        h_id = row["TEAM_ID_HOME"]
        a_id = row["TEAM_ID_AWAY"]
        
        r_h = team_elos.get(h_id, 1500.0)
        r_a = team_elos.get(a_id, 1500.0)
        
        r_h_adj = r_h + 100.0
        diff = r_h_adj - r_a
        e_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

        elo_home_pre[i] = r_h
        elo_away_pre[i] = r_a
        elo_diff_pre[i] = diff
        elo_prob_pre[i] = e_h

        home_won = bool(row["TARGET_HOME_W"] == 1)
        s_h = 1.0 if home_won else 0.0
        
        pts_h = float(row.get("PTS_HOME", 100.0))
        pts_a = float(row.get("PTS_AWAY", 100.0))
        mov = max(abs(pts_h - pts_a), 1.0)
        
        elo_diff_winner = diff if home_won else -diff
        elo_diff_winner = max(elo_diff_winner, 0.0)
        
        k_val = 20.0 * ((mov + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)
        delta_r = k_val * (s_h - e_h)

        team_elos[h_id] = r_h + delta_r
        team_elos[a_id] = r_a - delta_r

    df["ELO_HOME"] = elo_home_pre
    df["ELO_AWAY"] = elo_away_pre
    df["ELO_DIFF"] = elo_diff_pre
    df["ELO_EXPECTED_PROB"] = elo_prob_pre

    return df

def compute_head_to_head_features(matchups_df: pd.DataFrame) -> pd.DataFrame:
    df = matchups_df.sort_values(["GAME_DATE", "GAME_ID"]).copy().reset_index(drop=True)
    
    h2h_history: dict[tuple[int, int], list[tuple[int, float]]] = {}
    
    h2h_win_rate = np.zeros(len(df), dtype=float)
    h2h_point_diff = np.zeros(len(df), dtype=float)
    h2h_games_count = np.zeros(len(df), dtype=float)

    for i, row in df.iterrows():
        h_id = row["TEAM_ID_HOME"]
        a_id = row["TEAM_ID_AWAY"]
        pair_key = (min(h_id, a_id), max(h_id, a_id))

        history = h2h_history.get(pair_key, [])
        if not history:
            h2h_win_rate[i] = 0.5
            h2h_point_diff[i] = 0.0
            h2h_games_count[i] = 0.0
        else:
            recent = history[-5:]
            h_wins = sum(1 for (winner_id, diff) in recent if winner_id == h_id)
            h_pt_diff = sum(diff if winner_id == h_id else -diff for (winner_id, diff) in recent)
            
            h2h_win_rate[i] = h_wins / len(recent)
            h2h_point_diff[i] = h_pt_diff / len(recent)
            h2h_games_count[i] = min(len(history), 10.0)

        home_won = bool(row["TARGET_HOME_W"] == 1)
        winner = h_id if home_won else a_id
        pts_h = float(row.get("PTS_HOME", 100.0))
        pts_a = float(row.get("PTS_AWAY", 100.0))
        diff_pts = abs(pts_h - pts_a)

        if pair_key not in h2h_history:
            h2h_history[pair_key] = []
        h2h_history[pair_key].append((winner, diff_pts))

    df["H2H_WIN_RATE"] = h2h_win_rate
    df["H2H_POINT_DIFF"] = h2h_point_diff
    df["H2H_GAMES_COUNT"] = h2h_games_count

    return df
