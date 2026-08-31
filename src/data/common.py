import pandas as pd

def prepare_raw_team_logs(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ").astype(int)
    df["WIN"] = (df["WL"] == "W").astype(float)
    
    for pct_col in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
        if pct_col in df.columns:
            df[pct_col] = df[pct_col].fillna(0.0)

    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    opp_cols = ["GAME_ID", "TEAM_ID", "PTS", "FGM", "FGA", "FG3M", "FG3A", "DREB", "OREB", "REB"]
    available_opp_cols = [c for c in opp_cols if c in df.columns]
    
    rename_map = {
        "TEAM_ID": "OPP_TEAM_ID",
        "PTS": "PTS_ALLOWED",
        "FGM": "OPP_FGM",
        "FGA": "OPP_FGA",
        "FG3M": "OPP_FG3M",
        "FG3A": "OPP_FG3A",
        "DREB": "OPP_DREB",
        "OREB": "OPP_OREB",
        "REB": "OPP_REB"
    }
    
    opponents = df[available_opp_cols].rename(columns={k: v for k, v in rename_map.items() if k in available_opp_cols})
    merged = df.merge(opponents, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
    merged = merged.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
    return merged
