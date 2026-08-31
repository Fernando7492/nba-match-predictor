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

    opponents = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "PTS_ALLOWED"}
    )
    merged = df.merge(opponents, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
    merged = merged.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
    return merged
