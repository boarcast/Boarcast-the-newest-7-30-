# -*- coding: utf-8 -*-
"""
scripts/anomaly_detector.py
=============================
【模組二：今日/本週最奇葩數據自動偵測（Anomaly Detection）】（2026 賽季支援修訂版）

在每日增量更新（daily_update.py）完成後執行，對截至目前為止的全賽季逐球
數據進行多維度群組聚合，並用統計方法（Z-Score）與機器學習方法（Isolation
Forest）挖出「出現機率極低」的極端離群值事件，寫入 data/anomaly_today.json
供前端「今日最奇葩數據」專區使用。

三個聚合維度：
    A（球種對決）  groupby(batter_name, p_throws, pitch_type)      → 總球數
    B（球數對決）  groupby(batter_name, balls, strikes, pitch_type) → 揮棒率 / 揮空率
    C（極端結果）  groupby(pitcher_name, pitch_type)                → 高初速被擊出頻率 (launch_speed > 105)

【2026 修訂重點】
    [Rev 1] 資料來源改為讀取「動態當前賽季」的年度 Parquet 檔案
            （data/statcast_{year}.parquet，year 由系統時間動態決定，不再寫死年份），
            找不到時才退回讀取 data/statcast_current_season.parquet 相容捷徑。
    [Rev 2] 年度過濾機制：即使讀到的檔案內混有跨年度資料，聚合前一律先篩選
            game_date 屬於當前賽季年份的列，避免新賽季初期統計混入舊年度基準。
    [Rev 3] 賽季初期樣本量保護：若當前賽季總球數低於 MIN_SEASON_PITCHES_FOR_ANALYSIS
            （預設 5,000 球），Z-Score / Isolation Forest 皆无法產生有意義的統計結果
            （標準差易趨近 0、matrix 過度稀疏），此時直接跳過計算，改寫入
            {"status": "insufficient_data", ...} 供前端顯示「賽季初期資料累積中」，
            不再讓除以零或空矩陣擬合造成例外。

輸出：
    data/anomaly_today.json  —  最奇葩的 3 個事件（JSON array），
                                  或賽季初期時的 {"status": "insufficient_data", ...} 物件

用法：
    python scripts/anomaly_detector.py
    python scripts/anomaly_detector.py --method isolation_forest   # 改用 ML 方法
    python scripts/anomaly_detector.py --top-n 5                   # 想看更多候選事件
    python scripts/anomaly_detector.py --season 2026               # 手動指定分析年度
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================
# 常數設定
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CURRENT_SEASON_ALIAS_PATH = DATA_DIR / "statcast_current_season.parquet"
OUTPUT_PATH = DATA_DIR / "anomaly_today.json"

# 樣本門檻：該打者面對該投手手性的「總看球數」至少達到此門檻，
# 才有資格被視為「應該有一定樣本量，卻在特定球種上極端偏低」的事件
# （避免把「賽季初期資料還很少」誤判為離群值）。
MIN_TOTAL_PITCHES_VS_HANDEDNESS = 150

# [Rev 3] 賽季初期整體樣本保護門檻：全賽季（截至目前）逐球總數低於此值時，
# 直接跳過統計/ML 計算，避免除以零或矩陣過度稀疏導致的例外或無意義結果。
MIN_SEASON_PITCHES_FOR_ANALYSIS = 5000

# Z-Score 門檻：低於此值視為統計上的極端離群
Z_SCORE_THRESHOLD = -2.5

# 高初速定義（模組三/模組二共用標準）
HARD_HIT_LAUNCH_SPEED = 105.0

# 揮棒 / 揮空事件的 description 分類（Statcast description 欄位常見值）
SWING_DESCRIPTIONS = {
    "hit_into_play", "foul", "foul_tip", "swinging_strike",
    "swinging_strike_blocked", "missed_bunt", "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


# ==========================================================
# 資料載入
# ==========================================================

def _season_parquet_path(year: int) -> Path:
    return DATA_DIR / f"statcast_{year}.parquet"


def load_data(season: int | None) -> tuple[pd.DataFrame, int]:
    """[Rev 1] 載入指定（或動態推斷的當前）賽季資料。

    優先順序：
        1. 明確指定的 --season 年度檔案 data/statcast_{season}.parquet
        2. 若未指定，動態以系統目前年份（datetime.now().year）尋找對應年度檔案
        3. 找不到動態年度檔案時，退回讀取相容捷徑 statcast_current_season.parquet
           （由 daily_update.py 的 [Rev 1] 機制維護，永遠指向最新賽季）
    """
    target_year = season if season is not None else datetime.now().year
    candidate = _season_parquet_path(target_year)

    if candidate.exists():
        df = pd.read_parquet(candidate)
        print(f"📂 已載入 {target_year} 賽季資料 {candidate.name}：{len(df):,} 筆逐球紀錄。")
        return df, target_year

    if CURRENT_SEASON_ALIAS_PATH.exists():
        df = pd.read_parquet(CURRENT_SEASON_ALIAS_PATH)
        inferred_year = target_year
        if "game_date" in df.columns and not df["game_date"].isna().all():
            inferred_year = int(pd.to_datetime(df["game_date"]).dt.year.max())
        print(f"📂 找不到 {candidate.name}，改讀取相容捷徑 {CURRENT_SEASON_ALIAS_PATH.name}："
              f"{len(df):,} 筆逐球紀錄（推斷年度：{inferred_year}）。")
        return df, inferred_year

    print(f"❌ 找不到 {candidate} 或 {CURRENT_SEASON_ALIAS_PATH}，請先執行 scripts/daily_update.py 累積資料。",
          file=sys.stderr)
    sys.exit(1)


def _filter_to_season(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """[Rev 2] 年度過濾：只保留 game_date 屬於指定年度的列，避免混入跨年度資料。"""
    if "game_date" not in df.columns:
        return df
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    return df[dates.dt.year == year].copy()


# ==========================================================
# 維度 A：球種對決（batter x p_throws x pitch_type）→ 總球數 + Z-Score
# ==========================================================

def dimension_a_zscore(df: pd.DataFrame) -> pd.DataFrame:
    required = ["batter_name", "p_throws", "pitch_type"]
    if not all(c in df.columns for c in required):
        print("⚠️ 維度 A 所需欄位不足，略過。")
        return pd.DataFrame()

    grp = (
        df.dropna(subset=required)
        .groupby(required)
        .size()
        .reset_index(name="count")
    )

    # 該打者面對該投手手性的「總看球數」= 同一 batter_name + p_throws 底下所有球種加總
    totals = (
        grp.groupby(["batter_name", "p_throws"])["count"]
        .transform("sum")
    )
    grp["total_pitches_vs_handedness"] = totals

    # 對每個 (batter_name, p_throws) 群組，計算其底下各球種球數的平均與標準差，
    # 藉此找出「這位打者面對這個手性時，某一球種球數遠低於自己其他球種平均值」的事件。
    # 注意：刻意使用 transform 而非 groupby().apply()，因為 pandas 2.2+ 的
    # DataFrameGroupBy.apply() 預設會把分組欄位（batter_name / p_throws）從
    # 傳入的子表中排除（需另外設定 include_groups=True 才會保留），容易在
    # 後續程式碼引用這些欄位時產生難以察覺的 KeyError；用 transform 完全避開此陷阱。
    group_key = ["batter_name", "p_throws"]
    group_mean = grp.groupby(group_key)["count"].transform("mean")
    group_std = grp.groupby(group_key)["count"].transform("std", ddof=0)
    grp["z_score"] = np.where(
        (group_std == 0) | group_std.isna(),
        0.0,
        (grp["count"] - group_mean) / group_std,
    )

    # 套用樣本門檻 + Z-Score 門檻
    outliers = grp[
        (grp["total_pitches_vs_handedness"] >= MIN_TOTAL_PITCHES_VS_HANDEDNESS)
        & (grp["z_score"] <= Z_SCORE_THRESHOLD)
    ].copy()

    outliers = outliers.sort_values("z_score")
    return outliers


# ==========================================================
# 維度 B：球數對決（batter x balls x strikes x pitch_type）→ 揮棒率 / 揮空率
# ==========================================================

def dimension_b_swing_whiff(df: pd.DataFrame) -> pd.DataFrame:
    required = ["batter_name", "balls", "strikes", "pitch_type", "description"]
    if not all(c in df.columns for c in required):
        print("⚠️ 維度 B 所需欄位不足，略過。")
        return pd.DataFrame()

    work = df.dropna(subset=required).copy()
    work["is_swing"] = work["description"].isin(SWING_DESCRIPTIONS)
    work["is_whiff"] = work["description"].isin(WHIFF_DESCRIPTIONS)

    grp = (
        work.groupby(["batter_name", "balls", "strikes", "pitch_type"])
        .agg(pitch_count=("description", "size"),
             swings=("is_swing", "sum"),
             whiffs=("is_whiff", "sum"))
        .reset_index()
    )
    grp["swing_pct"] = grp["swings"] / grp["pitch_count"]
    grp["whiff_pct"] = np.where(grp["swings"] > 0, grp["whiffs"] / grp["swings"], np.nan)
    return grp


# ==========================================================
# 維度 C：極端結果（pitcher x pitch_type）→ 高初速被擊出頻率
# ==========================================================

def dimension_c_hard_hit_freq(df: pd.DataFrame) -> pd.DataFrame:
    name_col = "player_name" if "player_name" in df.columns else "pitcher_name"
    required = [name_col, "pitch_type", "launch_speed"]
    if not all(c in df.columns for c in required):
        print("⚠️ 維度 C 所需欄位不足，略過。")
        return pd.DataFrame()

    work = df.dropna(subset=[name_col, "pitch_type"]).copy()
    work["is_hard_hit"] = work["launch_speed"] > HARD_HIT_LAUNCH_SPEED

    grp = (
        work.groupby([name_col, "pitch_type"])
        .agg(total_pitches=("pitch_type", "size"),
             hard_hit_count=("is_hard_hit", "sum"),
             balls_in_play=("launch_speed", lambda s: s.notna().sum()))
        .reset_index()
        .rename(columns={name_col: "pitcher_name"})
    )
    grp = grp[grp["balls_in_play"] > 0].copy()
    grp["hard_hit_freq"] = grp["hard_hit_count"] / grp["balls_in_play"]
    return grp.sort_values("hard_hit_freq", ascending=False)


# ==========================================================
# 離群值篩選：方法一（Z-Score，主要方法，對應規格範例）
# ==========================================================

def top_events_zscore(df: pd.DataFrame, year: int, top_n: int) -> list[dict]:
    outliers = dimension_a_zscore(df)
    if outliers.empty:
        return []

    events = []
    for _, row in outliers.head(top_n).iterrows():
        platoon_label = "LHP" if row["p_throws"] == "L" else "RHP"
        pitch_label = row["pitch_type"]
        events.append({
            "player": row["batter_name"],
            "year": year,
            "metric": f"vs {platoon_label} {pitch_label}",
            "count": int(row["count"]),
            "description": (
                f"在 {year} 賽季面對{'左投手' if platoon_label == 'LHP' else '右投手'}的 "
                f"{pitch_label} 總共只看了 {int(row['count'])} 顆球"
                f"（同手性總球數 {int(row['total_pitches_vs_handedness'])} 顆，"
                f"Z-Score={row['z_score']:.2f}），屬於極端左右分工或使用率偏低的罕見數據。"
            ),
        })
    return events


# ==========================================================
# 離群值篩選：方法二（Isolation Forest，機器學習方法）
# ==========================================================

def top_events_isolation_forest(df: pd.DataFrame, year: int, top_n: int) -> list[dict]:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        print("⚠️ 找不到 scikit-learn，請先 pip install scikit-learn。改用 Z-Score 方法。")
        return top_events_zscore(df, year, top_n)

    required = ["batter_name", "p_throws", "pitch_type"]
    if not all(c in df.columns for c in required):
        print("⚠️ Isolation Forest 所需欄位不足，改用 Z-Score 方法。")
        return top_events_zscore(df, year, top_n)

    grp = (
        df.dropna(subset=required)
        .groupby(required)
        .size()
        .reset_index(name="count")
    )
    grp["total_pitches_vs_handedness"] = grp.groupby(["batter_name", "p_throws"])["count"].transform("sum")
    grp["pitch_pct"] = grp["count"] / grp["total_pitches_vs_handedness"]

    # 只考慮樣本量足夠的組合，避免把賽季初期小樣本誤判為異常
    grp = grp[grp["total_pitches_vs_handedness"] >= MIN_TOTAL_PITCHES_VS_HANDEDNESS].copy()
    if grp.empty:
        return []

    features = grp[["count", "total_pitches_vs_handedness", "pitch_pct"]].to_numpy()

    model = IsolationForest(contamination="auto", random_state=42)
    model.fit(features)
    # decision_function 越低代表越異常
    grp["anomaly_score"] = model.decision_function(features)
    grp = grp.sort_values("anomaly_score")

    events = []
    for _, row in grp.head(top_n).iterrows():
        platoon_label = "LHP" if row["p_throws"] == "L" else "RHP"
        pitch_label = row["pitch_type"]
        events.append({
            "player": row["batter_name"],
            "year": year,
            "metric": f"vs {platoon_label} {pitch_label}",
            "count": int(row["count"]),
            "description": (
                f"（Isolation Forest 偵測）在 {year} 賽季面對"
                f"{'左投手' if platoon_label == 'LHP' else '右投手'}的 {pitch_label}，"
                f"使用率或球數呈現異常樣態（總球數 {int(row['count'])} / "
                f"同手性總球數 {int(row['total_pitches_vs_handedness'])}，"
                f"異常分數={row['anomaly_score']:.3f}）。"
            ),
        })
    return events


# ==========================================================
# 主流程
# ==========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MLB 奇葩數據自動偵測腳本（2026 賽季支援修訂版）")
    parser.add_argument("--method", choices=["zscore", "isolation_forest"], default="zscore",
                         help="離群值篩選方法（預設 zscore，對應規格範例的統計學方法）")
    parser.add_argument("--top-n", type=int, default=3, help="輸出前 N 個最奇葩事件（預設 3）")
    parser.add_argument("--season", type=int, default=None,
                         help="[Rev 1] 手動指定分析年度；未指定時動態採用系統目前年份")
    args = parser.parse_args()

    df, year = load_data(args.season)
    df = _filter_to_season(df, year)

    total_pitches = len(df)
    print(f"📊 {year} 賽季目前累積逐球數：{total_pitches:,} 球。")

    # [Rev 3] 賽季初期樣本量保護：資料量不足以支撐統計 / ML 計算時，
    # 直接寫入狀態物件並結束，避免除以零或空矩陣擬合造成例外。
    if total_pitches < MIN_SEASON_PITCHES_FOR_ANALYSIS:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        status_payload = {
            "status": "insufficient_data",
            "year": year,
            "total_pitches": total_pitches,
            "min_required": MIN_SEASON_PITCHES_FOR_ANALYSIS,
            "message": f"{year} 賽季資料累積中（目前 {total_pitches:,} / 至少需要 "
                       f"{MIN_SEASON_PITCHES_FOR_ANALYSIS:,} 球），暫不提供奇葩數據偵測。",
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(status_payload, f, ensure_ascii=False, indent=2)
        print(f"🛌 樣本量不足（{total_pitches:,} < {MIN_SEASON_PITCHES_FOR_ANALYSIS:,}），"
              f"已寫入 insufficient_data 狀態至 {OUTPUT_PATH}，略過統計/ML 計算。")
        return

    if args.method == "isolation_forest":
        events = top_events_isolation_forest(df, year, args.top_n)
    else:
        events = top_events_zscore(df, year, args.top_n)

    if not events:
        print("ℹ️ 目前資料量足夠但未找到符合門檻的離群值事件。")
        events = []

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"💾 已寫入 {len(events)} 筆奇葩事件至 {OUTPUT_PATH}")
    for e in events:
        print(f"  - {e['player']} | {e['metric']} | count={e['count']}")


if __name__ == "__main__":
    main()
