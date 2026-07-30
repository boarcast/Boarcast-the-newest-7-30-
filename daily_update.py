# -*- coding: utf-8 -*-
"""
scripts/daily_update.py
========================
【模組一：100% 全自動化數據增量更新系統】（2026 賽季支援修訂版）

核心邏輯：
    拒絕每次重複下載整季數據。只抓取「昨天（today - 1 day，以美東時間為基準）」
    的單日全量逐球數據，與既有的歷史 Parquet 檔案進行增量合併，並依
    game_pk + at_bat_number + pitch_number 去除重複列，確保資料唯一性。

【2026 修訂重點】
    [Rev 1] 年份/路徑動態化：不再寫死單一 statcast_current_season.parquet，
            改以「資料所屬球季年份」自動存到 data/statcast_{year}.parquet，
            避免跨年度（例如 2025 賽季尾聲資料與 2026 開季資料）被混進同一個檔案。
            另外維護一份 data/statcast_current_season.parquet 作為「當前賽季」的
            相容捷徑（symlink，若平台不支援 symlink 則退回複製檔案），舊版前端／
            分析腳本若還在讀取這個固定檔名，也能自動指向最新賽季。
    [Rev 2] 開季前 / 休賽期保護邏輯：
            - pybaseball.statcast() 回傳空 DataFrame 時（春訓、休賽日、賽季未開打），
              視為正常情況，靜默略過寫檔，不中斷、不報錯。
            - 若目前月份落在休賽季（12 月～隔年 2 月），預設執行（不帶任何日期參數）
              會直接跳過抓取，避免對 Baseball Savant 發送注定落空的請求。
              （可用 --force 略過此保護，供除錯或手動補抓使用）
    [Rev 3] 補抓區間預設值升級：
            新增 --season 參數；只給 --season 而未給 --start/--end 時，
            自動以「該年 MLB 開幕日」（含海外開幕系列賽）作為補抓起始日，
            至今天（ET）或球季結束日之一，較早者為結束日。

用法：
    python scripts/daily_update.py                       # 抓取「美東時間昨天」
    python scripts/daily_update.py --date 2026-04-01      # 手動指定單日（除錯 / 補抓用）
    python scripts/daily_update.py --start 2026-03-26 --end 2026-04-03  # 補抓區間
    python scripts/daily_update.py --season 2026          # 補抓整個 2026 賽季至今
    python scripts/daily_update.py --force                # 忽略休賽季保護，強制執行

輸出：
    data/statcast_{year}.parquet             （逐年份存檔，主要資料來源）
    data/statcast_current_season.parquet     （指向當前賽季檔案的相容捷徑）
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from pybaseball import statcast
    import pybaseball
    pybaseball.cache.enable()
except ImportError:
    print("❌ 找不到 pybaseball，請先 pip install pybaseball", file=sys.stderr)
    raise

# ==========================================================
# 常數設定
# ==========================================================

# 專案根目錄 = 這支腳本所在目錄的上一層 (scripts/ 的上一層)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# [Rev 1] 相容捷徑：永遠指向「當前賽季」的 parquet 檔案，供尚未升級的
# 下游程式（如舊版 anomaly_detector 或前端）讀取固定檔名時使用。
CURRENT_SEASON_ALIAS_PATH = DATA_DIR / "statcast_current_season.parquet"


def season_parquet_path(year: int) -> Path:
    """回傳指定球季年份對應的 Parquet 檔案路徑（動態年份路徑，取代寫死檔名）。"""
    return DATA_DIR / f"statcast_{year}.parquet"


# 用來判斷「唯一一球」的複合鍵（模組需求：game_pk + at_bat_number + pitch_number）
DEDUPE_KEYS = ["game_pk", "at_bat_number", "pitch_number"]

# 若來源資料含有 play_id 欄位，優先改用 play_id 當唯一鍵（更嚴謹）
FALLBACK_DEDUPE_KEY = "play_id"

# [Rev 3] 已知的球季開幕日（涵蓋東京 / 首爾等海外開幕系列賽提前開打的年度）。
# 若當年度不在此表中，會自動退回「3 月最後一個週四」的粗略估算值。
SEASON_OPENING_DAY = {
    2024: "2024-03-20",  # 首爾海外開幕系列賽
    2025: "2025-03-18",  # 東京海外開幕系列賽
    2026: "2026-03-17",  # 預告東京海外開幕系列賽（實際日期以 MLB 官方公告為準）
}
DEFAULT_SEASON_END_MONTH_DAY = "11-05"  # 涵蓋例行賽 + 季後賽尾聲的保守估計

# 休賽季月份（12 月、1 月、2 月通常沒有 MLB 例行賽事）
OFFSEASON_MONTHS = {12, 1, 2}

# 只保留分析所需欄位，縮小 Parquet 體積、加快前端讀取速度
# （模組二 Anomaly Detection ／ 模組三 Matchup Edge ／ 模組四 Custom Stuff+ 都會用到）
KEEP_COLUMNS = [
    # 場次 / 打席識別
    "game_pk", "game_date", "at_bat_number", "pitch_number", "play_id",
    "home_team", "away_team", "inning", "inning_topbot",
    # 投打對戰身份
    "pitcher", "player_name", "batter", "batter_name",
    "p_throws", "stand",
    # 球種與物理特徵（模組四會用到）
    "pitch_type", "pitch_name",
    "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "plate_x", "plate_z",
    # 球數 / 結果（模組二會用到）
    "balls", "strikes", "description", "events", "type", "zone",
    # 打擊結果（模組二、三會用到）
    "launch_speed", "launch_angle", "bb_type",
    "estimated_ba_using_speedangle", "estimated_slg_using_speedangle",
]


def now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def get_yesterday_et() -> str:
    """回傳「美東時間昨天」的日期字串 (YYYY-MM-DD)。"""
    yesterday_et = now_et() - timedelta(days=1)
    return yesterday_et.strftime("%Y-%m-%d")


def season_opening_day(year: int) -> str:
    """回傳指定年度的球季開幕日估計值；若無公告資料則退回 3 月最後一個週四的粗估值。"""
    if year in SEASON_OPENING_DAY:
        return SEASON_OPENING_DAY[year]

    d = datetime(year, 3, 31)
    while d.weekday() != 3:  # 3 = Thursday
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def is_offseason(dt: datetime) -> bool:
    return dt.month in OFFSEASON_MONTHS


def fetch_statcast_range(start_dt: str, end_dt: str) -> pd.DataFrame:
    """呼叫 pybaseball.statcast() 抓取指定區間的逐球數據。"""
    print(f"📡 正在抓取 Statcast 逐球數據：{start_dt} ~ {end_dt} ...")
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    if df is None or df.empty:
        # [Rev 2] 開季前 / 休賽日視為正常情況，不視為錯誤，靜默略過即可。
        print(f"ℹ️ {start_dt} ~ {end_dt} 沒有抓到任何資料（春訓、休賽日或賽季尚未開打，屬正常情況，略過寫檔）。")
        return pd.DataFrame()
    print(f"✅ 抓到 {len(df):,} 筆逐球紀錄。")
    return df


def _restrict_columns(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 KEEP_COLUMNS 中，且實際存在於 df 的欄位。"""
    existing = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        print(f"ℹ️ 來源資料缺少以下欄位（將略過）：{missing}")
    return df[existing].copy()


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """依 play_id（若存在）或 game_pk+at_bat_number+pitch_number 去重複。"""
    before = len(df)
    if FALLBACK_DEDUPE_KEY in df.columns and df[FALLBACK_DEDUPE_KEY].notna().any():
        df = df.drop_duplicates(subset=[FALLBACK_DEDUPE_KEY], keep="last")
    else:
        keys = [k for k in DEDUPE_KEYS if k in df.columns]
        df = df.drop_duplicates(subset=keys, keep="last")
    after = len(df)
    if before != after:
        print(f"🧹 去重複：{before:,} → {after:,} 筆（移除 {before - after:,} 筆重複資料）。")
    return df


def _update_current_season_alias(target_path: Path) -> None:
    """[Rev 1] 讓 statcast_current_season.parquet 永遠指向最新一次寫入的賽季檔案。

    優先嘗試建立 symlink（節省磁碟空間、單一事實來源）；若執行環境不支援
    symlink（例如某些 Windows 設定或檔案系統限制），則退回直接複製檔案，
    確保下游程式仍能穩定讀到正確內容。
    """
    try:
        if CURRENT_SEASON_ALIAS_PATH.exists() or CURRENT_SEASON_ALIAS_PATH.is_symlink():
            CURRENT_SEASON_ALIAS_PATH.unlink()
        CURRENT_SEASON_ALIAS_PATH.symlink_to(target_path.name)
        print(f"🔗 已更新相容捷徑 {CURRENT_SEASON_ALIAS_PATH.name} -> {target_path.name}")
    except (OSError, NotImplementedError):
        shutil.copyfile(target_path, CURRENT_SEASON_ALIAS_PATH)
        print(f"📄 環境不支援 symlink，已改用複製更新相容捷徑 {CURRENT_SEASON_ALIAS_PATH.name}")


def merge_and_save(new_df: pd.DataFrame, season_year: int) -> None:
    """將新抓取的資料與該球季既有的歷史 Parquet 增量合併並寫回。

    [Rev 1] 目標檔案改為依 season_year 動態決定（data/statcast_{year}.parquet），
    不再寫死單一檔名，避免跨年度資料混合或彼此覆蓋。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_df = _restrict_columns(new_df)
    target_path = season_parquet_path(season_year)

    if target_path.exists():
        print(f"📂 讀取既有歷史檔案：{target_path}")
        old_df = pd.read_parquet(target_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        print(f"📂 尚無 {season_year} 賽季歷史檔案，將建立新的 Parquet 檔案。")
        combined = new_df

    combined = _dedupe(combined)

    # 依日期排序，方便後續增量檢查與除錯
    if "game_date" in combined.columns:
        combined = combined.sort_values("game_date").reset_index(drop=True)

    combined.to_parquet(target_path, index=False)
    print(f"💾 已寫回 {target_path}，目前總筆數：{len(combined):,} 筆。")

    _update_current_season_alias(target_path)


def _infer_season_year(start_dt: str, end_dt: str) -> int:
    """從補抓/更新的日期區間推斷所屬球季年份。

    多日跨年區間極為罕見（MLB 賽季不會橫跨曆年），保守起見一律採用起始日
    的年份；若未來真的需要橫跨年度補抓，可分批以單一年度範圍呼叫本腳本。
    """
    return int(start_dt.split("-")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB Statcast 逐日增量更新腳本（2026 賽季支援修訂版）")
    parser.add_argument("--date", help="手動指定單一日期 (YYYY-MM-DD)，覆蓋預設的『美東昨天』邏輯")
    parser.add_argument("--start", help="手動指定補抓區間起始日 (YYYY-MM-DD)，需搭配 --end")
    parser.add_argument("--end", help="手動指定補抓區間結束日 (YYYY-MM-DD)，需搭配 --start")
    parser.add_argument("--season", type=int,
                         help="[Rev 3] 指定球季年份，自動從該年開幕日補抓至今天（ET）或球季結束日，較早者為準；"
                              "若同時提供 --start/--end 則以 --start/--end 優先")
    parser.add_argument("--force", action="store_true",
                         help="[Rev 2] 忽略休賽季（12 月～隔年 2 月）自動跳過保護，強制執行抓取")
    args = parser.parse_args()

    current_et = now_et()

    if args.start and args.end:
        start_dt, end_dt = args.start, args.end
    elif args.date:
        start_dt = end_dt = args.date
    elif args.season:
        start_dt = season_opening_day(args.season)
        season_end = f"{args.season}-{DEFAULT_SEASON_END_MONTH_DAY}"
        today_str = current_et.strftime("%Y-%m-%d")
        end_dt = min(season_end, today_str)
    else:
        # [Rev 2] 預設（無任何日期參數）執行時的休賽季保護：
        # 12 月～隔年 2 月通常沒有例行賽逐球資料，直接跳過可省下白跑一趟的 API 請求。
        if is_offseason(current_et) and not args.force:
            print(
                f"🛌 目前為休賽季（{current_et.strftime('%Y-%m')}），MLB 通常沒有例行賽 Statcast 資料，"
                f"本次自動略過抓取。如需強制執行請加上 --force 參數。"
            )
            return
        start_dt = end_dt = get_yesterday_et()

    new_df = fetch_statcast_range(start_dt, end_dt)
    if new_df.empty:
        print("ℹ️ 本次無新資料需要合併，結束。")
        return

    season_year = _infer_season_year(start_dt, end_dt)
    merge_and_save(new_df, season_year)


if __name__ == "__main__":
    main()
