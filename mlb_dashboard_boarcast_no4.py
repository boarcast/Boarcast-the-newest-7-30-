# -*- coding: utf-8 -*-
"""
⚾ MLB 擊球數據模擬與球員視覺化儀表板
============================================
執行方式：
    pip install streamlit plotly pandas numpy pybaseball pillow kaleido requests
    streamlit run mlb_dashboard.py

功能總覽：
    1. 球員搜尋（輸入英文名/姓，或使用熱門球員快捷鍵）
    2. 逐場雙頭滑桿（1~162 場）微觀區間篩選
    3. 「外推至 162 場全賽季」等比例模擬開關
    4. 完整打擊數據面板（傳統成績 + Statcast 物理數據）
    5. 2D 球場擊球落點圖（Spray Chart，支援框選 Box Select）
    6. 8 種模式 3x3 好球帶熱區（數字覆蓋在格子中央）
    7. 一鍵匯出高畫質 PNG Scouting Report 卡片

本版修正紀錄（Bug Fix Log）：
    [Fix 1] 字型路徑改為跨平台搜尋 + 安全 fallback，避免非 Linux/無 DejaVu 環境 crash 或文字縮成極小點
    [Fix 2] 修正盜壘 (SB) / 盜壘失敗 (CS) 事件比對邏輯，改用 str.contains 涵蓋 stolen_base_2b/3b/home 等實際事件值
    [Fix 3] 外推 162 場模擬時，率數據 (AVG/OBP/SLG/OPS/ISO) 改為以外推後的整數重新計算，避免與逐項數據對不齊
    [Fix 4] 熱區空白格改用 np.nan 取代 0，避免無資料格被誤繪為「數值 0」的深色格
    [Fix 5] 側邊欄輸入框改用 key 綁定 st.session_state，讓快捷鍵點擊能正確同步更新輸入框顯示
    [Fix 6] 賽季起始日期提前，涵蓋東京海外開幕賽等 3 月中旬前開打的場次

本版新增優化（Enhancement Log）：
    [Enh 1] Spray Chart 改用「球隊主場球場向量外框」：依打席資料中最常見的 home_team，
            查表帶入該球場左外野線/左中/中外野/右中/右外野線的實際公開距離，繪製貼近真實
            外形的多邊形球場外框，取代先前固定半徑 330 呎的通用弧線。
    [Enh 2] 新增進階期望數據：xBA（估計打擊率）、xSLG（估計長打率，取自 Statcast 的
            estimated_ba_using_speedangle / estimated_slg_using_speedangle）、
            Barrel %（出色擊球率，採近似判定：初速與仰角關係）、
            Hard Hit %（強擊球率，初速 ≥ 95 mph 的球佔全部「有進場」打席比例）。
    [Enh 3] 新增跨頁面/跨模式 session_state 資料快取（_player_cache），單人主板與雙人 PK
            模式共用同一份快取，避免同一位球員/年度重複發送 HTTP 請求。
    [Enh 4] FONT_CANDIDATES 新增 Linux 常見中文字型路徑（Noto Sans CJK / 文泉驛正黑），
            匯出 PNG 報告卡時可正確顯示中文標題，不再出現缺字方塊。

本版第二階段升級（Stage 2 Enhancement Log）：
    [🏷️ Brand] 全新專屬品牌浮水印 boarcast：於個人 Scouting Report 卡、雙人 PK 對決卡、
               IG 限動直式圖卡右下角（或右上角）加入螢光綠 #00E676 主標 + 亮白描邊，
               並附「Powered by boarcast | Data: MLB Statcast」小字說明，強化社群分享識別度。
    [Stage2-1] 新增 IG 限時動態專用圖卡（9:16 直式 1080x1920），精簡排版、放大頭像、
               三圍數據與落點圖，供創作者免裁切一鍵發布。
    [Stage2-2] 微觀時間軸滑桿旁新增「🔥 近 7 場 / ⚡ 近 15 場 / 📊 近 30 場」快捷鍵，
               一鍵鎖定最新場次區間。
    [Stage2-3] 新增擊球型態分佈（GB% 滾地球 / LD% 平飛球 / FB% 飛球 / PU% 內野高飛），
               取自 Statcast bb_type 欄位。
    [Stage2-4] 新增全賽季手感起伏圖：15 場動態滾動 OPS 走勢圖（Rolling OPS Chart），
               以整季逐場分量做滾動加總後正確重算 OPS，避免對率數據直接取平均的偏誤。
"""

import io
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    from pybaseball import playerid_lookup, statcast_batter
    import pybaseball
    pybaseball.cache.enable()
    PYBASEBALL_AVAILABLE = True
except Exception:
    PYBASEBALL_AVAILABLE = False

# ==========================================================
# 常數設定
# ==========================================================

POPULAR_PLAYERS = [
    ("Shohei", "Ohtani"),
    ("Aaron", "Judge"),
    ("Yordan", "Alvarez"),
    ("Mookie", "Betts"),
    ("Juan", "Soto"),
    ("Ronald", "Acuna Jr."),
    ("Freddie", "Freeman"),
    ("Jose", "Ramirez"),
]

HOT_ZONE_METRICS = [
    "打擊率 (AVG)",
    "長打率 (SLG)",
    "純長打率 (ISO)",
    "全壘打 (HR)",
    "三壘安打 (3B)",
    "二壘安打 (2B)",
    "一壘安打 (1B)",
    "平均擊球初速 (Avg EV)",
]

ZONE_MAP = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
}

# 球種代碼 -> 中文顯示名稱（供 #2 球種過濾器使用）
PITCH_TYPE_NAMES = {
    "FF": "四縫線速球 FF", "SI": "伸卡球 SI", "FC": "卡特球 FC",
    "SL": "滑球 SL", "ST": "橫掃滑球 ST", "SV": "慢滑球 SV",
    "CU": "曲球 CU", "KC": "彎曲曲球 KC", "CS": "慢曲球 CS",
    "CH": "變速球 CH", "FS": "指叉球 FS", "FO": "叉指快速球 FO",
    "SC": "螺旋球 SC", "KN": "蝴蝶球 KN", "EP": "落葉球 EP",
    "PO": "投手犯規球 PO", "FA": "快速球(未分類) FA",
}

PLATOON_OPTIONS = ["全部 (All)", "面對左投 vs LHP", "面對右投 vs RHP"]

HIT_EVENTS = ["single", "double", "triple", "home_run"]

# [Fix 6] 起始日期提前至 2 月中旬，涵蓋東京 / 首爾等海外開幕賽場次
SEASON_START_MONTH_DAY = "02-15"
SEASON_END_MONTH_DAY = "11-05"

# [2026 Rev] 動態年份選單：移除寫死的年份清單（先前固定為 [2025, 2024, 2023, 2022]），
# 改以系統目前年份為基準往前推算，讓儀表板每年都能自動把新賽季（如 2026）
# 納入可選範圍，不需要每年手動改程式碼。
YEAR_SELECT_LOOKBACK = 3  # 除了當前年度外，額外往前保留幾個年度供選擇


def get_selectable_years() -> list[int]:
    """回傳可選年度清單，第一個元素（index 0，UI 預設值）永遠是系統目前年份。"""
    current_year = datetime.now().year
    return [current_year - i for i in range(YEAR_SELECT_LOOKBACK + 1)]


# [2026 Rev] 162 場外推模擬保護門檻：樣本場次數低於此值時，不允許啟用外推，
# 避免開季初期（例如只打了 2、3 場）就把數據暴力放大成整季預測，產生失真的
# 「外推打出 100 支全壘打」等荒謬結果。搭配下方既有的 MAX_EXTRAPOLATION_FACTOR
# 倍率上限，形成「門檻 + 上限」雙重防呆。
MIN_GAMES_FOR_EXTRAPOLATION = 10

# 常見跨平台字型候選路徑（依序嘗試），供 [Fix 1] 使用
# [Enh 4] 中文字型優先於西文 DejaVu/Arial 之後、Windows 字型之前插入，
# 涵蓋 Streamlit Community Cloud 等常見 Linux 部署環境常見的 CJK 字型套件
# （fonts-noto-cjk / fonts-wqy-zenhei），避免匯出 PNG 時中文標題出現缺字方塊。
#
# [優化 2] 雲端部署缺中文字型防呆：
#   最優先改嘗試載入「專案內建字型檔」assets/NotoSansTC-Regular.ttf（相對於本檔案所在目錄）。
#   只要把該字型檔案放進專案的 assets/ 資料夾一起部署（例如上傳到 GitHub repo），
#   不論部署到哪種 Linux 環境（Streamlit Community Cloud 等，通常「不保證」系統已安裝
#   任何中文字型），都能穩定顯示繁體中文，不再依賴伺服器系統字型是否存在。
#   若你是部署到 Streamlit Community Cloud 且不想內建字型檔，也可改在專案根目錄
#   新增 packages.txt，內容加入一行 `fonts-noto-cjk`，讓雲端建置時自動 apt-get 安裝，
#   即可命中下方 /usr/share/fonts/.../NotoSansCJK 這幾個候選路徑。
import os as _os
_ASSETS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets")

FONT_CANDIDATES_REGULAR = [
    _os.path.join(_ASSETS_DIR, "NotoSansTC-Regular.ttf"),        # ⭐ 專案內建字型（優先，跨環境最穩定）
    _os.path.join(_ASSETS_DIR, "NotoSansTC-Regular.otf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Linux (Debian/Ubuntu)
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",                    # Linux (RHEL/Fedora)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Linux: fonts-noto-cjk
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",   # Linux: fonts-noto-cjk (變體路徑)
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",        # Linux: 部分發行版路徑
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",             # Linux: fonts-wqy-zenhei
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",               # Linux: 部分發行版路徑
    "/Library/Fonts/Arial.ttf",                                  # macOS
    "/Library/Fonts/Arial Unicode.ttf",                          # macOS（含中文）
    "/System/Library/Fonts/Supplemental/Arial.ttf",              # macOS
    "/System/Library/Fonts/PingFang.ttc",                        # macOS 中文
    "C:\\Windows\\Fonts\\arial.ttf",                              # Windows
    "C:\\Windows\\Fonts\\msjh.ttc",                               # Windows 中文
]
FONT_CANDIDATES_BOLD = [
    _os.path.join(_ASSETS_DIR, "NotoSansTC-Bold.ttf"),           # ⭐ 專案內建粗體字型（優先）
    _os.path.join(_ASSETS_DIR, "NotoSansTC-Bold.otf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",              # 文泉驛正黑無獨立 Bold，重複使用
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "C:\\Windows\\Fonts\\msjhbd.ttc",
]

# ==========================================================
# 🏷️ boarcast 專屬品牌浮水印設定
# ==========================================================
BRAND_NAME = "boarcast"
BRAND_TAGLINE = "Powered by boarcast | Data: MLB Statcast"
BRAND_COLOR_NEON = "#00E676"    # 螢光綠，主品牌識別色
BRAND_COLOR_BRIGHT = "#F5F5F5"  # 質感亮白，作為輔助/陰影對比色

# [Enh 1] 各球隊主場球場「向量外框」概略距離資料（單位：呎，Feet）。
# 資料來自各球場公開揭示之全壘打牆距離（左外野線 LF／左中 LCF／中外野 CF／
# 右中 RCF／右外野線 RF），僅為近似值（部分球場因牆面非對稱或多段折角，
# 這裡以 5 個錨點 + 分段線性內插近似其外形，非精確逐公尺測繪）。
# key 使用 Statcast home_team 縮寫。找不到對應球隊時使用 DEFAULT 通用值。
STADIUM_DIMENSIONS = {
    "LAD": {"LF": 330, "LCF": 375, "CF": 395, "RCF": 375, "RF": 330},   # Dodger Stadium
    "NYY": {"LF": 318, "LCF": 399, "CF": 408, "RCF": 385, "RF": 314},   # Yankee Stadium
    "BOS": {"LF": 310, "LCF": 379, "CF": 390, "RCF": 380, "RF": 302},   # Fenway Park（左外野綠色怪物）
    "CHC": {"LF": 355, "LCF": 368, "CF": 400, "RCF": 368, "RF": 353},   # Wrigley Field
    "SF": {"LF": 339, "LCF": 364, "CF": 399, "RCF": 415, "RF": 309},    # Oracle Park（右外野特短）
    "HOU": {"LF": 315, "LCF": 362, "CF": 409, "RCF": 373, "RF": 326},   # Minute Maid Park
    "ATL": {"LF": 335, "LCF": 375, "CF": 400, "RCF": 375, "RF": 325},   # Truist Park
    "PHI": {"LF": 329, "LCF": 374, "CF": 401, "RCF": 369, "RF": 330},   # Citizens Bank Park
    "STL": {"LF": 336, "LCF": 375, "CF": 400, "RCF": 375, "RF": 335},   # Busch Stadium
    "SD": {"LF": 336, "LCF": 390, "CF": 396, "RCF": 391, "RF": 322},    # Petco Park
    "TOR": {"LF": 328, "LCF": 375, "CF": 400, "RCF": 375, "RF": 328},   # Rogers Centre
    "TEX": {"LF": 329, "LCF": 372, "CF": 407, "RCF": 374, "RF": 326},   # Globe Life Field
    "SEA": {"LF": 331, "LCF": 378, "CF": 401, "RCF": 381, "RF": 326},   # T-Mobile Park
    "NYM": {"LF": 335, "LCF": 379, "CF": 408, "RCF": 380, "RF": 330},   # Citi Field
    "BAL": {"LF": 333, "LCF": 364, "CF": 400, "RCF": 373, "RF": 318},   # Camden Yards
    "MIL": {"LF": 344, "LCF": 371, "CF": 400, "RCF": 374, "RF": 345},   # American Family Field
    "MIN": {"LF": 339, "LCF": 377, "CF": 404, "RCF": 367, "RF": 328},   # Target Field
    "CLE": {"LF": 325, "LCF": 370, "CF": 405, "RCF": 375, "RF": 325},   # Progressive Field
    "DET": {"LF": 345, "LCF": 370, "CF": 412, "RCF": 365, "RF": 330},   # Comerica Park
    "KC": {"LF": 330, "LCF": 375, "CF": 410, "RCF": 375, "RF": 330},    # Kauffman Stadium
    "CWS": {"LF": 330, "LCF": 375, "CF": 400, "RCF": 375, "RF": 335},   # Guaranteed Rate Field
    "OAK": {"LF": 330, "LCF": 367, "CF": 400, "RCF": 367, "RF": 330},   # Oakland Coliseum
    "LAA": {"LF": 330, "LCF": 387, "CF": 396, "RCF": 370, "RF": 330},   # Angel Stadium
    "AZ": {"LF": 330, "LCF": 376, "CF": 407, "RCF": 376, "RF": 334},    # Chase Field
    "COL": {"LF": 347, "LCF": 390, "CF": 415, "RCF": 375, "RF": 350},   # Coors Field
    "MIA": {"LF": 344, "LCF": 386, "CF": 400, "RCF": 392, "RF": 335},   # loanDepot park
    "TB": {"LF": 315, "LCF": 370, "CF": 404, "RCF": 370, "RF": 322},    # Tropicana Field
    "WSH": {"LF": 336, "LCF": 377, "CF": 402, "RCF": 370, "RF": 335},   # Nationals Park
    "CIN": {"LF": 328, "LCF": 379, "CF": 404, "RCF": 371, "RF": 325},   # Great American Ball Park
    "PIT": {"LF": 325, "LCF": 383, "CF": 399, "RCF": 375, "RF": 320},   # PNC Park
    "DEFAULT": {"LF": 330, "LCF": 375, "CF": 400, "RCF": 375, "RF": 330},
}

# ==========================================================
# 🕶️ 賽博戰情室視覺主題 (Cyberpunk Cyber-Ops Theme)
# ==========================================================
# 命名對應設計計劃書中的中文顏色語彙，方便日後對照調整。
CYBER_BG_BASE = "#05070C"          # 宇宙深邃黑（Base Layer）
CYBER_BG_PANEL = "#10151F"         # 碳晶深灰藍（Layer 2 - Tactical Panels）
CYBER_BG_PANEL_ALT = "#141B28"     # 面板次要底色（表格列交錯等場合使用）
CYBER_BORDER_IDLE = "rgba(56, 189, 248, 0.18)"   # 暗流微光藍線（未互動時）
CYBER_BORDER_HOVER = "rgba(56, 189, 248, 0.85)"  # 滑鼠滑過時的亮起邊框
CYBER_NEON_BLUE = "#00E5FF"        # 電子發光霓虹藍（Primary Glow）
CYBER_NEON_PINK = "#FF2E9A"        # 極光霓虹粉紅（Alert Mode）
CYBER_MUTED_STEEL = "#7C8AA5"      # 鋼鐵灰藍色（Muted Specs）
CYBER_TEXT_PRIMARY = "#E7F6FF"     # 主要內文字色（冷白，避免死白刺眼）
CYBER_FONT_STACK = (
    "'JetBrains Mono', 'Fira Code', 'SFMono-Regular', 'IBM Plex Mono', "
    "Menlo, Consolas, 'Courier New', monospace"
)

# [Alert Mode 門檻] 觸發「極光霓虹粉紅」警示的極端數據標準：
#   - 出色擊球初速：Statcast 官方 Barrel 判定約在 98mph 起算，這裡取更嚴格的
#     115mph（大谷翔平等級的巨砲等級）作為「破紀錄感」的視覺警示門檻。
#   - 面對火球：對手投出 100mph 以上快速球，屬於聯盟頂級火球等級。
EXIT_VELO_ALERT_MPH = 115.0
PITCH_VELO_ALERT_MPH = 100.0


def inject_cyberpunk_theme() -> None:
    """注入全站「終極賽博黑客風」CSS 主題。

    設計原則：不靠圖片堆砌，純粹用背景分層、邊框微光、霓虹發光數據與等寬
    字體排版，打造戰情室 / 鋼鐵人控制艙式的硬核介面。針對 Streamlit 內建元件
    (data-testid 選擇器較不受版本更新影響，優先使用) 進行整體覆寫。
    """
    st.markdown(
        f"""
        <style>
        /* ---- 全站底色：宇宙深邃黑 + 極低對比度星空漸層 ---- */
        .stApp {{
            background:
                radial-gradient(ellipse at 15% 0%, rgba(0, 229, 255, 0.05), transparent 45%),
                radial-gradient(ellipse at 85% 100%, rgba(255, 46, 154, 0.04), transparent 40%),
                {CYBER_BG_BASE};
            color: {CYBER_TEXT_PRIMARY};
        }}

        /* ---- 全站字體：強制鎖定等寬碼農體 ---- */
        html, body, .stApp, [class*="css"], p, span, div, label,
        input, textarea, select, button {{
            font-family: {CYBER_FONT_STACK} !important;
        }}

        /* ---- 標題：霓虹藍發光 ---- */
        h1, h2, h3, h4, .stApp h1, .stApp h2, .stApp h3 {{
            color: {CYBER_NEON_BLUE} !important;
            text-shadow: 0 0 6px rgba(0, 229, 255, 0.55), 0 0 18px rgba(0, 229, 255, 0.25);
            letter-spacing: 0.03em;
        }}

        /* ---- 側邊欄：碳晶深灰藍戰術面板 + 右側微光邊框 ---- */
        section[data-testid="stSidebar"] {{
            background: {CYBER_BG_PANEL};
            border-right: 1px solid {CYBER_BORDER_IDLE};
        }}
        section[data-testid="stSidebar"] * {{
            color: {CYBER_TEXT_PRIMARY};
        }}

        /* ---- 分頁 Tabs：面板化 + 選中頁籤發光底線 ---- */
        button[data-baseweb="tab"] {{
            color: {CYBER_MUTED_STEEL} !important;
            font-family: {CYBER_FONT_STACK} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {CYBER_NEON_BLUE} !important;
            border-bottom: 2px solid {CYBER_NEON_BLUE} !important;
            text-shadow: 0 0 8px rgba(0, 229, 255, 0.6);
        }}

        /* ---- 數據方塊 (st.metric)：面板化 + 邊框，數值霓虹發光 ---- */
        div[data-testid="stMetric"] {{
            background: {CYBER_BG_PANEL};
            border: 1px solid {CYBER_BORDER_IDLE};
            border-radius: 6px;
            padding: 0.85rem 0.9rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            border-color: {CYBER_BORDER_HOVER};
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.25);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {CYBER_MUTED_STEEL} !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        div[data-testid="stMetricValue"] {{
            color: {CYBER_NEON_BLUE} !important;
            text-shadow: 0 0 6px rgba(0, 229, 255, 0.5);
            font-variant-numeric: tabular-nums;
        }}

        /* ---- 我方自訂的 cyber-tile（用於極端數據 Alert Mode 高亮）---- */
        .cyber-tile {{
            background: {CYBER_BG_PANEL};
            border: 1px solid {CYBER_BORDER_IDLE};
            border-radius: 6px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.4rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        .cyber-tile:hover {{
            border-color: {CYBER_BORDER_HOVER};
            box-shadow: 0 0 14px rgba(56, 189, 248, 0.25);
        }}
        .cyber-tile-label {{
            color: {CYBER_MUTED_STEEL};
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }}
        .cyber-tile-value {{
            font-size: 1.75rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }}
        .cyber-tile.neon-primary .cyber-tile-value {{
            color: {CYBER_NEON_BLUE};
            text-shadow: 0 0 6px rgba(0, 229, 255, 0.5);
        }}
        .cyber-tile.neon-alert {{
            border-color: rgba(255, 46, 154, 0.55);
            box-shadow: 0 0 16px rgba(255, 46, 154, 0.28);
            animation: cyber-alert-pulse 1.8s ease-in-out infinite;
        }}
        .cyber-tile.neon-alert .cyber-tile-value {{
            color: {CYBER_NEON_PINK};
            text-shadow: 0 0 8px rgba(255, 46, 154, 0.75), 0 0 20px rgba(255, 46, 154, 0.35);
        }}
        .cyber-tile.neon-alert .cyber-tile-label::before {{
            content: "⚡ ";
        }}
        @keyframes cyber-alert-pulse {{
            0%, 100% {{ box-shadow: 0 0 12px rgba(255, 46, 154, 0.25); }}
            50% {{ box-shadow: 0 0 22px rgba(255, 46, 154, 0.55); }}
        }}

        /* ---- 按鈕：暗面板 + 藍線邊框，hover 時亮起 ---- */
        .stButton > button {{
            background: {CYBER_BG_PANEL};
            color: {CYBER_NEON_BLUE};
            border: 1px solid {CYBER_BORDER_IDLE};
            font-family: {CYBER_FONT_STACK} !important;
            letter-spacing: 0.04em;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            border-color: {CYBER_NEON_BLUE};
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.45);
            color: {CYBER_NEON_BLUE};
        }}

        /* ---- 輸入框 / 下拉選單 / 滑桿：統一面板化 ---- */
        div[data-baseweb="input"], div[data-baseweb="select"] > div {{
            background: {CYBER_BG_PANEL} !important;
            border-color: {CYBER_BORDER_IDLE} !important;
            color: {CYBER_TEXT_PRIMARY} !important;
        }}

        /* ---- [UI Fix] 側邊欄搜尋欄文字隱形修正 ----
           側邊欄輸入文字與下拉選單選中文字，強制鎖定亮白色，並鎖定深色背景，
           確保與白字對比度足夠；-webkit-text-fill-color 額外防止部分行動裝置
           瀏覽器（尤其是自動填入/深色模式）把文字顏色蓋回黑色導致「打字時看不見」。 */
        section[data-testid="stSidebar"] input[data-testid="stTextInput"],
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] div[data-baseweb="input"] input,
        section[data-testid="stSidebar"] div[data-baseweb="select"] div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] input {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background-color: {CYBER_BG_PANEL} !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="input"],
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: {CYBER_BG_PANEL} !important;
            border-color: {CYBER_BORDER_IDLE} !important;
        }}
        /* 下拉選單展開後的彈出選單（Popover）常會被 render 到 sidebar 容器之外，
           需另外用 baseweb 的 popover/menu 選擇器鎖定背景與文字顏色。 */
        div[data-baseweb="popover"] ul[role="listbox"],
        div[data-baseweb="menu"] {{
            background-color: #1e293b !important;
        }}
        div[data-baseweb="popover"] ul[role="listbox"] li,
        div[data-baseweb="menu"] li {{
            color: #ffffff !important;
        }}
        div[data-testid="stSlider"] [role="slider"] {{
            background-color: {CYBER_NEON_BLUE} !important;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.6);
        }}
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
            background: {CYBER_NEON_BLUE} !important;
        }}

        /* ---- 提示訊息：暗底 + 左側色條，取代預設的亮色背景 ---- */
        div[data-testid="stAlertContainer"] {{
            background: {CYBER_BG_PANEL} !important;
            color: {CYBER_TEXT_PRIMARY} !important;
            border-left: 3px solid {CYBER_NEON_BLUE};
        }}

        /* ---- DataFrame / 表格：面板化底色 ---- */
        div[data-testid="stDataFrame"] {{
            background: {CYBER_BG_PANEL};
            border: 1px solid {CYBER_BORDER_IDLE};
            border-radius: 6px;
        }}

        /* ---- Plotly 圖表容器：加上微光邊框，呼應戰術面板感 ---- */
        div[data-testid="stPlotlyChart"] {{
            background: {CYBER_BG_PANEL};
            border: 1px solid {CYBER_BORDER_IDLE};
            border-radius: 6px;
            padding: 0.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cyber_stat_tile(container, label: str, value_text: str, is_alert: bool = False) -> None:
    """繪製自訂的「賽博戰術面板」數據方塊，供極端數據 Alert Mode 高亮使用。

    is_alert=True 時套用極光霓虹粉紅樣式（含呼吸燈動畫），一般情況套用
    電子發光霓虹藍樣式，與 st.metric 的整體視覺主題保持一致。
    """
    glow_class = "neon-alert" if is_alert else "neon-primary"
    container.markdown(
        f"""
        <div class="cyber-tile {glow_class}">
            <div class="cyber-tile-label">{label}</div>
            <div class="cyber-tile-value">{value_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# 1. 數據獲取與預處理模組 (Data Fetching)
# ==========================================================


def _strip_accents(text) -> str:
    """[優化 3] 西班牙文/重音符號忽略機制：

    將字串以 Unicode NFD（Normalization Form Decomposed）正規化，把「基底字母」
    與「重音/變音符號」拆成兩個獨立的 Unicode 字元（例如 'ñ' 拆成 'n' + 組合用
    的顎化符號），接著濾除所有 unicodedata 分類為 'Mn'（Mark, Nonspacing，也就是
    附加在基底字母上的變音符號）的字元，只留下基底字母本身。

    最後統一轉為小寫並去除頭尾空白，回傳一個「純 ASCII、忽略重音、忽略大小寫」
    的正規化字串，供雙向比對使用：不論使用者輸入的是 'Acuna' 還是 'Acuña'、
    'Jose' 還是 'José'，正規化後都會得到相同的 'acuna' / 'jose'，即可互相比對成功。
    """
    if text is None:
        return ""
    nfd = unicodedata.normalize("NFD", str(text))
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return stripped.strip().lower()


@st.cache_data(show_spinner=False)
def get_player_data(first_name: str, last_name: str, year: int):
    """搜尋球員並抓取 Statcast 逐球資料，回傳打席 (PA) 等級資料表"""
    player_prof = playerid_lookup(last_name, first_name)

    # [優化 3] 西班牙文與特殊重音符號忽略機制：
    #   pybaseball 的 playerid_lookup 本身是「精確字串比對」，若使用者輸入的是
    #   不含重音的一般英文拼法（例如 Acuna、Jose、Nunez），而資料庫（Chadwick
    #   Register）中登記的官方姓名帶有重音符號（Acuña、José、Núñez），會直接
    #   查無結果並回傳空的 DataFrame。
    #
    #   這裡在原始查詢查無結果時，改抓取完整的 Chadwick 球員名冊（chadwick_register），
    #   將「使用者輸入」與「名冊中每一位球員的姓名」都透過 _strip_accents() 雙向去
    #   除重音符號 + 轉小寫後再比對，如此一來無論使用者輸入 Acuna 或 Acuña，
    #   都能命中同一位球員，大幅提升拉丁美洲球員姓名的搜尋容錯力。
    if player_prof.empty:
        try:
            full_register = pybaseball.chadwick_register()
        except Exception:
            full_register = None

        if full_register is not None and not full_register.empty:
            norm_last_input = _strip_accents(last_name)
            norm_first_input = _strip_accents(first_name)

            reg = full_register.dropna(subset=["name_last", "name_first", "key_mlbam"]).copy()
            reg_last_norm = reg["name_last"].map(_strip_accents)
            reg_first_norm = reg["name_first"].map(_strip_accents)

            match_mask = (reg_last_norm == norm_last_input) & (reg_first_norm == norm_first_input)
            candidates = reg[match_mask]

            if not candidates.empty:
                player_prof = candidates.reset_index(drop=True)

    if player_prof.empty:
        return None, None, None

    # [Fix 5] 姓名 ID 查詢優化：同名同姓查到多筆結果時，依 mlb_played_last
    # （該球員最後一次出賽的年份）由新到舊排序，取「最新現役／最近仍在打球」
    # 的那一位野手，而不是預設回傳的第一筆（可能是幾十年前的退役球員同名同姓）。
    if len(player_prof) > 1 and "mlb_played_last" in player_prof.columns:
        player_prof = player_prof.sort_values(
            "mlb_played_last", ascending=False, na_position="last"
        ).reset_index(drop=True)

    player_id = int(player_prof["key_mlbam"].values[0])
    # [Fix 6] 提前起始日期，避免漏抓海外開幕賽（如東京 Series）
    start_date = f"{year}-{SEASON_START_MONTH_DAY}"
    end_date = f"{year}-{SEASON_END_MONTH_DAY}"

    raw = statcast_batter(start_date, end_date, player_id)
    if raw is None or raw.empty:
        return None, player_id, None

    df = raw.copy()

    # [Fix 4] 嚴格過濾例行賽：Statcast 回傳的資料範圍涵蓋春訓 (S) 與季後賽
    # (F/D/L/W 等)，若不過濾，總場次很容易「爆量」超過 162 場（例如常見的
    # 185+ 場）。這裡立刻只保留 game_type == 'R'（Regular Season）的資料列，
    # 徹底排除春訓與季後賽場次。
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()
    if df.empty:
        return None, player_id, None

    # [Fix 4] 以 (game_date, game_pk) 作為單場聚合依據：先依日期、同日內再依
    # game_pk 排序，避免雙重賽（同一天兩場比賽、game_pk 不同）被誤判為同一場，
    # 也避免僅依 game_date 排序時，同一天兩場比賽的先後順序不穩定。
    # 排序穩定後，重新生成 1 到 162 的標準例行賽場次編號 (game_num) 作為 X 軸，
    # 徹底解決先前場次編號離譜暴增的問題。
    game_dates = (
        df[["game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk"])
        .reset_index(drop=True)
    )
    game_dates["game_num"] = range(1, len(game_dates) + 1)
    df = df.merge(game_dates[["game_date", "game_pk", "game_num"]], on=["game_date", "game_pk"], how="left")

    # ---- [Fix 2] 先從「完整逐球資料 df」統計盜壘相關事件，再篩選打席結果 ----
    # Statcast 的盜壘/盜壘失敗事件值通常是 stolen_base_2b / stolen_base_3b /
    # stolen_base_home、caught_stealing_2b / caught_stealing_3b / caught_stealing_home
    # 等等，而非單純的 "stolen_base" / "caught_stealing"。這些事件本身就有非空的
    # events 欄位（它們本身就是一個獨立的「事件」row），所以不會被下面的 PA 篩選排除，
    # 但先前用「==」精確比對字串會完全比對不到，導致 SB/CS 恆為 0。
    # 這裡改用 str.contains 進行子字串比對，並把結果另外記錄下來，
    # 避免依賴後續以 events.notna() 篩出的 pa_df（該篩選本身邏輯不變，但比對條件修正）。
    events_str = df["events"].astype(str)
    df["is_stolen_base"] = events_str.str.contains("stolen_base", na=False)
    df["is_caught_stealing"] = events_str.str.contains("caught_stealing", na=False)

    # 只保留每個打席「結果」那一球（events 非空），避免逐球重複計算
    pa_df = df[df["events"].notna()].copy()

    # 標記安打 / 打數（AB）：打數不含保送、觸身球、犧牲觸擊/高飛
    non_ab_events = [
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "sac_fly_double_play", "catcher_interf",
    ]
    pa_df["is_ab"] = ~pa_df["events"].isin(non_ab_events)
    pa_df["is_hit"] = pa_df["events"].isin(HIT_EVENTS)

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    pa_df["total_bases"] = pa_df["events"].map(tb_map).fillna(0)

    total_games = int(game_dates["game_num"].max())
    return pa_df, player_id, total_games


def determine_player_team(pa_df: pd.DataFrame) -> str:
    """[優化 4] 動態球隊判定機制：

    直接讀取（或用 mode 取眾數）Statcast 資料中的 home_team 欄位來當作「球員
    當前球隊」，只在球員該賽季『每一場都打主場』時才會剛好正確；只要球員曾經
    打過一場客場比賽，該場逐球資料的 home_team 記錄的其實是『對手主場球隊』，
    而不是球員自己的球隊，就可能被誤判（例如大谷翔平作客到小熊主場比賽時，
    home_team 欄位值會是 'CHC'，若直接取用就會誤顯示「大谷翔平：小熊隊」）。

    正確做法：棒球規則裡，每局「上半 (Top)」固定由客隊 (away_team) 進攻、
    主隊守備；「下半 (Bot)」則相反，由主隊 (home_team) 進攻、客隊守備。
    因此只要讀取 Statcast 逐球資料中的 inning_topbot 欄位，就能 100% 精準反推
    「打者當天究竟代表哪一隊上場打擊」：
        inning_topbot 為 'Top' → 當時是客隊在打擊 → 球員屬於 away_team
        inning_topbot 為 'Bot' → 當時是主隊在打擊 → 球員屬於 home_team

    逐球套用這條規則後，再以最新的 game_date 出賽紀錄為準（球員季中轉隊時，
    永遠顯示離現在最近一場比賽推算出的球隊），即可不受主客場影響，穩定且
    正確地顯示球員目前代表的球隊。
    """
    required_cols = {"home_team", "away_team", "inning_topbot", "game_date"}
    if not required_cols.issubset(pa_df.columns):
        # 資料不完整（例如舊快取或欄位缺漏）時，安全退回舊邏輯，避免直接爆錯
        if "home_team" in pa_df.columns and not pa_df["home_team"].dropna().empty:
            return str(pa_df["home_team"].iloc[-1])
        return "N/A"

    df = pa_df.dropna(subset=["home_team", "away_team", "inning_topbot", "game_date"])
    if df.empty:
        return "N/A"

    is_top = df["inning_topbot"].astype(str).str.upper().str.startswith("TOP")
    player_team = np.where(is_top, df["away_team"], df["home_team"])

    # 以 game_date 由舊到新排序，取「最新一場出賽紀錄」推算出的球隊為準，
    # 讓球員季中轉隊也能正確反映最新代表的球隊。
    order = np.argsort(df["game_date"].values)
    latest_team = player_team[order][-1]
    return str(latest_team)


def fetch_player_headshot_url(player_id: int) -> str:
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/"
        f"people/{player_id}/headshot/67/current"
    )


# ==========================================================
# 2. 162 場放大模擬核心演算法 (Simulation Engine)
# ==========================================================


def calculate_stats(pa_df: pd.DataFrame, scale_to_162: bool, games_in_sample: int) -> dict:
    """計算選定區間打擊數據，並可等比例外推至 162 場"""
    if pa_df.empty:
        return {k: 0 for k in [
            "AVG", "OBP", "SLG", "OPS", "ISO",
            "1B", "2B", "3B", "HR", "H", "AB", "HBP", "SF",
            "SB", "CS", "BB", "SO", "RBI",
            "Avg_EV", "Max_EV", "Avg_LA", "Max_Pitch_Velo_Faced",
            "Max_HR_Dist", "Avg_HR_Dist",
            "xBA", "xSLG", "Barrel_Pct", "HardHit_Pct",
            "GB_Pct", "LD_Pct", "FB_Pct", "PU_Pct", "BattedBall_N",
            "Extrap_Capped", "Extrap_Factor", "Extrap_LowSample",
        ]}

    ab = int(pa_df["is_ab"].sum())
    hits = int(pa_df["is_hit"].sum())
    singles = int((pa_df["events"] == "single").sum())
    doubles = int((pa_df["events"] == "double").sum())
    triples = int((pa_df["events"] == "triple").sum())
    hrs = int((pa_df["events"] == "home_run").sum())
    bbs = int((pa_df["events"] == "walk").sum())
    sos = int((pa_df["events"] == "strikeout").sum())

    # [OBP Fix] 上壘率分母修正：OBP = (H+BB+HBP) / (AB+BB+HBP+SF)，
    # 過去版本分母誤用「AB+BB」（漏算 HBP、SF），會系統性低估 OBP（進而拖累 OPS）。
    # HBP／SF 欄位可能因來源資料版本不同而缺值，統一 fillna(0) 防呆，避免分母出現 NaN。
    hbps = int(pd.to_numeric((pa_df["events"] == "hit_by_pitch"), errors="coerce").fillna(0).sum())
    sfs = int(pd.to_numeric(
        pa_df["events"].isin(["sac_fly", "sac_fly_double_play"]), errors="coerce"
    ).fillna(0).sum())

    # [Fix 2] 改用預先算好的 is_stolen_base / is_caught_stealing（子字串比對），
    # 若欄位不存在（例如舊快取資料）則安全退回 0，不再用會恆為 0 的舊比對方式
    sbs = int(pa_df["is_stolen_base"].sum()) if "is_stolen_base" in pa_df.columns else 0
    css = int(pa_df["is_caught_stealing"].sum()) if "is_caught_stealing" in pa_df.columns else 0

    rbi = int(pa_df["rbi"].sum()) if "rbi" in pa_df.columns else 0

    total_bases = singles + doubles * 2 + triples * 3 + hrs * 4

    avg_ev = float(pa_df["launch_speed"].mean()) if "launch_speed" in pa_df.columns else np.nan
    max_ev = float(pa_df["launch_speed"].max()) if "launch_speed" in pa_df.columns else np.nan
    avg_la = float(pa_df["launch_angle"].mean()) if "launch_angle" in pa_df.columns else np.nan

    # [Cyber-Ops Rev] 本區間內面對過的最快球速（release_speed 為投手釋放球速），
    # 供「極端數據高亮 Alert Mode」判斷是否觸發火球警示（例如面對 100mph+ 快速球）。
    max_pitch_velo_faced = float(pa_df["release_speed"].max()) if "release_speed" in pa_df.columns else np.nan

    hr_df = pa_df[pa_df["events"] == "home_run"]
    if "hit_distance_sc" in pa_df.columns and not hr_df.empty:
        max_hr_dist = float(hr_df["hit_distance_sc"].max())
        avg_hr_dist = float(hr_df["hit_distance_sc"].mean())
    else:
        max_hr_dist, avg_hr_dist = np.nan, np.nan

    # -------- [Enh 2] 進階期望數據：xBA / xSLG / Barrel% / Hard Hit% --------
    # xBA / xSLG：直接取 Statcast 官方逐打席估計值（estimated_ba_using_speedangle /
    # estimated_slg_using_speedangle）之平均。這兩欄僅在「有擊球初速/仰角」的打席
    # （球有進場）才有值，三振/保送/觸身球等打席無此欄位，取平均時會自動被排除，
    # 這與官方 xBA/xSLG 僅計入有效擊球樣本的定義一致。
    # 注意：此為描述性「期望值」比率，不隨 162 場外推而改變，因此在外推區塊之前計算，
    # 且外推 (scale_to_162) 不會、也不應該調整這幾個比率欄位。
    if "estimated_ba_using_speedangle" in pa_df.columns:
        xba = pa_df["estimated_ba_using_speedangle"].mean()
        xba = float(xba) if pd.notna(xba) else np.nan
    else:
        xba = np.nan

    if "estimated_slg_using_speedangle" in pa_df.columns:
        xslg = pa_df["estimated_slg_using_speedangle"].mean()
        xslg = float(xslg) if pd.notna(xslg) else np.nan
    else:
        xslg = np.nan

    if "launch_speed" in pa_df.columns:
        ev_series = pa_df["launch_speed"]
        la_series = pa_df["launch_angle"] if "launch_angle" in pa_df.columns else pd.Series(np.nan, index=pa_df.index)
        batted_ball_mask = ev_series.notna()
        n_batted = int(batted_ball_mask.sum())
        if n_batted > 0:
            hard_hit_pct = float((ev_series[batted_ball_mask] >= 95).sum()) / n_batted

            # Barrel 近似判定：初速需 >= 98 mph；仰角「合格區間」隨初速增加而擴大，
            # 98mph 時約 26°~30°，116mph（以上）時擴大到 8°~50°，中間以線性內插近似。
            # 這是簡化版本，並非 Statcast 官方逐 mph 對照表，僅供儀表板參考使用。
            def _is_barrel(ev, la):
                if pd.isna(ev) or pd.isna(la) or ev < 98:
                    return False
                ev_clamped = min(ev, 116)
                t = (ev_clamped - 98) / (116 - 98)  # 0~1
                lower = 26 - t * (26 - 8)
                upper = 30 + t * (50 - 30)
                return lower <= la <= upper

            barrel_count = sum(
                _is_barrel(ev, la) for ev, la in zip(ev_series[batted_ball_mask], la_series[batted_ball_mask])
            )
            barrel_pct = barrel_count / n_batted
        else:
            hard_hit_pct, barrel_pct = np.nan, np.nan
    else:
        hard_hit_pct, barrel_pct = np.nan, np.nan

    # -------- [新增] 擊球型態分佈 (Batted Ball Type Distribution) --------
    # 擷取 Statcast 的 bb_type 欄位：ground_ball（滾地球）/ line_drive（平飛球）/
    # fly_ball（飛球）/ popup（內野高飛球）。僅在球被打進場內時才有值，
    # 三振、保送、觸身球等無擊球事件不計入分母，符合一般擊球型態率的定義。
    if "bb_type" in pa_df.columns:
        bb_series = pa_df["bb_type"].dropna()
        n_batted_ball = int(len(bb_series))
        if n_batted_ball > 0:
            gb_pct = float((bb_series == "ground_ball").sum()) / n_batted_ball
            ld_pct = float((bb_series == "line_drive").sum()) / n_batted_ball
            fb_pct = float((bb_series == "fly_ball").sum()) / n_batted_ball
            pu_pct = float((bb_series == "popup").sum()) / n_batted_ball
        else:
            gb_pct = ld_pct = fb_pct = pu_pct = np.nan
    else:
        n_batted_ball = 0
        gb_pct = ld_pct = fb_pct = pu_pct = np.nan

    # -------- 外推至 162 場全賽季 (Scale to 162 Games) --------
    # [優化 3] 外推極限防呆：
    #   games_in_sample > 0 只保證不會除以零，但若使用者選到極端小樣本區間
    #   （例如快篩鎖定的區間剛好只涵蓋 1~4 場比賽），162/games_in_sample 可能
    #   高達 40~162 倍，會把「1 次三振」誇張放大成「上百次三振」等滑稽的極端預測值。
    #   這裡將外推倍率上限鎖定在 MAX_EXTRAPOLATION_FACTOR（預設 10 倍），
    #   並標記 Extrap_LowSample（樣本 < 5 場）供 UI 顯示警示文字，讓使用者清楚
    #   知道這只是「參考用途」的預測值，而非嚴謹的全賽季推估。
    MAX_EXTRAPOLATION_FACTOR = 10.0
    extrap_capped = False
    extrap_low_sample = games_in_sample < 5
    factor = 1.0
    if scale_to_162 and games_in_sample > 0:
        raw_factor = 162.0 / games_in_sample
        factor = min(raw_factor, MAX_EXTRAPOLATION_FACTOR)
        extrap_capped = raw_factor > MAX_EXTRAPOLATION_FACTOR
        ab = int(round(ab * factor))
        singles = int(round(singles * factor))
        doubles = int(round(doubles * factor))
        triples = int(round(triples * factor))
        hrs = int(round(hrs * factor))
        hits = singles + doubles + triples + hrs
        sbs = int(round(sbs * factor))
        css = int(round(css * factor))
        bbs = int(round(bbs * factor))
        sos = int(round(sos * factor))
        rbi = int(round(rbi * factor))
        hbps = int(round(hbps * factor))
        sfs = int(round(sfs * factor))
        total_bases = singles + doubles * 2 + triples * 3 + hrs * 4

    # [Fix 3] 率數據一律以「最終（外推後或原始）整數統計」重新計算，
    # 確保 AVG/OBP/SLG/OPS/ISO 與同時顯示的 H/AB/HR 等整數數據完全對齊，
    # 不再出現外推前算好的舊率數據與外推後整數兜不起來的誤差。
    # [OBP Fix] 先加總、後相除：OBP = (H+BB+HBP) / (AB+BB+HBP+SF)，與 MLB 官方公式一致。
    avg = hits / ab if ab > 0 else 0.0
    obp_denom = ab + bbs + hbps + sfs
    obp = (hits + bbs + hbps) / obp_denom if obp_denom > 0 else 0.0
    slg = total_bases / ab if ab > 0 else 0.0
    # [OPS Fix] 強制以修正後的 OBP + SLG 相加，避免與原始資料庫 OPS 欄位脫鉤。
    ops = obp + slg
    iso = slg - avg

    return {
        "AVG": avg, "OBP": obp, "SLG": slg, "OPS": ops, "ISO": iso,
        "1B": singles, "2B": doubles, "3B": triples, "HR": hrs,
        "H": hits, "AB": ab, "HBP": hbps, "SF": sfs,
        "SB": sbs, "CS": css, "BB": bbs, "SO": sos, "RBI": rbi,
        "Avg_EV": avg_ev, "Max_EV": max_ev, "Avg_LA": avg_la,
        "Max_Pitch_Velo_Faced": max_pitch_velo_faced,
        "Max_HR_Dist": max_hr_dist, "Avg_HR_Dist": avg_hr_dist,
        "xBA": xba, "xSLG": xslg, "Barrel_Pct": barrel_pct, "HardHit_Pct": hard_hit_pct,
        "GB_Pct": gb_pct, "LD_Pct": ld_pct, "FB_Pct": fb_pct, "PU_Pct": pu_pct,
        "BattedBall_N": n_batted_ball,
        "Extrap_Capped": extrap_capped, "Extrap_Factor": factor, "Extrap_LowSample": extrap_low_sample,
    }


# ==========================================================
# 3. 8 種模式 3x3 好球帶熱區 (Interactive Hot Zone)
# ==========================================================


def render_hot_zone(pa_df: pd.DataFrame, metric_choice: str) -> go.Figure:
    """繪製 3x3 九宮格熱區，8 種模式動態切換，數字覆蓋在格子中央"""
    grid_vals = np.full((3, 3), np.nan)
    grid_txts = np.empty((3, 3), dtype=object)
    grid_txts[:] = "N/A"

    has_zone = "zone" in pa_df.columns

    for z, (r, c) in ZONE_MAP.items():
        z_df = pa_df[pa_df["zone"] == z] if has_zone else pa_df.iloc[0:0]
        if len(z_df) == 0:
            # [Fix 4] 無資料格改留 np.nan，而非數值 0。
            # px.imshow 對 NaN 預設會渲染成透明/留白，不會被色階誤解讀為
            # 「數值 0」（例如打擊率 0.000）對應的深色格。
            grid_vals[r, c] = np.nan
            grid_txts[r, c] = "N/A"
            continue

        ab = int(z_df["is_ab"].sum())
        hits = int(z_df["is_hit"].sum())
        val, txt = 0.0, "N/A"

        if metric_choice == "打擊率 (AVG)":
            val = hits / ab if ab > 0 else 0.0
            txt = f"{val:.3f}"
        elif metric_choice == "長打率 (SLG)":
            val = z_df["total_bases"].sum() / ab if ab > 0 else 0.0
            txt = f"{val:.3f}"
        elif metric_choice == "純長打率 (ISO)":
            _avg = hits / ab if ab > 0 else 0.0
            _slg = z_df["total_bases"].sum() / ab if ab > 0 else 0.0
            val = _slg - _avg
            txt = f"{val:.3f}"
        elif metric_choice == "全壘打 (HR)":
            val = int((z_df["events"] == "home_run").sum())
            txt = f"{int(val)}"
        elif metric_choice == "三壘安打 (3B)":
            val = int((z_df["events"] == "triple").sum())
            txt = f"{int(val)}"
        elif metric_choice == "二壘安打 (2B)":
            val = int((z_df["events"] == "double").sum())
            txt = f"{int(val)}"
        elif metric_choice == "一壘安打 (1B)":
            val = int((z_df["events"] == "single").sum())
            txt = f"{int(val)}"
        elif metric_choice == "平均擊球初速 (Avg EV)":
            m = z_df["launch_speed"].mean() if "launch_speed" in z_df.columns else np.nan
            val = float(m) if not np.isnan(m) else np.nan
            txt = f"{val:.1f}" if not np.isnan(m) else "N/A"

        # 若該格有打席但特定指標仍算不出數值（如缺 launch_speed），同樣留 NaN 而非 0
        grid_vals[r, c] = val if (isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val))) else np.nan
        grid_txts[r, c] = txt

    fig = px.imshow(
        grid_vals,
        color_continuous_scale="RdBu_r",
        labels=dict(color=metric_choice),
        x=["內角 In", "中間 Mid", "外角 Out"],
        y=["高角 High", "中間 Mid", "低角 Low"],
        aspect="equal",
    )
    fig.update_traces(
        text=grid_txts,
        texttemplate="%{text}",
        textfont_size=20,
        textfont_color="black",
        hovertemplate="%{y} / %{x}<br>數值: %{z}<extra></extra>",
    )
    fig.update_layout(
        title=f"好球帶熱區分析 - {metric_choice}",
        width=440,
        height=440,
        coloraxis_showscale=True,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ==========================================================
# 4. 2D 棒球場擊球落點圖 (Spray Chart)
# ==========================================================


def render_spray_chart(pa_df: pd.DataFrame, home_team: str = None) -> go.Figure:
    """繪製 2D 球場擊球落點圖，含出局/安打分色，支援框選 (Box Select)。

    [Enh 1] 依 home_team 從 STADIUM_DIMENSIONS 查表帶入該球場左外野線/左中/
    中外野/右中/右外野線的實際距離，繪製貼近真實外形的球場向量外框，
    取代先前固定半徑 330 呎的通用弧線。查無對應球隊時退回 DEFAULT 通用值。
    """
    spray_df = pa_df.dropna(subset=["hc_x", "hc_y"]).copy()

    if spray_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="擊球落點分佈（此區間無座標資料）",
            width=500, height=500,
        )
        return fig

    # Statcast 座標系統轉換：原點移到本壘板，Y 軸反轉朝向外野
    spray_df["plot_x"] = spray_df["hc_x"] - 125.42
    spray_df["plot_y"] = 198.27 - spray_df["hc_y"]

    def classify(ev):
        if ev in HIT_EVENTS:
            return ev
        return "out / 其他"

    spray_df["result"] = spray_df["events"].apply(classify)

    color_map = {
        "single": "#4CAF50",
        "double": "#2196F3",
        "triple": "#FF9800",
        "home_run": "#E53935",
        "out / 其他": "#9E9E9E",
    }

    fig = px.scatter(
        spray_df,
        x="plot_x",
        y="plot_y",
        color="result",
        color_discrete_map=color_map,
        hover_data=["launch_speed", "launch_angle", "events", "game_num"],
        title="擊球落點分佈圖 (可拖曳滑鼠框選 Box Select 過濾)",
    )

    # [Enh 1] 疊加球隊專屬球場向量外框：以左外野線/左中/中外野/右中/右外野線
    # 5 個錨點（角度 135°/112.5°/90°/67.5°/45°）+ 分段線性內插畫出貼近真實外形
    # 的全壘打牆外框，取代先前所有球場共用的固定半徑 330 呎弧線。
    dims = STADIUM_DIMENSIONS.get(home_team, STADIUM_DIMENSIONS["DEFAULT"])
    anchor_deg = np.array([135, 112.5, 90, 67.5, 45])
    anchor_dist = np.array([dims["LF"], dims["LCF"], dims["CF"], dims["RCF"], dims["RF"]])
    theta_deg = np.linspace(45, 135, 100)
    wall_dist = np.interp(theta_deg, anchor_deg[::-1], anchor_dist[::-1])
    theta = np.radians(theta_deg)
    stadium_label = home_team if home_team else "通用"
    fig.add_trace(
        go.Scatter(
            x=wall_dist * np.cos(theta),
            y=wall_dist * np.sin(theta),
            mode="lines",
            line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dash"),
            name=f"全壘打牆（{stadium_label} 主場）",
            hoverinfo="skip",
        )
    )
    # 邊線
    fig.add_trace(go.Scatter(x=[0, 300], y=[0, 300], mode="lines",
                              line=dict(color="rgba(255,255,255,0.2)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, -300], y=[0, 300], mode="lines",
                              line=dict(color="rgba(255,255,255,0.2)"), showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        width=520, height=520,
        xaxis=dict(range=[-350, 350], showgrid=False, zeroline=False, title=None),
        yaxis=dict(range=[-20, 400], showgrid=False, zeroline=False, title=None),
        dragmode="select",
        plot_bgcolor="#0b3d0b",
    )
    return fig


# ==========================================================
# 4b. 全賽季手感起伏圖 (Rolling OPS 走勢圖)
# ==========================================================


def compute_rolling_ops(pa_df_full: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    以「整個賽季」逐場資料計算每場 OPS 分量，並取近 N 場（預設 15 場）動態滾動 OPS。

    做法：
      1. 依 game_num 分組，逐場加總 打數/安打/壘打數/保送 等基礎分量。
      2. 用 rolling(window).sum() 對這些「分量」做滾動加總（而非直接對 OPS 值取滾動平均），
         確保滾動 OPS 的分子分母關係正確（rate-of-rates 的正確算法），
         避免『先算單場 OPS 再對 OPS 取平均』造成的偏誤。
      3. 資料不足 window 場數的最前面幾場，改用「目前已累積場次」的滾動視窗
         （min_periods=1），讓賽季初期也能顯示逐步累積的動態 OPS。
    """
    if pa_df_full.empty or "game_num" not in pa_df_full.columns:
        return pd.DataFrame(columns=["game_num", "rolling_ops"])

    df = pa_df_full.copy()
    non_ab_events = ["walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play", "catcher_interf"]
    df["is_ab_r"] = ~df["events"].isin(non_ab_events)
    df["is_bb_r"] = df["events"] == "walk"
    # [OBP Fix] 滾動 OPS 同步套用官方 OBP 公式，納入 HBP／SF。
    df["is_hbp_r"] = df["events"] == "hit_by_pitch"
    df["is_sf_r"] = df["events"].isin(["sac_fly", "sac_fly_double_play"])
    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    df["tb_r"] = df["events"].map(tb_map).fillna(0)
    df["is_hit_r"] = df["events"].isin(HIT_EVENTS)

    per_game = df.groupby("game_num").agg(
        ab=("is_ab_r", "sum"),
        bb=("is_bb_r", "sum"),
        hbp=("is_hbp_r", "sum"),
        sf=("is_sf_r", "sum"),
        tb=("tb_r", "sum"),
        h=("is_hit_r", "sum"),
    ).reset_index().sort_values("game_num")

    roll_ab = per_game["ab"].rolling(window=window, min_periods=1).sum()
    roll_bb = per_game["bb"].rolling(window=window, min_periods=1).sum()
    roll_hbp = per_game["hbp"].rolling(window=window, min_periods=1).sum()
    roll_sf = per_game["sf"].rolling(window=window, min_periods=1).sum()
    roll_tb = per_game["tb"].rolling(window=window, min_periods=1).sum()
    roll_h = per_game["h"].rolling(window=window, min_periods=1).sum()

    obp_denom = (roll_ab + roll_bb + roll_hbp + roll_sf).replace(0, np.nan)
    obp = (roll_h + roll_bb + roll_hbp) / obp_denom
    slg = roll_tb / roll_ab.replace(0, np.nan)
    rolling_ops = (obp + slg).fillna(0.0)

    return pd.DataFrame({"game_num": per_game["game_num"], "rolling_ops": rolling_ops})


def render_rolling_ops_chart(pa_df_full: pd.DataFrame, player_name: str, window: int = 15):
    """繪製全賽季「近 N 場動態滾動 OPS」折線圖，讓使用者一眼看出爆發期與低潮期。

    [Fix 1] 繪圖防護機制：
        若球員在該賽季無有效擊球數據（roll_df 為空），不再強行建立/回傳一個
        「看起來正常但其實沒有資料」的 Figure，而是直接回傳 None，交由呼叫端
        以 st.warning() 溫和提示使用者，不讓程式繼續嘗試繪製造成非預期行為。
    """
    roll_df = compute_rolling_ops(pa_df_full, window=window)

    if roll_df.empty:
        return None

    season_avg_ops = roll_df["rolling_ops"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_df["game_num"], y=roll_df["rolling_ops"],
        mode="lines", line=dict(color="#00E676", width=3),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.12)",
        name=f"近 {window} 場滾動 OPS",
        # [Fix 1] 原本用 Python 的 "%d" 搭配 % 運算子組字串，卻同時把 Plotly 專屬的
        # "%{x}" / "%{y}" 樣板語法留在同一個字串裡再套用 % 格式化；"%{" 對 Python 的
        # % 運算子而言不是合法的格式碼，因而在執行期直接丟出 ValueError 讓整頁 Crash。
        # 改用 f-string：Python 端要代入的 window 用一般 f-string 語法 {window}，
        # Plotly 端要保留給前端樣板引擎解析的 {x} / {y} 則用雙大括號 {{x}} / {{y}} 轉義，
        # 兩者互不干擾。
        hovertemplate=f"第 {{x}} 場<br>近{window}場滾動 OPS: {{y:.3f}}<extra></extra>",
    ))
    fig.add_hline(
        y=season_avg_ops, line=dict(color="#FFD54F", dash="dash", width=1.5),
        annotation_text=f"賽季均值 {season_avg_ops:.3f}", annotation_position="top left",
    )
    fig.update_layout(
        title=f"🔥 {player_name} 全賽季手感起伏圖（近 {window} 場動態滾動 OPS）",
        xaxis_title="場次 (Game Number)",
        yaxis_title="Rolling OPS",
        width=980, height=380,
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#111111", paper_bgcolor="#111111",
        font=dict(color="#EEEEEE"),
    )
    # [優化 5] X 軸刻度防重疊：賽季最多可達 162 場，若刻度逐一標示（1,2,3...162）
    # 會在窄版面下擠成一團數字看不清楚。改用 tickmode='auto' 交給 Plotly 自動挑選
    # 刻度位置，並以 nticks 限制刻度數量上限（約 12~15 個），確保無論賽季總場次
    # 多寡，X 軸的場次數字都能維持適當間距、清楚可讀。
    fig.update_xaxes(gridcolor="#333333", tickmode="auto", nticks=12)
    fig.update_yaxes(gridcolor="#333333")
    return fig


def render_pk_rolling_ops_chart(pa_df_a: pd.DataFrame, name_a: str,
                                 pa_df_b: pd.DataFrame, name_b: str, window: int = 15):
    """繪製雙球員「近 N 場滾動 OPS」雙線疊加對比圖（僅供 PK 對比模式使用）。

    [Fix 3] PK 圖表安全疊加：呼叫端必須先確認球員 A、B 兩者皆有有效的例行賽
    資料，才能呼叫本函式；本函式內部同時也會再次檢查兩邊的 roll_df 是否皆非空，
    任一方無資料就直接回傳 None，絕不畫出「單邊缺線」或空白誤導的疊加圖。
    """
    roll_a = compute_rolling_ops(pa_df_a, window=window)
    roll_b = compute_rolling_ops(pa_df_b, window=window)

    if roll_a.empty or roll_b.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_a["game_num"], y=roll_a["rolling_ops"],
        mode="lines", line=dict(color="#00E676", width=3),
        name=f"🔵 {name_a}",
        hovertemplate=f"{name_a}｜第 {{x}} 場<br>近{window}場滾動 OPS: {{y:.3f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll_b["game_num"], y=roll_b["rolling_ops"],
        mode="lines", line=dict(color="#FF5252", width=3),
        name=f"🔴 {name_b}",
        hovertemplate=f"{name_b}｜第 {{x}} 場<br>近{window}場滾動 OPS: {{y:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"🥊 {name_a} vs {name_b}　近 {window} 場滾動 OPS 對比",
        xaxis_title="場次 (Game Number)",
        yaxis_title="Rolling OPS",
        width=980, height=380,
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="#111111", paper_bgcolor="#111111",
        font=dict(color="#EEEEEE"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    # [優化 5] 同樣套用自動刻度 + 上限數量，避免雙人 PK 對比圖在 1~162 場的
    # 場次刻度重疊擠在一起（雙人疊圖版面通常更窄，此防護更為重要）。
    fig.update_xaxes(gridcolor="#333333", tickmode="auto", nticks=12)
    fig.update_yaxes(gridcolor="#333333")
    return fig


# ==========================================================
# 5. 一鍵生成 Scouting Report 報告卡 (PNG Exporter)
# ==========================================================


def _fig_to_pil(fig: go.Figure, w: int, h: int) -> Image.Image:
    try:
        img_bytes = fig.to_image(format="png", width=w, height=h, scale=2)
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    except Exception:
        # kaleido 若未安裝，回傳空白占位圖
        placeholder = Image.new("RGBA", (w, h), (34, 34, 34, 255))
        d = ImageDraw.Draw(placeholder)
        d.text((20, h // 2), "圖表匯出需安裝 kaleido 套件", fill="white")
        return placeholder


# [優化 2] CJK 字型可用性偵測：
# 純西文字型（DejaVuSans / Arial 等）雖然「載入成功」，但完全不含中文字形，
# 繪圖時中文字元會被畫成缺字方塊（tofu）。這裡用路徑關鍵字概略判斷該候選字型
# 是否具備中文字形能力，供 main() 在側邊欄顯示「⚠️ 尚未偵測到中文字型」提醒，
# 讓使用者知道要放入 assets/NotoSansTC-Regular.ttf 或設定 packages.txt。
_CJK_CAPABLE_KEYWORDS = ("notosans", "noto-cjk", "notosanscjk", "wqy", "pingfang", "msjh", "assets")
_cjk_font_available_cache = {"checked": False, "available": False}


def _path_looks_cjk_capable(path: str) -> bool:
    lower = path.lower()
    return any(kw in lower for kw in _CJK_CAPABLE_KEYWORDS)


def check_cjk_font_status() -> bool:
    """實際嘗試載入候選清單中『看起來具備中文字形』的字型，回傳是否成功找到。

    結果快取於模組全域變數，避免每次 rerun 都重複掃描磁碟。
    """
    if _cjk_font_available_cache["checked"]:
        return _cjk_font_available_cache["available"]

    found = False
    for path in FONT_CANDIDATES_REGULAR:
        if not _path_looks_cjk_capable(path):
            continue
        try:
            ImageFont.truetype(path, 20)
            found = True
            break
        except Exception:
            continue

    _cjk_font_available_cache["checked"] = True
    _cjk_font_available_cache["available"] = found
    return found


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    [Fix 1] 跨平台安全字型載入。
    依序嘗試常見 Linux / macOS / Windows 字型路徑；若全部失敗，
    改用 PIL 內建 bitmap 字型，但盡量透過 load_default(size=...)
    （Pillow >= 10.1 支援）維持指定字級，避免匯出圖片文字被壓縮成
    無法辨識的極小點、排版整個跑掉。若該 Pillow 版本不支援
    size 參數，最後才退回無法調整大小的預設字型，並在主流程中
    可視需要提示使用者安裝合適字型。
    """
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    # 全部候選路徑都載入失敗，才退回內建字型
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        # 舊版 Pillow 不支援 size 參數，最後手段
        return ImageFont.load_default()


def apply_brand_watermark(
    card: Image.Image,
    corner: str = "br",
    scale: float = 1.0,
) -> None:
    """
    🏷️ 在圖卡上加入專屬品牌浮水印：boarcast

    於指定角落（預設右下角 br）繪製：
        - 主標：boarcast（螢光綠 #00E676，粗體、外加一圈亮白描邊以增加對比與辨識度）
        - 副標：Powered by boarcast | Data: MLB Statcast（小字亮白）

    設計目的：確保圖卡被分享到社群平台（PTT / IG / Threads）時，
    即使被裁切、壓縮或轉發，仍具備高度可辨識的品牌識別度。

    corner: "br" 右下角 (預設) / "tr" 右上角 / "tl" 左上角 / "bl" 左下角
    scale: 依卡片尺寸縮放浮水印字級與邊距（IG 直式圖卡尺寸較大，可傳入 >1 的 scale）
    """
    W, H = card.size
    draw = ImageDraw.Draw(card, "RGBA")

    brand_font = _load_font(int(40 * scale), bold=True)
    tag_font = _load_font(int(20 * scale), bold=False)

    margin = int(36 * scale)

    brand_bbox = draw.textbbox((0, 0), BRAND_NAME, font=brand_font)
    brand_w, brand_h = brand_bbox[2] - brand_bbox[0], brand_bbox[3] - brand_bbox[1]
    tag_bbox = draw.textbbox((0, 0), BRAND_TAGLINE, font=tag_font)
    tag_w, tag_h = tag_bbox[2] - tag_bbox[0], tag_bbox[3] - tag_bbox[1]

    block_w = max(brand_w, tag_w)
    block_h = brand_h + int(8 * scale) + tag_h

    if corner == "br":
        x0, y0 = W - margin - block_w, H - margin - block_h
    elif corner == "tr":
        x0, y0 = W - margin - block_w, margin
    elif corner == "tl":
        x0, y0 = margin, margin
    else:  # "bl"
        x0, y0 = margin, H - margin - block_h

    # 半透明底色色塊，讓浮水印在任何背景（球場綠地／深色卡片）上都清晰可辨
    pad = int(14 * scale)
    draw.rounded_rectangle(
        [x0 - pad, y0 - pad, x0 + block_w + pad, y0 + block_h + pad],
        radius=int(10 * scale),
        fill=(10, 10, 10, 150),
    )

    # 主標：螢光綠字 + 亮白細描邊，社群平台縮圖也能一眼辨識
    outline = int(max(1, 2 * scale))
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x0 + dx, y0 + dy), BRAND_NAME, font=brand_font, fill=BRAND_COLOR_BRIGHT)
    draw.text((x0, y0), BRAND_NAME, font=brand_font, fill=BRAND_COLOR_NEON)

    draw.text((x0, y0 + brand_h + int(8 * scale)), BRAND_TAGLINE, font=tag_font, fill=BRAND_COLOR_BRIGHT)


def generate_scouting_card(
    player_name: str,
    year: int,
    stats: dict,
    is_scaled: bool,
    game_range: tuple,
    headshot_url: str,
    hotzone_fig: go.Figure,
    spray_fig: go.Figure,
) -> io.BytesIO:
    """合成球員照片 + 數據 + 熱區 + 落點圖成一張高畫質 PNG 報告卡"""
    W, H = 1600, 2000
    card = Image.new("RGB", (W, H), (17, 17, 17))
    draw = ImageDraw.Draw(card)

    title_font = _load_font(56, bold=True)
    sub_font = _load_font(30, bold=True)
    body_font = _load_font(28)
    small_font = _load_font(22)

    # 標頭
    draw.text((60, 40), f"{player_name}", fill="white", font=title_font)
    draw.text((60, 110), "SCOUTING REPORT", fill="#00E676", font=sub_font)
    mode_text = "🚀 162 場全賽季模擬預測" if is_scaled else f"選定區間：第 {game_range[0]}~{game_range[1]} 場"
    draw.text((60, 155), f"{year} 賽季　|　{mode_text}", fill="#BBBBBB", font=body_font)

    # 球員照片
    try:
        resp = requests.get(headshot_url, timeout=8)
        headshot = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        headshot = headshot.resize((260, 260))
        card.paste(headshot, (W - 320, 40), headshot)
    except Exception:
        pass

    y = 230
    draw.line((60, y, W - 60, y), fill="#333333", width=2)
    y += 30

    # 傳統成績列
    line1 = f"AVG {stats['AVG']:.3f}    OBP {stats['OBP']:.3f}    SLG {stats['SLG']:.3f}    OPS {stats['OPS']:.3f}    ISO {stats['ISO']:.3f}"
    draw.text((60, y), line1, fill="white", font=sub_font)
    y += 55

    line2 = (f"H {stats['H']}  (1B {stats['1B']} / 2B {stats['2B']} / 3B {stats['3B']} / HR {stats['HR']})   "
             f"AB {stats['AB']}   RBI {stats['RBI']}")
    draw.text((60, y), line2, fill="white", font=body_font)
    y += 45

    line3 = f"BB {stats['BB']}   SO {stats['SO']}   SB {stats['SB']}   CS {stats['CS']}"
    draw.text((60, y), line3, fill="white", font=body_font)
    y += 45

    avg_ev = stats['Avg_EV']
    max_ev = stats['Max_EV']
    avg_la = stats['Avg_LA']
    ev_txt = (
        f"Avg EV {avg_ev:.1f} mph   Max EV {max_ev:.1f} mph   Avg LA {avg_la:.1f}°"
        if not (np.isnan(avg_ev) if isinstance(avg_ev, float) else False)
        else "Statcast 物理數據不足"
    )
    draw.text((60, y), ev_txt, fill="#FFD54F", font=body_font)
    y += 45

    # [Enh 2] 進階期望數據列 (xBA / xSLG / Barrel% / Hard Hit%)
    xba_v = stats.get("xBA", np.nan)
    xslg_v = stats.get("xSLG", np.nan)
    brl_v = stats.get("Barrel_Pct", np.nan)
    hh_v = stats.get("HardHit_Pct", np.nan)
    xba_txt = f"{xba_v:.3f}" if pd.notna(xba_v) else "N/A"
    xslg_txt = f"{xslg_v:.3f}" if pd.notna(xslg_v) else "N/A"
    brl_txt = f"{brl_v * 100:.1f}%" if pd.notna(brl_v) else "N/A"
    hh_txt = f"{hh_v * 100:.1f}%" if pd.notna(hh_v) else "N/A"
    exp_txt = f"xBA {xba_txt}   xSLG {xslg_txt}   Barrel% {brl_txt}   Hard Hit% {hh_txt}"
    draw.text((60, y), exp_txt, fill="#80D8FF", font=body_font)
    y += 60

    draw.line((60, y, W - 60, y), fill="#333333", width=2)
    y += 30

    # 貼上熱區與落點圖
    hz_img = _fig_to_pil(hotzone_fig, 700, 700)
    sp_img = _fig_to_pil(spray_fig, 780, 780)
    card.paste(hz_img.resize((700, 700)), (60, y))
    card.paste(sp_img.resize((760, 760)), (800, y))
    y += 740

    draw.text((60, H - 60), f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Data: MLB Statcast via pybaseball",
               fill="#666666", font=small_font)

    # 🏷️ boarcast 專屬品牌浮水印（右下角，螢光綠 + 亮白描邊，附「Powered by boarcast」小字說明）
    apply_brand_watermark(card, corner="br", scale=1.0)

    buf = io.BytesIO()
    card.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ==========================================================
# 5b. 左右投拆分 & 球種過濾（新增功能 #1 / #2）
# ==========================================================


def apply_platoon_pitch_filter(df: pd.DataFrame, platoon_choice: str, selected_pitch_types: list) -> pd.DataFrame:
    """依左右投拆分（p_throws）與球種過濾器（pitch_type）篩選打席資料"""
    out = df
    if platoon_choice == "面對左投 vs LHP" and "p_throws" in out.columns:
        out = out[out["p_throws"] == "L"]
    elif platoon_choice == "面對右投 vs RHP" and "p_throws" in out.columns:
        out = out[out["p_throws"] == "R"]

    if "pitch_type" in out.columns and selected_pitch_types:
        out = out[out["pitch_type"].isin(selected_pitch_types)]

    return out


def render_platoon_pitch_filters(pa_df: pd.DataFrame, key_prefix: str):
    """繪製左右投拆分選單 + 球種多選過濾器，回傳 (platoon_choice, selected_pitch_types)"""
    colf1, colf2 = st.columns([1, 2])
    with colf1:
        platoon_choice = st.selectbox(
            "⚔️ 左右投拆分 (Platoon Splits)",
            PLATOON_OPTIONS,
            key=f"{key_prefix}_platoon",
        )
    with colf2:
        if "pitch_type" in pa_df.columns:
            available_pitch_types = sorted(pa_df["pitch_type"].dropna().unique().tolist())
        else:
            available_pitch_types = []
        pitch_labels = [PITCH_TYPE_NAMES.get(pt, pt) for pt in available_pitch_types]
        label_to_code = dict(zip(pitch_labels, available_pitch_types))
        selected_labels = st.multiselect(
            "🎯 球種過濾器 (Pitch Type Filter)",
            pitch_labels,
            default=pitch_labels,
            key=f"{key_prefix}_pitchtype",
        )
        selected_pitch_types = [label_to_code[l] for l in selected_labels]
    return platoon_choice, selected_pitch_types


# ==========================================================
# 5c. 雙球員 PK 對比卡合成（新增功能 #3）
# ==========================================================


def generate_pk_card(buf_a: io.BytesIO, buf_b: io.BytesIO, name_a: str, name_b: str) -> io.BytesIO:
    """將兩張個人 Scouting Report 卡片左右拼接成一張雙人 PK 對決圖卡"""
    img_a = Image.open(buf_a).convert("RGB")
    img_b = Image.open(buf_b).convert("RGB")

    h = max(img_a.height, img_b.height)
    gap = 20
    vs_w = 140
    w = img_a.width + img_b.width + gap * 2 + vs_w

    canvas = Image.new("RGB", (w, h), (10, 10, 10))
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (img_a.width + gap * 2 + vs_w, 0))

    draw = ImageDraw.Draw(canvas)
    vs_font = _load_font(64, bold=True)
    vs_x = img_a.width + gap
    draw.rectangle([vs_x, 0, vs_x + vs_w, h], fill=(20, 20, 20))
    vs_text = "VS"
    tb = draw.textbbox((0, 0), vs_text, font=vs_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((vs_x + (vs_w - tw) / 2, h / 2 - th / 2 - 200), vs_text, fill="#E53935", font=vs_font)

    # 🏷️ 整張 PK 對決圖卡頂部中央再蓋一次 boarcast 浮水印，強化整體品牌識別度
    apply_brand_watermark(canvas, corner="tr", scale=1.1)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ==========================================================
# 5d. IG 限時動態專用圖卡 (9:16 直式 1080x1920)
# ==========================================================


def generate_ig_story_card(
    player_name: str,
    year: int,
    stats: dict,
    is_scaled: bool,
    game_range: tuple,
    headshot_url: str,
    spray_fig: go.Figure,
) -> io.BytesIO:
    """
    生成 9:16 直式 IG 限時動態圖卡（1080x1920），供創作者免裁切一鍵發布。

    版面配置（由上至下）：
      1. 頂部：球員名 + 賽季/區間標籤
      2. 放大版球員頭像（置中圓形裁切）
      3. 三圍數據大字：AVG / OBP / SLG / OPS
      4. 落點圖（縮小置入）
      5. boarcast 浮水印（右下角，社群分享識別）
    """
    W, H = 1080, 1920
    card = Image.new("RGB", (W, H), (10, 10, 12))
    draw = ImageDraw.Draw(card)

    # 背景漸層色塊，增加 IG 限動的視覺層次
    for i in range(H):
        shade = int(10 + (i / H) * 14)
        draw.line([(0, i), (W, i)], fill=(shade, shade, shade + 4))

    name_font = _load_font(72, bold=True)
    tag_font = _load_font(34, bold=True)
    stat_num_font = _load_font(64, bold=True)
    stat_label_font = _load_font(28, bold=False)

    # ---- 頂部標題 ----
    name_bbox = draw.textbbox((0, 0), player_name, font=name_font)
    name_w = name_bbox[2] - name_bbox[0]
    draw.text(((W - name_w) / 2, 90), player_name, fill="white", font=name_font)

    mode_text = "🚀 162 場全賽季模擬" if is_scaled else f"第 {game_range[0]}~{game_range[1]} 場"
    tag_text = f"{year} 賽季　|　{mode_text}"
    tag_bbox = draw.textbbox((0, 0), tag_text, font=tag_font)
    tag_w = tag_bbox[2] - tag_bbox[0]
    draw.text(((W - tag_w) / 2, 185), tag_text, fill=BRAND_COLOR_NEON, font=tag_font)

    # ---- 放大版球員頭像（圓形裁切，置中） ----
    avatar_d = 520
    avatar_cx, avatar_top = W // 2, 260
    try:
        resp = requests.get(headshot_url, timeout=8)
        headshot = Image.open(io.BytesIO(resp.content)).convert("RGBA").resize((avatar_d, avatar_d))
        mask = Image.new("L", (avatar_d, avatar_d), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, avatar_d, avatar_d), fill=255)
        ring = Image.new("RGBA", (avatar_d + 16, avatar_d + 16), (0, 0, 0, 0))
        ring_draw = ImageDraw.Draw(ring)
        ring_draw.ellipse((0, 0, avatar_d + 16, avatar_d + 16), fill=BRAND_COLOR_NEON)
        card.paste(ring, (avatar_cx - avatar_d // 2 - 8, avatar_top - 8), ring)
        card.paste(headshot, (avatar_cx - avatar_d // 2, avatar_top), mask)
    except Exception:
        pass

    y = avatar_top + avatar_d + 60

    # ---- 三圍數據大字：AVG / OBP / SLG / OPS ----
    quad = [
        ("AVG", f"{stats['AVG']:.3f}"),
        ("OBP", f"{stats['OBP']:.3f}"),
        ("SLG", f"{stats['SLG']:.3f}"),
        ("OPS", f"{stats['OPS']:.3f}"),
    ]
    col_w = W // 4
    for i, (label, val) in enumerate(quad):
        cx = col_w * i + col_w // 2
        val_bbox = draw.textbbox((0, 0), val, font=stat_num_font)
        val_w = val_bbox[2] - val_bbox[0]
        draw.text((cx - val_w / 2, y), val, fill="white", font=stat_num_font)
        lbl_bbox = draw.textbbox((0, 0), label, font=stat_label_font)
        lbl_w = lbl_bbox[2] - lbl_bbox[0]
        draw.text((cx - lbl_w / 2, y + 80), label, fill="#AAAAAA", font=stat_label_font)
    y += 150

    hr_line = f"HR {stats['HR']}    RBI {stats['RBI']}    H {stats['H']}"
    hr_bbox = draw.textbbox((0, 0), hr_line, font=tag_font)
    draw.text(((W - (hr_bbox[2] - hr_bbox[0])) / 2, y), hr_line, fill="#FFD54F", font=tag_font)
    y += 70

    draw.line((80, y, W - 80, y), fill="#333333", width=2)
    y += 30

    # ---- 落點圖（縮小置入，寬度貼齊卡片） ----
    sp_w = W - 120
    sp_h = int(sp_w * 0.85)
    sp_img = _fig_to_pil(spray_fig, sp_w, sp_h)
    card.paste(sp_img.resize((sp_w, sp_h)), (60, y))

    # ---- boarcast 浮水印（右下角，scale 放大以配合直式大圖） ----
    apply_brand_watermark(card, corner="br", scale=1.3)

    buf = io.BytesIO()
    card.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ==========================================================
# 5e. PNG 圖卡匯出快取（[優化 1] 解決 Kaleido 效能瓶頸）
# ==========================================================
#
# 問題背景：fig.to_image() 底層透過 Kaleido 呼叫無頭 Chromium 渲染 Plotly 圖表，
# 速度慢且吃記憶體；PK 模式一次要合成兩張 Scouting Report（各自內含熱區圖 + 落點圖）
# 再拼接成對決卡，等於一次觸發 4 次以上的 Kaleido 渲染，在 Streamlit Community Cloud
# 這類免費資源環境上很容易 Timeout 或 OOM。
#
# 解法：以「球員/賽季/場次區間/是否外推 + 關鍵數據指紋」組成 cache key，
# 快取最終合成好的 PNG bytes（而非 Figure 物件本身，避免序列化 Plotly Figure 的額外負擔）。
# 使用者在同一個 session 中重複點擊相同參數的匯出按鈕時，直接回傳先前已合成好的圖檔，
# 不再重新呼叫 Kaleido。


def _stats_fingerprint(stats: dict) -> tuple:
    """將 stats 字典中會影響圖卡畫面的關鍵數值，轉成可雜湊的 tuple 指紋，供快取 key 使用。"""
    keys = [
        "AVG", "OBP", "SLG", "OPS", "ISO", "1B", "2B", "3B", "HR", "H", "AB",
        "SB", "CS", "BB", "SO", "RBI", "Avg_EV", "Max_EV", "Avg_LA",
        "xBA", "xSLG", "Barrel_Pct", "HardHit_Pct",
        "GB_Pct", "LD_Pct", "FB_Pct", "PU_Pct",
    ]
    fp = []
    for k in keys:
        v = stats.get(k)
        if isinstance(v, float):
            v = None if np.isnan(v) else round(v, 4)
        fp.append(v)
    return tuple(fp)


def get_card_cache() -> dict:
    """跨 rerun 存活的 PNG 圖卡快取（存放於 st.session_state，key -> PNG bytes）。"""
    return st.session_state.setdefault("_card_cache", {})


def _cached_png(cache_key: tuple, build_fn) -> io.BytesIO:
    """依 cache_key 查快取；命中則直接回傳既有 PNG bytes 包成的新 BytesIO（避免共用同一個
    已被讀取過、游標移動過的 BytesIO 物件），未命中才呼叫 build_fn() 實際合成圖卡並存入快取。
    """
    cache = get_card_cache()
    if cache_key in cache:
        return io.BytesIO(cache[cache_key])
    buf = build_fn()
    data = buf.getvalue()
    cache[cache_key] = data
    return io.BytesIO(data)


def generate_scouting_card_cached(player_name, year, stats, is_scaled, game_range,
                                   headshot_url, hotzone_fig, spray_fig) -> io.BytesIO:
    key = ("scouting", player_name, year, tuple(game_range), is_scaled, _stats_fingerprint(stats))
    return _cached_png(key, lambda: generate_scouting_card(
        player_name=player_name, year=year, stats=stats, is_scaled=is_scaled,
        game_range=game_range, headshot_url=headshot_url,
        hotzone_fig=hotzone_fig, spray_fig=spray_fig,
    ))


def generate_ig_story_card_cached(player_name, year, stats, is_scaled, game_range,
                                   headshot_url, spray_fig) -> io.BytesIO:
    key = ("ig", player_name, year, tuple(game_range), is_scaled, _stats_fingerprint(stats))
    return _cached_png(key, lambda: generate_ig_story_card(
        player_name=player_name, year=year, stats=stats, is_scaled=is_scaled,
        game_range=game_range, headshot_url=headshot_url, spray_fig=spray_fig,
    ))


def generate_pk_card_cached(buf_a: io.BytesIO, buf_b: io.BytesIO, name_a: str, name_b: str,
                             fingerprint_a: tuple, fingerprint_b: tuple) -> io.BytesIO:
    key = ("pk", name_a, name_b, fingerprint_a, fingerprint_b)
    return _cached_png(key, lambda: generate_pk_card(buf_a, buf_b, name_a, name_b))


def load_player(fn: str, ln: str, year: int):
    """統一的球員資料載入流程，回傳 (pa_df, player_id, total_games) 或 (None, None, None)。

    [Enh 3] 跨頁面/跨模式 session_state 快取重用：
    「主頁面單人分析」與「雙球員 PK 對比模式」共用同一份 st.session_state["_player_cache"]，
    以正規化（去除頭尾空白 + 轉小寫）後的 (first_name, last_name, year) 作為 key。
    例如已在主頁面載入過 Shohei Ohtani，PK 模式再次選擇 Ohtani（即使大小寫或空白不同）時，
    會直接命中此快取，不再重複觸發 get_player_data /（進一步）statcast_batter 的資料處理與
    HTTP 請求，降低對 MLB / Baseball Savant 伺服器的請求頻率並縮短等候時間。
    底層 get_player_data 本身也已用 @st.cache_data 依「原始大小寫」的參數快取，
    這裡的正規化 key 則額外處理「同一人但輸入大小寫/空白不同」時仍能命中快取的情況。
    """
    cache = st.session_state.setdefault("_player_cache", {})
    cache_key = (fn.strip().lower(), ln.strip().lower(), year)

    if cache_key in cache:
        return cache[cache_key]

    try:
        pa_df, player_id, total_games = get_player_data(fn, ln, year)
    except Exception as e:
        st.error(f"抓取 {fn} {ln} 資料時發生錯誤：{e}")
        return None, None, None

    # [Fix 5] 無數據溫和提示：區分「真的查無此球員」與「查得到球員，但該年
    # 未出賽 / 沒有例行賽 Statcast 擊球紀錄」兩種情況，後者改用 st.info() 溫和
    # 提示，而非讓使用者誤以為是程式錯誤的預設 Python 錯誤訊息。
    if pa_df is None:
        if player_id is None:
            st.error(f"找不到球員 {fn} {ln}，請確認姓名拼寫是否正確。")
        else:
            st.info(f"ℹ️ {fn} {ln} 於 {year} 賽季無例行賽擊球數據，暫無法顯示分析面板。")
        return None, None, None

    if pa_df.empty:
        st.info(f"ℹ️ {fn} {ln} 於 {year} 賽季無例行賽擊球數據，暫無法顯示分析面板。")
        return None, None, None

    cache[cache_key] = (pa_df, player_id, total_games)
    return pa_df, player_id, total_games


def render_player_panel(pa_df: pd.DataFrame, player_name: str, player_id: int,
                         total_games: int, year: int, key_prefix: str, width_scale: float = 1.0):
    """繪製單一球員的完整分析面板（供主板與 PK 模式共用），回傳 (stats, fig_hz, fig_spray)"""
    headshot_url = fetch_player_headshot_url(player_id)

    col_img, col_info = st.columns([1, 4])
    with col_img:
        st.image(headshot_url, width=int(120 * width_scale))
    with col_info:
        st.subheader(f"👤 {player_name}（{year}）")
        team = determine_player_team(pa_df)
        bats = pa_df["stand"].iloc[-1] if "stand" in pa_df.columns else "N/A"
        st.caption(f"總打席：{len(pa_df)}　|　球隊：{team}　|　打擊習慣：{bats}")

    max_games = max(int(pa_df["game_num"].max()), 1)
    slider_max = max(max_games, 2)
    range_key = f"{key_prefix}_range"

    # ---- 近期手感快捷鍵：近 7 / 15 / 30 場 ----
    # 點擊後直接改寫 session_state 中滑桿的值，並 rerun，
    # 讓滑桿自動鎖定在該球員「最新」的場次區間（即 [max(1, 總場次-N+1), 總場次]）。
    st.caption("⏱️ 近期手感快捷鍵")
    qb1, qb2, qb3, qb_spacer = st.columns([1, 1, 1, 3])
    if qb1.button("🔥 近 7 場", key=f"{key_prefix}_quick7"):
        st.session_state[range_key] = (max(1, max_games - 7 + 1), max_games)
        st.rerun()
    if qb2.button("⚡ 近 15 場", key=f"{key_prefix}_quick15"):
        st.session_state[range_key] = (max(1, max_games - 15 + 1), max_games)
        st.rerun()
    if qb3.button("📊 近 30 場", key=f"{key_prefix}_quick30"):
        st.session_state[range_key] = (max(1, max_games - 30 + 1), max_games)
        st.rerun()

    game_range = st.slider(
        "選定場次區間", min_value=1, max_value=slider_max,
        value=(1, max_games), key=range_key,
    )
    # [2026 Rev] 外推功能改為「門檻式」開放：樣本場次數不足 MIN_GAMES_FOR_EXTRAPOLATION
    # （預設 10 場）時，直接停用外推開關並提示原因，而非讓使用者勾選後才顯示警告，
    # 從源頭避免賽季初期極端小樣本被外推成失真的整季預測。
    games_in_sample_preview = game_range[1] - game_range[0] + 1
    extrapolation_allowed = games_in_sample_preview >= MIN_GAMES_FOR_EXTRAPOLATION
    if extrapolation_allowed:
        scale_to_162 = st.checkbox("🚀 外推至 162 場模擬", value=False, key=f"{key_prefix}_scale")
    else:
        st.checkbox(
            "🚀 外推至 162 場模擬", value=False, key=f"{key_prefix}_scale",
            disabled=True,
            help=f"選定區間僅 {games_in_sample_preview} 場比賽，未達最低 {MIN_GAMES_FOR_EXTRAPOLATION} 場門檻，暫不開放外推模擬。",
        )
        scale_to_162 = False
        st.caption(f"ℹ️ 選定區間僅 {games_in_sample_preview} 場比賽（未達 {MIN_GAMES_FOR_EXTRAPOLATION} 場門檻），"
                   f"外推至 162 場功能暫時停用，避免賽季初期小樣本產生失真預測。")

    platoon_choice, selected_pitch_types = render_platoon_pitch_filters(pa_df, key_prefix)

    games_in_sample = game_range[1] - game_range[0] + 1
    range_df = pa_df[(pa_df["game_num"] >= game_range[0]) & (pa_df["game_num"] <= game_range[1])]
    filtered_df = apply_platoon_pitch_filter(range_df, platoon_choice, selected_pitch_types)

    stats = calculate_stats(filtered_df, scale_to_162, games_in_sample)

    if scale_to_162 and (stats.get("Extrap_Capped") or stats.get("Extrap_LowSample")):
        if stats.get("Extrap_Capped"):
            st.warning(
                f"⚠️ 選定區間僅 {games_in_sample} 場比賽，外推倍率已從理論值上限鎖定為 "
                f"{stats['Extrap_Factor']:.1f} 倍，避免產生誇張的極端預測值；此結果僅供參考。"
            )
        elif stats.get("Extrap_LowSample"):
            st.warning(f"⚠️ 選定區間僅 {games_in_sample} 場比賽，樣本數過少，外推結果僅供參考。")

    # [UI Fix] 底部核心數據列重構：標籤（細體、低飽和灰藍）與數值（粗體、高亮白）
    # 視覺權重分流，Flexbox baseline 對齊，並加上底部呼吸空間，避免緊貼黑底。
    _stat_row_items = [
        ("AVG", f"{stats['AVG']:.3f}"),
        ("OBP", f"{stats['OBP']:.3f}"),
        ("SLG", f"{stats['SLG']:.3f}"),
        ("OPS", f"{stats['OPS']:.3f}"),
        ("ISO", f"{stats['ISO']:.3f}"),
        ("HR", f"{stats['HR']}"),
    ]
    _stat_row_html = "".join(
        f'<div style="display:flex;flex-direction:column;align-items:flex-start;">'
        f'<span style="font-weight:500;color:#8892b0;font-size:0.78rem;'
        f'letter-spacing:0.06em;text-transform:uppercase;">{label}</span>'
        f'<span style="font-weight:700;color:#ffffff;font-size:1.35rem;'
        f'font-variant-numeric:tabular-nums;">{value}</span>'
        f'</div>'
        for label, value in _stat_row_items
    )
    st.markdown(
        f'<div style="display:flex;flex-direction:row;align-items:baseline;'
        f'gap:24px;padding-bottom:24px;flex-wrap:wrap;">{_stat_row_html}</div>',
        unsafe_allow_html=True,
    )

    # [Enh 2] 進階期望數據列（xBA / xSLG / Barrel% / Hard Hit%）
    xba_txt = f"{stats['xBA']:.3f}" if pd.notna(stats["xBA"]) else "N/A"
    xslg_txt = f"{stats['xSLG']:.3f}" if pd.notna(stats["xSLG"]) else "N/A"
    barrel_txt = f"{stats['Barrel_Pct'] * 100:.1f}%" if pd.notna(stats["Barrel_Pct"]) else "N/A"
    hardhit_txt = f"{stats['HardHit_Pct'] * 100:.1f}%" if pd.notna(stats["HardHit_Pct"]) else "N/A"
    st.caption(f"📈 期望數據　xBA {xba_txt}　|　xSLG {xslg_txt}　|　Barrel% {barrel_txt}　|　Hard Hit% {hardhit_txt}")

    # ---- 擊球型態分佈 (GB% / LD% / FB% / PU%) ----
    gb_txt = f"{stats['GB_Pct'] * 100:.1f}%" if pd.notna(stats.get("GB_Pct")) else "N/A"
    ld_txt = f"{stats['LD_Pct'] * 100:.1f}%" if pd.notna(stats.get("LD_Pct")) else "N/A"
    fb_txt = f"{stats['FB_Pct'] * 100:.1f}%" if pd.notna(stats.get("FB_Pct")) else "N/A"
    pu_txt = f"{stats['PU_Pct'] * 100:.1f}%" if pd.notna(stats.get("PU_Pct")) else "N/A"
    st.caption(f"🎯 擊球型態分佈　滾地球 GB% {gb_txt}　|　平飛球 LD% {ld_txt}　|　"
               f"飛球 FB% {fb_txt}　|　內野高飛 PU% {pu_txt}")

    metric_choice = st.radio(
        "熱區指標", HOT_ZONE_METRICS, horizontal=True,
        label_visibility="collapsed", key=f"{key_prefix}_metric",
    )
    fig_hz = render_hot_zone(filtered_df, metric_choice)

    # [Enh 1] 依打席資料中最常見的 home_team，帶入該球隊主場球場的向量外框
    if "home_team" in filtered_df.columns and not filtered_df["home_team"].dropna().empty:
        home_team = filtered_df["home_team"].mode().iloc[0]
    elif "home_team" in pa_df.columns and not pa_df["home_team"].dropna().empty:
        home_team = pa_df["home_team"].mode().iloc[0]
    else:
        home_team = None
    fig_spray = render_spray_chart(filtered_df, home_team=home_team)

    # RWD（功能 #4）：手機窄螢幕自動由並排改為上下堆疊，寬圖表改用 use_container_width
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_hz, use_container_width=True, key=f"{key_prefix}_hz_chart")
    with c2:
        st.plotly_chart(fig_spray, use_container_width=True, key=f"{key_prefix}_spray_chart")

    return stats, fig_hz, fig_spray, game_range, scale_to_162, headshot_url


# ==========================================================
# 6. Streamlit 主頁面 UI 流程控制
# ==========================================================


def main():
    st.set_page_config(page_title="MLB Scouting & 162-Game Simulator", layout="wide", page_icon="⚾")
    inject_cyberpunk_theme()
    st.title("⚾ MLB 擊球數據模擬與球員視覺化儀表板")
    st.caption(
        f"🔵 電子發光霓虹藍＝一般數據　|　🌸 極光霓虹粉紅＝極端數據警示 "
        f"(Max EV ≥ {EXIT_VELO_ALERT_MPH:.0f}mph 或面對火球 ≥ {PITCH_VELO_ALERT_MPH:.0f}mph)"
    )

    if not PYBASEBALL_AVAILABLE:
        st.error("⚠️ 尚未安裝 pybaseball，請先執行：pip install pybaseball")
        st.stop()

    # [優化 2] 雲端部署中文字型缺字防呆提醒：
    # 僅在偵測不到任何具備中文字形的字型時提示一次，避免每次都跳警告干擾使用。
    if not check_cjk_font_status():
        st.sidebar.warning(
            "⚠️ 目前環境偵測不到中文字型，匯出的 PNG 圖卡標題可能出現缺字方塊。\n\n"
            "建議：將 NotoSansTC-Regular.ttf（及 Bold 版本）放進專案的 assets/ 資料夾，"
            "或在專案根目錄新增 packages.txt 並寫入一行 `fonts-noto-cjk` 後重新部署。"
        )

    # [Fix 5] 在讀取 widget 之前，先確保 session_state 有預設值，
    # 之後 text_input 直接用 key 綁定 session_state，兩者永遠同步：
    # 熱門球員按鈕寫入 session_state["first_name"/"last_name"] 後，
    # 側邊欄輸入框會在下一次 rerun 自動顯示新值，不需要額外傳入 value。
    st.session_state.setdefault("first_name", "Shohei")
    st.session_state.setdefault("last_name", "Ohtani")

    # ---------------- 側邊欄：球員搜尋 ----------------
    st.sidebar.header("🔍 球員搜尋")
    # 注意：不再傳入 value=，改為純 key 綁定，避免 Streamlit 對
    # 「同時提供 value 與 session_state key」丟出警告，也才能讓
    # 快捷鍵點擊確實反映在輸入框上。
    first_name = st.sidebar.text_input("名 (First Name)", key="first_name")
    last_name = st.sidebar.text_input("姓 (Last Name)", key="last_name")
    selected_year = st.sidebar.selectbox("選擇賽季年份", get_selectable_years(), index=0)

    # [Fix 2] 移除非法直接賦值：
    # 先前的寫法是在 st.sidebar.text_input(key="first_name") 這個 Widget 已於
    # 本次 script 執行中「實例化」之後，才又直接對 st.session_state["first_name"]
    # 賦值，這會立刻觸發 StreamlitAPIException（"cannot be set after the widget
    # is instantiated"）。改用按鈕的 on_click 回呼函式：Streamlit 會在重新執行
    # 整份 script、也就是在 text_input 這個 Widget 重新被實例化「之前」，先執行
    # 這裡的回呼函式，此時對 session_state 賦值是安全、合法的生命週期階段，
    # 也因此不再需要額外呼叫 st.rerun()（按鈕點擊本身就會觸發一次自然的 rerun）。
    def _apply_quick_player(fn: str, ln: str):
        st.session_state["first_name"] = fn
        st.session_state["last_name"] = ln
        st.session_state["pending_load"] = True

    def _apply_random_player():
        fn, ln = POPULAR_PLAYERS[np.random.randint(0, len(POPULAR_PLAYERS))]
        _apply_quick_player(fn, ln)

    st.sidebar.markdown("**⭐ 熱門球員快捷鍵**")
    cols = st.sidebar.columns(2)
    for i, (fn, ln) in enumerate(POPULAR_PLAYERS):
        cols[i % 2].button(
            f"{fn} {ln}", key=f"quick_{fn}_{ln}",
            on_click=_apply_quick_player, args=(fn, ln),
        )

    # ---- 功能 #5：隨機載入明星球員（降低新用戶第一次使用門檻） ----
    st.sidebar.button("🎲 隨機載入明星球員", on_click=_apply_random_player)

    load_clicked = st.sidebar.button("📥 載入球員數據", type="primary")

    # [優化 4] PK 模式 / 多次查詢後的 Session State 快取清理機制：
    # _player_cache 會持續累積每位查過的球員之逐球 DataFrame，_card_cache 則累積已合成的
    # PNG bytes；長時間單一 session 內查了多位球員、多個賽季後可能占用不少記憶體。
    # 提供手動清除按鈕，讓使用者可視需要主動釋放。
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 清除記憶體快取"):
        st.session_state["_player_cache"] = {}
        st.session_state["_card_cache"] = {}
        st.sidebar.success("已釋放球員資料與圖卡快取記憶體！")

    if load_clicked or st.session_state.pop("pending_load", False):
        fn = st.session_state.get("first_name", first_name)
        ln = st.session_state.get("last_name", last_name)
        with st.spinner(f"正在抓取 {fn} {ln} 的 MLB Statcast 資料..."):
            pa_df, player_id, total_games = load_player(fn, ln, selected_year)

        if pa_df is not None:
            st.session_state["pa_df"] = pa_df
            st.session_state["player_name"] = f"{fn} {ln}"
            st.session_state["player_id"] = player_id
            st.session_state["total_games"] = total_games
            st.session_state["year"] = selected_year

    if "pa_df" not in st.session_state:
        st.info("👈 請於左側輸入球員姓名並點擊「載入球員數據」開始使用（或試試 🎲 隨機載入明星球員）。")
        return

    # ---------------- 主分頁：單人分析 vs 雙人 PK 對比（功能 #3） ----------------
    tab_solo, tab_pk = st.tabs(["📊 球員分析主板", "🥊 雙球員 PK 對比模式"])

    with tab_solo:
        pa_df = st.session_state["pa_df"]
        player_name = st.session_state["player_name"]
        player_id = st.session_state["player_id"]
        total_games = st.session_state["total_games"]
        year = st.session_state["year"]

        st.subheader("🎛️ 微觀時間軸、左右投拆分與球種過濾控制器")
        stats, fig_hz, fig_spray, game_range, scale_to_162, headshot_url = render_player_panel(
            pa_df, player_name, player_id, total_games, year, key_prefix="solo",
        )

        st.markdown("### 📊 戰績數據指標")
        if scale_to_162:
            st.caption(f"⚠️ 以下數據為【第 {game_range[0]}~{game_range[1]} 場】外推至 162 場的模擬預測值")
        else:
            st.caption(f"顯示【第 {game_range[0]}~{game_range[1]} 場】實際數據")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("打擊率 AVG", f"{stats['AVG']:.3f}")
        m2.metric("上壘率 OBP", f"{stats['OBP']:.3f}")
        m3.metric("長打率 SLG", f"{stats['SLG']:.3f}")
        m4.metric("攻擊指數 OPS", f"{stats['OPS']:.3f}")
        m5.metric("純長打率 ISO", f"{stats['ISO']:.3f}")
        ev_display = f"{stats['Avg_EV']:.1f} mph" if not np.isnan(stats['Avg_EV']) else "N/A"
        m6.metric("平均初速 Avg EV", ev_display)

        n1, n2, n3, n4, n5, n6 = st.columns(6)
        n1.metric("一壘安打 1B", stats["1B"])
        n2.metric("二壘安打 2B", stats["2B"])
        n3.metric("三壘安打 3B", stats["3B"])
        n4.metric("全壘打 HR", stats["HR"])
        n5.metric("總安打 H", stats["H"])
        n6.metric("打點 RBI", stats["RBI"])

        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("盜壘 SB", stats["SB"])
        p2.metric("盜壘失敗 CS", stats["CS"])
        p3.metric("保送 BB", stats["BB"])
        p4.metric("三振 SO", stats["SO"])
        # [Cyber-Ops Rev] 極端數據高亮 (Alert Mode)：最高擊球初速若達 115mph+
        # 巨砲等級，切換為極光霓虹粉紅樣式，讓球迷一眼看出這是破紀錄等級的數據。
        max_ev_val = stats["Max_EV"]
        max_ev_display = f"{max_ev_val:.1f} mph" if not np.isnan(max_ev_val) else "N/A"
        max_ev_alert = (not np.isnan(max_ev_val)) and max_ev_val >= EXIT_VELO_ALERT_MPH
        render_cyber_stat_tile(p5, "最高初速 MAX EV", max_ev_display, is_alert=max_ev_alert)
        la_display = f"{stats['Avg_LA']:.1f}°" if not np.isnan(stats['Avg_LA']) else "N/A"
        p6.metric("平均仰角 Avg LA", la_display)

        # [Cyber-Ops Rev] 面對火球高亮：本區間面對過最快的一顆球（release_speed），
        # 達 100mph+ 頂級火球等級時同樣切換為警示粉紅樣式。
        max_pitch_velo = stats.get("Max_Pitch_Velo_Faced", np.nan)
        if not np.isnan(max_pitch_velo):
            fb1, _, _ = st.columns(3)
            fb_display = f"{max_pitch_velo:.1f} mph"
            fb_alert = max_pitch_velo >= PITCH_VELO_ALERT_MPH
            render_cyber_stat_tile(fb1, "本區間面對最快球速", fb_display, is_alert=fb_alert)

        if not np.isnan(stats.get("Max_HR_Dist", np.nan)):
            st.caption(f"🏟️ 全壘打距離：最遠 {stats['Max_HR_Dist']:.0f} ft　|　平均 {stats['Avg_HR_Dist']:.0f} ft")

        # [Enh 2] 進階期望數據 (xBA/xSLG/Barrel%/Hard Hit%) 指標列
        st.markdown("##### 📈 進階期望數據 (Statcast Expected Stats)")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("估計打擊率 xBA", f"{stats['xBA']:.3f}" if pd.notna(stats["xBA"]) else "N/A")
        q2.metric("估計長打率 xSLG", f"{stats['xSLG']:.3f}" if pd.notna(stats["xSLG"]) else "N/A")
        q3.metric("出色擊球率 Barrel%", f"{stats['Barrel_Pct'] * 100:.1f}%" if pd.notna(stats["Barrel_Pct"]) else "N/A")
        q4.metric("強擊球率 Hard Hit%", f"{stats['HardHit_Pct'] * 100:.1f}%" if pd.notna(stats["HardHit_Pct"]) else "N/A")

        # 擊球型態分佈 (GB% / LD% / FB% / PU%)
        st.markdown("##### 🎯 擊球型態分佈 (Batted Ball Type)")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("滾地球 GB%", f"{stats['GB_Pct'] * 100:.1f}%" if pd.notna(stats.get("GB_Pct")) else "N/A")
        r2.metric("平飛球 LD%", f"{stats['LD_Pct'] * 100:.1f}%" if pd.notna(stats.get("LD_Pct")) else "N/A")
        r3.metric("飛球 FB%", f"{stats['FB_Pct'] * 100:.1f}%" if pd.notna(stats.get("FB_Pct")) else "N/A")
        r4.metric("內野高飛 PU%", f"{stats['PU_Pct'] * 100:.1f}%" if pd.notna(stats.get("PU_Pct")) else "N/A")

        # 全賽季手感起伏圖 (Rolling OPS 走勢圖)：一律以整季 pa_df 計算，不受微觀區間篩選影響
        st.markdown("---")
        st.markdown("##### 📈 全賽季手感起伏圖 (15-Game Rolling OPS)")
        fig_rolling = render_rolling_ops_chart(pa_df, player_name, window=15)
        if fig_rolling is None:
            st.warning("⚠️ 該球員本賽季無足夠的例行賽擊球數據，暫無法繪製滾動 OPS 走勢圖。")
        else:
            st.plotly_chart(fig_rolling, use_container_width=True, key="solo_rolling_ops")

        st.markdown("---")
        st.subheader("📥 匯出個人專屬 Scouting Report 簡報卡")
        if st.button("🖼️ 生成高清 Scouting Report 圖片 (PNG)", key="solo_export"):
            st.caption("⏳ 圖表合成中，約需 3~5 秒...（相同區間/外推設定重複產生時會直接讀取快取，秒開）")
            with st.spinner("正在合成報告卡片..."):
                card_buf = generate_scouting_card_cached(
                    player_name=player_name, year=year, stats=stats,
                    is_scaled=scale_to_162, game_range=game_range,
                    headshot_url=headshot_url, hotzone_fig=fig_hz, spray_fig=fig_spray,
                )
            st.image(card_buf, caption="報告卡預覽", width=500)
            st.download_button(
                label="📩 點擊下載報告圖檔 (PNG)", data=card_buf,
                file_name=f"{player_name.replace(' ', '_')}_scouting_report.png",
                mime="image/png", key="solo_download",
            )

        st.markdown("##### 📱 IG 限時動態專用圖卡（9:16 直式 1080x1920）")
        st.caption("精簡排版、放大球員頭像、三圍數據與落點圖，並印上 boarcast 浮水印，免裁切一鍵發布限動。")
        if st.button("📱 生成 IG 限動直式圖卡 (PNG)", key="solo_export_ig"):
            st.caption("⏳ 圖表合成中，約需 3~5 秒...")
            with st.spinner("正在合成 IG 限動圖卡..."):
                ig_buf = generate_ig_story_card_cached(
                    player_name=player_name, year=year, stats=stats,
                    is_scaled=scale_to_162, game_range=game_range,
                    headshot_url=headshot_url, spray_fig=fig_spray,
                )
            st.image(ig_buf, caption="IG 限動圖卡預覽（9:16）", width=320)
            st.download_button(
                label="📩 下載 IG 限動圖卡 (PNG)", data=ig_buf,
                file_name=f"{player_name.replace(' ', '_')}_IG_story.png",
                mime="image/png", key="solo_download_ig",
            )

    with tab_pk:
        st.subheader("🥊 雙球員數據 PK / 對比模式")
        st.caption("輸入兩位球員姓名並分別載入，即可並排比較數據面板、熱區圖與落點圖，並可生成雙人對決 PNG 圖卡。")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🔵 球員 A")
            a_fn = st.text_input("名 (First Name)", value="Shohei", key="pk_a_fn")
            a_ln = st.text_input("姓 (Last Name)", value="Ohtani", key="pk_a_ln")
            a_year = st.selectbox("賽季", get_selectable_years(), index=0, key="pk_a_year")
            if st.button("📥 載入球員 A", key="pk_a_load"):
                # [Fix 3] 球員 A 的抓取獨立包在自己的 try-except 中，
                # 任何未預期的例外都只影響這一側，不會波及球員 B。
                try:
                    with st.spinner(f"正在抓取 {a_fn} {a_ln} 的資料..."):
                        pa_df_a, pid_a, tg_a = load_player(a_fn, a_ln, a_year)
                    if pa_df_a is not None:
                        st.session_state["pk_a"] = (pa_df_a, f"{a_fn} {a_ln}", pid_a, tg_a, a_year)
                except Exception as e:
                    st.error(f"⚠️ 載入球員 A（{a_fn} {a_ln}）時發生未預期錯誤：{e}")

        with col_b:
            st.markdown("#### 🔴 球員 B")
            b_fn = st.text_input("名 (First Name)", value="Aaron", key="pk_b_fn")
            b_ln = st.text_input("姓 (Last Name)", value="Judge", key="pk_b_ln")
            b_year = st.selectbox("賽季", get_selectable_years(), index=0, key="pk_b_year")
            if st.button("📥 載入球員 B", key="pk_b_load"):
                # [Fix 3] 球員 B 的抓取獨立包在自己的 try-except 中，
                # 任何未預期的例外都只影響這一側，不會波及球員 A。
                try:
                    with st.spinner(f"正在抓取 {b_fn} {b_ln} 的資料..."):
                        pa_df_b, pid_b, tg_b = load_player(b_fn, b_ln, b_year)
                    if pa_df_b is not None:
                        st.session_state["pk_b"] = (pa_df_b, f"{b_fn} {b_ln}", pid_b, tg_b, b_year)
                except Exception as e:
                    st.error(f"⚠️ 載入球員 B（{b_fn} {b_ln}）時發生未預期錯誤：{e}")

        if "pk_a" in st.session_state or "pk_b" in st.session_state:
            st.markdown("---")
            col_panel_a, col_panel_b = st.columns(2)

            # [Fix 3] 兩端獨立防護：球員 A、B 的面板渲染各自包在獨立的
            # try-except 內。任一邊資料異常或渲染失敗，只在該側顯示警告，
            # 絕不讓例外往外擴散拖垮整個 PK 頁面，另一側仍可正常顯示。
            result_a = None
            with col_panel_a:
                if "pk_a" in st.session_state:
                    try:
                        pa_df_a, name_a, pid_a, tg_a, year_a = st.session_state["pk_a"]
                        if pa_df_a is None or pa_df_a.empty:
                            st.warning(f"⚠️ 球員 A（{name_a}）於選定賽季無有效例行賽數據，暫無法顯示面板。")
                        else:
                            stats_a, hz_a, spray_a, range_a, scale_a, headshot_a = render_player_panel(
                                pa_df_a, name_a, pid_a, tg_a, year_a, key_prefix="pka", width_scale=0.7,
                            )
                            st.markdown(f"**AVG/OBP/SLG/OPS**　{stats_a['AVG']:.3f} / {stats_a['OBP']:.3f} / "
                                        f"{stats_a['SLG']:.3f} / {stats_a['OPS']:.3f}　|　HR {stats_a['HR']}　RBI {stats_a['RBI']}")
                            xba_a = f"{stats_a['xBA']:.3f}" if pd.notna(stats_a["xBA"]) else "N/A"
                            brl_a = f"{stats_a['Barrel_Pct'] * 100:.1f}%" if pd.notna(stats_a["Barrel_Pct"]) else "N/A"
                            hh_a = f"{stats_a['HardHit_Pct'] * 100:.1f}%" if pd.notna(stats_a["HardHit_Pct"]) else "N/A"
                            st.caption(f"📈 xBA {xba_a}　|　Barrel% {brl_a}　|　Hard Hit% {hh_a}")
                            result_a = (pa_df_a, name_a, year_a, stats_a, hz_a, spray_a, range_a, scale_a, headshot_a)
                    except Exception as e:
                        st.warning(f"⚠️ 球員 A 面板渲染時發生問題，已略過此側顯示：{e}")
                else:
                    st.info("👆 尚未載入球員 A 的資料。")

            result_b = None
            with col_panel_b:
                if "pk_b" in st.session_state:
                    try:
                        pa_df_b, name_b, pid_b, tg_b, year_b = st.session_state["pk_b"]
                        if pa_df_b is None or pa_df_b.empty:
                            st.warning(f"⚠️ 球員 B（{name_b}）於選定賽季無有效例行賽數據，暫無法顯示面板。")
                        else:
                            stats_b, hz_b, spray_b, range_b, scale_b, headshot_b = render_player_panel(
                                pa_df_b, name_b, pid_b, tg_b, year_b, key_prefix="pkb", width_scale=0.7,
                            )
                            st.markdown(f"**AVG/OBP/SLG/OPS**　{stats_b['AVG']:.3f} / {stats_b['OBP']:.3f} / "
                                        f"{stats_b['SLG']:.3f} / {stats_b['OPS']:.3f}　|　HR {stats_b['HR']}　RBI {stats_b['RBI']}")
                            xba_b = f"{stats_b['xBA']:.3f}" if pd.notna(stats_b["xBA"]) else "N/A"
                            brl_b = f"{stats_b['Barrel_Pct'] * 100:.1f}%" if pd.notna(stats_b["Barrel_Pct"]) else "N/A"
                            hh_b = f"{stats_b['HardHit_Pct'] * 100:.1f}%" if pd.notna(stats_b["HardHit_Pct"]) else "N/A"
                            st.caption(f"📈 xBA {xba_b}　|　Barrel% {brl_b}　|　Hard Hit% {hh_b}")
                            result_b = (pa_df_b, name_b, year_b, stats_b, hz_b, spray_b, range_b, scale_b, headshot_b)
                    except Exception as e:
                        st.warning(f"⚠️ 球員 B 面板渲染時發生問題，已略過此側顯示：{e}")
                else:
                    st.info("👆 尚未載入球員 B 的資料。")

            # [Fix 3] PK 圖表安全疊加：只有在球員 A 與球員 B 兩者皆取得有效的
            # 例行賽數據（result_a 與 result_b 皆非 None）時，才進行雙線滾動
            # OPS 疊加渲染，以及合成雙人 PK 對決圖卡；只要有一方缺資料，一律
            # 降級為「僅顯示已成功載入那一側」，不觸發任何疊加或合成流程。
            if result_a is not None and result_b is not None:
                pa_df_a, name_a, year_a, stats_a, hz_a, spray_a, range_a, scale_a, headshot_a = result_a
                pa_df_b, name_b, year_b, stats_b, hz_b, spray_b, range_b, scale_b, headshot_b = result_b

                st.markdown("---")
                st.markdown("##### 📈 雙人滾動 OPS 對比（近 15 場）")
                fig_pk_rolling = render_pk_rolling_ops_chart(pa_df_a, name_a, pa_df_b, name_b, window=15)
                if fig_pk_rolling is None:
                    st.warning("⚠️ 雙方例行賽資料不足，暫無法繪製滾動 OPS 對比圖。")
                else:
                    st.plotly_chart(fig_pk_rolling, use_container_width=True, key="pk_rolling_ops")

                st.markdown("---")
                if st.button("🖼️ 生成雙人 PK 對決圖卡 (PNG)", key="pk_export"):
                    st.caption("⏳ 圖表合成中，約需 5~10 秒（雙人需合成 2 張報告卡，相同參數重複產生時會直接讀取快取，秒開）...")
                    with st.spinner("正在合成雙人 PK 對決圖卡..."):
                        buf_a = generate_scouting_card_cached(
                            player_name=name_a, year=year_a, stats=stats_a, is_scaled=scale_a,
                            game_range=range_a, headshot_url=headshot_a, hotzone_fig=hz_a, spray_fig=spray_a,
                        )
                        buf_b = generate_scouting_card_cached(
                            player_name=name_b, year=year_b, stats=stats_b, is_scaled=scale_b,
                            game_range=range_b, headshot_url=headshot_b, hotzone_fig=hz_b, spray_fig=spray_b,
                        )
                        pk_buf = generate_pk_card_cached(
                            buf_a, buf_b, name_a, name_b,
                            fingerprint_a=(tuple(range_a), scale_a, _stats_fingerprint(stats_a)),
                            fingerprint_b=(tuple(range_b), scale_b, _stats_fingerprint(stats_b)),
                        )
                    st.image(pk_buf, caption="PK 對決圖卡預覽", width=700)
                    st.download_button(
                        label="📩 下載 PK 對決圖卡 (PNG)", data=pk_buf,
                        file_name=f"{name_a.replace(' ', '_')}_vs_{name_b.replace(' ', '_')}_PK.png",
                        mime="image/png", key="pk_download",
                    )
            elif result_a is not None or result_b is not None:
                st.info("👆 只有一位球員成功載入完整資料；請載入另一位球員，即可解鎖雙線滾動 OPS 對比圖與 PK 對決圖卡。")
        else:
            st.info("👆 請先分別載入球員 A 與球員 B 的數據。")


if __name__ == "__main__":
    main()
