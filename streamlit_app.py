"""Free Streamlit Cloud version of the flight assistant."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st


FLIGHTS = """origin,destination,departure_date,return_date,airline,airline_type,is_direct,transfer_city,total_price,flight_duration_hours,baggage_included,source
TPE,KIX,2026-07-01,2026-07-06,Tigerair Taiwan,LCC,True,,10800,2.7,False,Skyscanner
TPE,KIX,2026-07-03,2026-07-08,Peach,LCC,True,,11200,2.8,False,Google Flights
TPE,KIX,2026-07-05,2026-07-10,EVA Air,FSC,True,,14800,2.6,True,Airline Website
TPE,KIX,2026-07-08,2026-07-13,Tigerair Taiwan,LCC,True,,10600,2.7,False,Skyscanner
TPE,KIX,2026-07-12,2026-07-18,HK Express,LCC,False,HKG,9800,6.1,False,OTA
TPE,FUK,2026-07-02,2026-07-07,Starlux,FSC,True,,9600,2.3,True,Airline Website
TPE,FUK,2026-07-04,2026-07-09,Tigerair Taiwan,LCC,True,,8800,2.4,False,Skyscanner
TPE,FUK,2026-07-09,2026-07-14,HK Express,LCC,False,HKG,9200,5.8,False,OTA
TPE,FUK,2026-07-18,2026-07-23,Peach,LCC,True,,10100,2.4,False,Skyscanner
TPE,OKA,2026-07-01,2026-07-05,Peach,LCC,True,,8200,1.6,False,Skyscanner
TPE,OKA,2026-07-04,2026-07-08,Tigerair Taiwan,LCC,True,,7900,1.7,False,OTA
TPE,OKA,2026-07-11,2026-07-16,EVA Air,FSC,True,,12400,1.6,True,Google Flights
TPE,NRT,2026-07-02,2026-07-08,Jetstar Japan,LCC,True,,9900,3.2,False,Skyscanner
TPE,NRT,2026-07-05,2026-07-11,EVA Air,FSC,True,,15800,3.1,True,Airline Website
TPE,NRT,2026-07-10,2026-07-15,Peach,LCC,True,,10500,3.2,False,OTA
TPE,NRT,2026-07-19,2026-07-24,HK Express,LCC,False,HKG,9700,6.6,False,OTA
TPE,ICN,2026-07-01,2026-07-05,T’way Air,LCC,True,,7600,2.5,False,Skyscanner
TPE,ICN,2026-07-03,2026-07-08,EVA Air,FSC,True,,11200,2.4,True,Airline Website
TPE,ICN,2026-07-09,2026-07-14,T’way Air,LCC,True,,8100,2.5,False,OTA
TPE,ICN,2026-07-13,2026-07-18,HK Express,LCC,False,HKG,7300,5.9,False,OTA
"""

HISTORY = """origin,destination,month,monthly_average_price,annual_average_price,historical_low_price,historical_high_price,recommended_target_price
TPE,KIX,2026-07,12800,12200,8800,19800,10800
TPE,FUK,2026-07,11600,10900,7600,17200,9800
TPE,OKA,2026-07,9800,9500,6500,15000,8500
TPE,NRT,2026-07,13800,13200,9200,21000,11500
TPE,ICN,2026-07,10200,9600,6200,15800,8300
"""

AIRPORT_OPTIONS = {
    "台北桃園 TPE": "TPE",
    "台北松山 TSA": "TSA",
    "高雄 KHH": "KHH",
    "大阪關西 KIX": "KIX",
    "福岡 FUK": "FUK",
    "沖繩那霸 OKA": "OKA",
    "名古屋中部 NGO": "NGO",
    "東京成田 NRT": "NRT",
    "東京羽田 HND": "HND",
    "札幌新千歲 CTS": "CTS",
    "仙台 SDJ": "SDJ",
    "廣島 HIJ": "HIJ",
    "鹿兒島 KOJ": "KOJ",
    "首爾仁川 ICN": "ICN",
    "釜山 PUS": "PUS",
    "香港 HKG": "HKG",
    "澳門 MFM": "MFM",
    "上海浦東 PVG": "PVG",
    "上海虹橋 SHA": "SHA",
    "北京首都 PEK": "PEK",
    "北京大興 PKX": "PKX",
    "曼谷 BKK": "BKK",
    "清邁 CNX": "CNX",
    "新加坡 SIN": "SIN",
    "吉隆坡 KUL": "KUL",
    "馬尼拉 MNL": "MNL",
    "胡志明市 SGN": "SGN",
    "河內 HAN": "HAN",
    "峴港 DAD": "DAD",
    "雅加達 CGK": "CGK",
    "峇里島 DPS": "DPS",
    "雪梨 SYD": "SYD",
    "墨爾本 MEL": "MEL",
    "洛杉磯 LAX": "LAX",
    "舊金山 SFO": "SFO",
    "紐約 JFK": "JFK",
    "倫敦希斯洛 LHR": "LHR",
    "巴黎戴高樂 CDG": "CDG",
    "阿姆斯特丹 AMS": "AMS",
    "法蘭克福 FRA": "FRA",
}

AIRPORT_LABEL_BY_CODE = {code: label for label, code in AIRPORT_OPTIONS.items()}


def airport_label(code: str) -> str:
    """Return a readable airport label for a code."""
    return AIRPORT_LABEL_BY_CODE.get(code, code)


def ntd(value):
    """Format a number as NT dollars."""
    return "-" if pd.isna(value) else f"NT${float(value):,.0f}"


@st.cache_data
def load_data():
    """Load sample data bundled in this cloud demo."""
    flights = pd.read_csv(StringIO(FLIGHTS))
    history = pd.read_csv(StringIO(HISTORY))
    flights["departure_date"] = pd.to_datetime(flights["departure_date"])
    flights["return_date"] = pd.to_datetime(flights["return_date"])
    return flights, history


def search(df, origin, destination, month, budget, direct_only, include_lcc):
    """Search and score flights."""
    out = df[
        df["origin"].eq(origin)
        & df["destination"].eq(destination)
        & df["departure_date"].dt.strftime("%Y-%m").eq(month)
    ].copy()
    if direct_only:
        out = out[out["is_direct"]]
    if not include_lcc:
        out = out[out["airline_type"] != "LCC"]
    if out.empty:
        return out
    price = out["total_price"].astype(float)
    duration = out["flight_duration_hours"].astype(float)
    price_score = 45 if price.max() == price.min() else 45 * (1 - (price - price.min()) / (price.max() - price.min()))
    duration_score = 20 if duration.max() == duration.min() else 20 * (1 - (duration - duration.min()) / (duration.max() - duration.min()))
    out["price_status"] = np.where(out["total_price"] <= budget, "Within Budget", "Over Budget")
    out["value_score"] = (price_score + duration_score + out["is_direct"].map({True: 20, False: 8}) + out["baggage_included"].map({True: 15, False: 5})).round(1)
    return out.sort_values(["total_price", "value_score"], ascending=[True, False])


def ignav_key() -> str:
    """Read Ignav API key from Streamlit secrets."""
    try:
        return str(st.secrets.get("IGNAV_API_KEY", "")).strip()
    except Exception:
        return ""


def fetch_ignav_round_trip(api_key: str, origin: str, destination: str, departure_date, return_date, adults: int, direct_only: bool):
    """Fetch live round-trip fares from Ignav and return a dataframe plus a status message."""
    payload = {
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date.strftime("%Y-%m-%d"),
        "return_date": return_date.strftime("%Y-%m-%d"),
        "adults": adults,
        "cabin_class": "economy",
        "market": "TW",
    }
    if direct_only:
        payload["max_stops"] = 0

    try:
        response = requests.post(
            "https://ignav.com/api/fares/round-trip",
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return pd.DataFrame(), f"Ignav API 查詢失敗：{exc}"
    except ValueError:
        return pd.DataFrame(), "Ignav API 回傳格式不是 JSON。"

    rows = []
    for item in data.get("itineraries", []):
        outbound = item.get("outbound", {}) or {}
        inbound = item.get("inbound", {}) or {}
        outbound_segments = outbound.get("segments", []) or []
        inbound_segments = inbound.get("segments", []) or []
        price = item.get("price", {}) or {}
        amount = price.get("amount")
        currency = price.get("currency", "TWD")
        is_direct = len(outbound_segments) <= 1 and len(inbound_segments) <= 1
        transfer_points = []
        for segment in outbound_segments[:-1] + inbound_segments[:-1]:
            airport = segment.get("arrival_airport")
            if airport:
                transfer_points.append(airport)

        rows.append(
            {
                "origin": origin,
                "destination": destination,
                "departure_date": pd.to_datetime(payload["departure_date"]),
                "return_date": pd.to_datetime(payload["return_date"]),
                "airline": outbound.get("carrier") or "Ignav itinerary",
                "airline_type": "API",
                "is_direct": is_direct,
                "transfer_city": ", ".join(transfer_points),
                "total_price": pd.to_numeric(amount, errors="coerce"),
                "price_currency": currency,
                "flight_duration_hours": round((float(outbound.get("duration_minutes", 0)) + float(inbound.get("duration_minutes", 0))) / 60, 1),
                "baggage_included": False,
                "source": "Ignav API",
                "ignav_id": item.get("ignav_id", ""),
            }
        )

    if not rows:
        return pd.DataFrame(), "Ignav API 沒有回傳符合條件的航班，已保留 CSV 範例資料可比較。"

    out = pd.DataFrame(rows).dropna(subset=["total_price"]).sort_values("total_price")
    out["price_status"] = "API result"
    out["value_score"] = 80
    return out, f"Ignav API 已回傳 {len(out)} 筆票價。"


def display(df):
    """Show a formatted dataframe."""
    if df.empty:
        st.info("目前沒有符合條件的資料。")
        return
    view = df.copy()
    if "origin" in view:
        view["origin"] = view["origin"].map(airport_label)
    if "destination" in view:
        view["destination"] = view["destination"].map(airport_label)
    for col in ["departure_date", "return_date"]:
        if col in view:
            view[col] = view[col].dt.strftime("%Y-%m-%d")
    for col in ["total_price", "lowest_price"]:
        if col in view:
            if "price_currency" in view and col == "total_price":
                view[col] = view.apply(lambda row: ntd(row[col]) if row.get("price_currency", "TWD") == "TWD" else f"{float(row[col]):,.0f} {row.get('price_currency')}", axis=1)
            else:
                view[col] = view[col].apply(ntd)
    st.dataframe(view, use_container_width=True, hide_index=True)


def airline_price_summary(df):
    """Build airline-level price statistics without summing ticket prices."""
    if df.empty:
        return pd.DataFrame(columns=["airline", "lowest_price", "average_price", "fare_count"])
    return (
        df.groupby("airline", as_index=False)
        .agg(
            lowest_price=("total_price", "min"),
            average_price=("total_price", "mean"),
            fare_count=("total_price", "count"),
        )
        .sort_values(["lowest_price", "average_price"])
        .reset_index(drop=True)
    )


def strategy(history, origin, destination, month, current_price):
    """Analyze whether to buy, wait, or monitor."""
    row = history[(history["origin"].eq(origin)) & (history["destination"].eq(destination)) & (history["month"].eq(month))]
    if row.empty:
        return "Monitor", "目前沒有足夠歷史資料，建議先持續追蹤。", {}
    item = row.iloc[0]
    if current_price <= item["recommended_target_price"]:
        return "Buy", "目前價格已低於建議入手價，若航班時間與行李條件符合需求，建議可以下手。", item
    if current_price <= item["monthly_average_price"] * 0.9:
        return "Consider", "目前價格比該月份平均低至少 10%，屬於值得考慮的區間。", item
    if current_price > item["monthly_average_price"] * 1.15:
        return "Wait", "目前價格高於該月份平均 15% 以上，建議等待或改看鄰近日。", item
    return "Monitor", "目前價格接近歷史區間中段，建議設定降價提醒並觀察。", item


st.set_page_config(page_title="旅遊機票助手", layout="wide")
st.title("旅遊機票助手")
st.caption("全免費雲端版：使用範例 CSV 資料，不串航空 API，不爬蟲。")

flights, history = load_data()
st.sidebar.header("搜尋條件")
origin_label = st.sidebar.selectbox("出發地", list(AIRPORT_OPTIONS.keys()), index=0)
destination_query = st.sidebar.text_input("搜尋目的地（城市/機場/代碼）", "")
destination_candidates = [
    label
    for label in AIRPORT_OPTIONS
    if not destination_query
    or destination_query.lower() in label.lower()
    or destination_query.upper() == AIRPORT_OPTIONS[label]
]
if not destination_candidates:
    destination_candidates = ["找不到內建結果，請在下方輸入 IATA 三碼"]
destination_label = st.sidebar.selectbox(
    "目的地",
    destination_candidates,
    index=0 if destination_query else destination_candidates.index("大阪關西 KIX"),
)
origin = AIRPORT_OPTIONS[origin_label]
custom_destination = st.sidebar.text_input("自訂目的地 IATA 三碼（找不到時使用）", "").strip().upper()
if custom_destination:
    if len(custom_destination) == 3 and custom_destination.isalpha():
        destination = custom_destination
        destination_label = f"自訂目的地 {custom_destination}"
    else:
        st.sidebar.warning("自訂目的地請輸入 3 個英文字母，例如 NGO、LAX、CDG。")
        destination = AIRPORT_OPTIONS.get(destination_label, "KIX")
elif destination_label in AIRPORT_OPTIONS:
    destination = AIRPORT_OPTIONS[destination_label]
else:
    destination = "KIX"
st.sidebar.caption(f"系統會自動使用機場代碼：{origin} → {destination}")
month = st.sidebar.text_input("出發月份", "2026-07")
budget = st.sidebar.number_input("預算", 1000, 100000, 12000, 500)
min_nights = st.sidebar.number_input("停留天數最小值", 1, 30, 3)
max_nights = st.sidebar.number_input("停留天數最大值", 1, 60, 7)
direct_only = st.sidebar.checkbox("是否只看直飛")
include_lcc = st.sidebar.checkbox("是否包含廉航", True)
st.sidebar.divider()
st.sidebar.subheader("即時票價 API")
api_key = ignav_key()
use_ignav = st.sidebar.checkbox("使用 Ignav 即時查詢", value=bool(api_key))
api_departure_date = st.sidebar.date_input("API 出發日期", pd.to_datetime(f"{month}-01").date())
api_return_date = st.sidebar.date_input("API 回程日期", pd.to_datetime(f"{month}-06").date())
api_adults = st.sidebar.number_input("成人數", 1, 9, 1)

if "ignav_results" not in st.session_state:
    st.session_state.ignav_results = pd.DataFrame()
if "ignav_message" not in st.session_state:
    st.session_state.ignav_message = ""

if use_ignav and st.sidebar.button("查詢 Ignav 即時票價"):
    if not api_key:
        st.session_state.ignav_results = pd.DataFrame()
        st.session_state.ignav_message = "尚未設定 IGNAV_API_KEY，會使用 CSV 範例資料。"
    elif api_return_date < api_departure_date:
        st.session_state.ignav_results = pd.DataFrame()
        st.session_state.ignav_message = "回程日期不可早於出發日期。"
    else:
        st.session_state.ignav_results, st.session_state.ignav_message = fetch_ignav_round_trip(
            api_key, origin, destination, api_departure_date, api_return_date, int(api_adults), direct_only
        )

results = search(flights, origin, destination, month, budget, direct_only, include_lcc)
if use_ignav and not st.session_state.ignav_results.empty:
    results = st.session_state.ignav_results.copy()
tabs = st.tabs(["航班搜尋", "票價分析", "日期最佳化", "降價提醒", "昂貴日期熱圖", "訂票策略庫"])

with tabs[0]:
    st.subheader("隱藏航班搜尋器")
    st.caption(f"目前查詢：{airport_label(origin)} → {airport_label(destination)}")
    if use_ignav:
        st.info(st.session_state.ignav_message or "按側邊欄「查詢 Ignav 即時票價」後，會使用 API 結果；尚未查詢前會顯示 CSV 範例資料。")
    c1, c2, c3 = st.columns(3)
    c1.metric("符合航班", len(results))
    c2.metric("最低票價", ntd(results["total_price"].min()) if not results.empty else "-")
    c3.metric("預算內", int((results["total_price"] <= budget).sum()) if not results.empty else 0)
    display(results)
    if not results.empty:
        st.markdown("**各航空最低票價比較**")
        airline_summary = airline_price_summary(results)
        st.bar_chart(airline_summary.set_index("airline")["lowest_price"])
        st.caption("圖表使用每家航空的最低票價，不會把同一家航空的多筆票價加總。")
        summary_view = airline_summary.rename(
            columns={
                "airline": "航空公司",
                "lowest_price": "最低票價",
                "average_price": "平均票價",
                "fare_count": "筆數",
            }
        )
        summary_view["最低票價"] = summary_view["最低票價"].apply(ntd)
        summary_view["平均票價"] = summary_view["平均票價"].apply(ntd)
        st.dataframe(summary_view, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("票價策略分析")
    if results.empty:
        st.info("需要先有符合條件的航班才能分析。")
    else:
        decision, reason, item = strategy(history, origin, destination, month, float(results["total_price"].min()))
        c1, c2, c3 = st.columns(3)
        c1.metric("目前最低價", ntd(results["total_price"].min()))
        c2.metric("建議入手價", ntd(item.get("recommended_target_price")))
        c3.metric("策略判斷", decision)
        st.write(reason)

with tabs[2]:
    st.subheader("找出最適合的出發與回程日期")
    opt = results.copy()
    if not opt.empty:
        opt["nights"] = (opt["return_date"] - opt["departure_date"]).dt.days
        opt = opt[opt["nights"].between(min_nights, max_nights)].head(10)
        opt["recommendation_reason"] = opt.apply(lambda r: f"{'低於預算' if r.total_price <= budget else '高於預算'}，{'直飛' if r.is_direct else '需轉機'}，停留 {int(r.nights)} 晚。", axis=1)
    display(opt)

with tabs[3]:
    st.subheader("觸發降價通知")
    alerts = flights[(flights["total_price"] <= budget) & (flights["departure_date"].dt.strftime("%Y-%m").eq(month))].sort_values("total_price").head(5)
    display(alerts)
    if alerts.empty:
        st.text_area("通知文字預覽", "目前沒有符合條件的降價通知。", height=160)
    else:
        r = alerts.iloc[0]
        st.text_area("通知文字預覽", f"【機票降價通知】\n{airport_label(r.origin)} → {airport_label(r.destination)}\n日期：{r.departure_date:%Y-%m-%d} ～ {r.return_date:%Y-%m-%d}\n航空：{r.airline}\n價格：{ntd(r.total_price)}\n是否直飛：{'是' if r.is_direct else '否'}\n來源：{r.source}", height=220)

with tabs[4]:
    st.subheader("避開昂貴日期")
    heat = results.groupby(results["departure_date"].dt.strftime("%Y-%m-%d"))["total_price"].min().reset_index() if not results.empty else pd.DataFrame()
    if not heat.empty:
        heat.columns = ["date", "lowest_price"]
        heat["price_level"] = np.select([heat["lowest_price"] < budget * 0.85, heat["lowest_price"] <= budget], ["Cheap", "Normal"], default="Expensive")
        heat["marker"] = np.where(heat["lowest_price"].eq(heat["lowest_price"].min()), "lowest", "blank")
    display(heat)

with tabs[5]:
    st.subheader("訂票策略庫")
    for title, body in [
        ("比較航空官網與 OTA", "同一班航班至少比較航空官網、Skyscanner/Google Flights、常用 OTA。"),
        ("檢查行李、座位、付款手續費", "廉航低價常不含托運行李、選位與刷卡手續費，請把必要費用加回總價。"),
        ("留意廉航回程時間", "回程若是清晨或深夜，可能增加住宿、交通或請假成本。"),
        ("確認轉機是否同一張票", "若 OTA 組合不同航空分段票，延誤時不一定受到保障。"),
        ("低價出現時先截圖記錄查詢時間", "看到低於建議入手價時，先截圖包含航班、票價、來源與查詢時間。"),
    ]:
        st.markdown(f"**{title}**")
        st.write(body)
