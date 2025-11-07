#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# -----------------------------
# 상수: kWh당 탄소배출계수 (kg CO2e/kWh)
# -----------------------------
EMISSION_FACTOR_KG_PER_KWH = 0.495  # 국내 전력 1kWh 생산시 약 0.495 kgCO2e

# =========================
# 1) 한글 폰트 (NanumGothic.ttf 를 앱 폴더에 넣어두면 자동 사용)
# =========================
def set_korean_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "NanumGothic.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()


# =========================
# 2) 유틸/지표 계산 함수
# =========================
def price_with_cagr(base_price, base_year, year, cagr):
    """기준연도 대비 연복리(cagr) 상승 단가"""
    return base_price * (1 + cagr) ** (year - base_year)

def npv(rate: float, cashflows: list[float]) -> float:
    """NPV: 첫 해(설치연도) 현금흐름이 cashflows[0] (t=0) 기준"""
    if rate <= -0.999999:
        return float("nan")
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))

def irr_bisection(cashflows: list[float], lo=-0.99, hi=3.0, tol=1e-7, max_iter=200):
    """
    IRR: 부호변화가 있어야 수렴이 잘 됨. 이분법으로 안정적으로 탐색.
    반환값이 None이면 IRR이 정의되지 않거나 범위 내에서 찾지 못한 경우.
    """
    def f(r):
        try:
            return npv(r, cashflows)
        except Exception:
            return np.nan

    f_lo, f_hi = f(lo), f(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return None  # 근 찾기 곤란(부호변화 없음)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if np.isnan(f_mid):
            return None
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid

def discounted_payback(cashflows: list[float], rate: float):
    """
    할인 회수기간(DPP): 할인 누적현금흐름이 0을 넘는 최초 시점까지의 연수.
    선형 보간으로 부분 연도 반영. 없으면 None.
    """
    disc = []
    cum = 0.0
    for t, cf in enumerate(cashflows):
        val = cf / ((1 + rate) ** t)
        cum += val
        disc.append(cum)

    # 최초로 0 이상이 되는 시점 찾기
    for k, v in enumerate(disc):
        if v >= 0:
            if k == 0:
                return 0.0
            prev = disc[k - 1]
            # k-1년도 말엔 prev<0, k년도 말엔 v>=0
            # 그 사이 어느 시점에 0이 되는지 선형 보간
            frac = 0.0 if v == prev else (-prev) / (v - prev)
            return (k - 1) + max(0.0, min(1.0, frac))
    return None

def won_formatter(x, pos):
    return f"{int(x):,}"


# =========================
# 3) CSV에서 "연도별 PV 연간 발전량(kWh)" 계산 (자동 탐색)
# =========================
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {c.replace(" ", "").lower(): c for c in df.columns}
    for k in candidates:
        if k in norm:
            return norm[k]
    return None

@st.cache_data
def load_pv_kwh_by_year_from_csv() -> dict | None:
    """
    앱 폴더 또는 data/ 폴더에서 jeju.csv를 자동 탐색.
    CSV의 '연도'와 '일사합(MJ/m2)'로부터 연도별 PV 연간 발전량(kWh)을 계산해서 dict[year]=kWh 반환.
    컬럼명이 다르면 후보 리스트를 추가해줘.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "jeju.csv"),
        os.path.join(base_dir, "data", "jeju.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.info("⚠️ CSV(jeju.csv)를 찾지 못했습니다. 기본 PV 연간 발전량 값으로 진행합니다.")
        return None

    # 인코딩 시도
    df = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        st.warning("⚠️ CSV를 읽지 못했습니다. 기본값으로 진행합니다.")
        return None

    # 연도/일사합 컬럼 찾기
    year_col = _find_col(df, ["연도", "year", "일시", "date"])
    ghi_col  = _find_col(df, [
        "일사합(mj/m2)", "일사합(mj/m²)", "일사합", "ghimj", "ghi_mj", "ghi(mj/m2)"
    ])
    if year_col is None or ghi_col is None:
        st.warning("⚠️ CSV에서 연도/일사합 컬럼을 찾지 못했습니다. 기본값으로 진행합니다.")
        return None

    # 연도 해석
    # '일시'가 날짜면 year로 변환, 숫자면 그대로
    try:
        year_series = pd.to_numeric(df[year_col], errors="coerce")
        if year_series.notna().mean() < 0.5:
            # 숫자가 아니라면 날짜 파싱
            year_series = pd.to_datetime(df[year_col], errors="coerce").dt.year
    except Exception:
        year_series = pd.to_datetime(df[year_col], errors="coerce").dt.year

    df["_year_"] = year_series
    df = df.dropna(subset=["_year_"]).copy()
    df["_year_"] = df["_year_"].astype(int)

    # 일사합 수치
    df["_ghi_mj_"] = pd.to_numeric(df[ghi_col], errors="coerce")

    # 만약 CSV가 "연도별 합계 1행씩"이면 groupby가 의미 없지만 안전하게 합계
    yearly_mj_per_m2 = df.groupby("_year_", as_index=True)["_ghi_mj_"].sum()

    # MJ/m² → kWh/m²
    yearly_kwh_per_m2 = yearly_mj_per_m2 * 0.27778

    # --- 설비 면적 & PR(간단 고정값) ---
    PANEL_W, PANEL_H, N_MODULES = 1.46, 0.98, 250
    AREA_M2 = PANEL_W * PANEL_H * N_MODULES  # 357.7 m²
    PR = 0.76  # (예) 0.82*0.98*(1-0.02)*0.97 ≈ 0.76

    # 연도별 PV 연간 발전량(kWh)
    yearly_pv_kwh = (yearly_kwh_per_m2 * AREA_M2 * PR).to_dict()  # {year: kWh}
    return yearly_pv_kwh


# =========================
# 4) 기본 파라미터
# =========================
def make_v2g_model_params():
    return {
        # PV (이 값은 CSV가 있으면 자동으로 '연도별'로 대체됩니다)
        "pv_capacity_kw": 125,            # 정보용
        "pv_annual_kwh": 153_300,         # (fallback) CSV 없을 때 사용할 기본값
        "self_use_ratio": 0.60,           # 자가소비 비율

        # V2G
        "num_v2g_chargers": 6,            # 대수
        "v2g_charger_unit_cost": 25_000_000,  # 대당 CAPEX
        "v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,
        "v2g_operating_days": 300,        # 연 가동일

        # 단가(연복리 상승)
        "tariff_base_year": 2025,
        "pv_base_price": 160,             # 원/kWh
        "v2g_price_gap": 30,              # PV 대비 V2G 프리미엄
        "price_cagr": 0.043,              # 전력단가 상승률

        # O&M
        "om_ratio": 0.015,                # CAPEX 대비 연간 O&M 비율
    }


# =========================
# 5) 현금흐름 빌드 (CSV 연동 반영)
# =========================
def build_yearly_cashflows(install_year: int, current_year: int, p: dict,
                           pv_kwh_by_year: dict | None = None):
    """
    pv_kwh_by_year: {연도:int -> PV 연간 발전량(kWh)}. 없으면 p['pv_annual_kwh'] 고정값 사용.
    """
    years = list(range(install_year, current_year + 1))

    # V2G(연간 고정) 계산
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]

    # CAPEX/O&M
    capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost = capex_total * p["om_ratio"]

    yearly_cash, cumulative = [], []
    pv_revenues, v2g_revenues, om_costs, capex_list = [], [], [], []
    # 탄소/에너지 통계를 위해 연도별 PV/Surplus도 저장
    pv_kwh_year_list, pv_surplus_kwh_year_list = [], []

    # 만약 CSV dict가 있으면 해당 연도 값, 없으면 평균 또는 p 기본값 사용
    default_pv_kwh = p["pv_annual_kwh"]
    dict_mean = None
    if pv_kwh_by_year:
        dict_mean = float(np.mean(list(pv_kwh_by_year.values())))

    cum = 0.0
    for i, year in enumerate(years):
        # 연도별 PV kWh 결정
        if pv_kwh_by_year and year in pv_kwh_by_year:
            pv_kwh = float(pv_kwh_by_year[year])
        elif pv_kwh_by_year and dict_mean is not None:
            pv_kwh = dict_mean  # 해당 연도 자료 없으면 평균 사용
        else:
            pv_kwh = float(default_pv_kwh)

        pv_surplus_kwh = pv_kwh * (1 - p["self_use_ratio"])

        # 단가(연복리)
        pv_price_y = price_with_cagr(p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"])
        v2g_price_y = pv_price_y + p["v2g_price_gap"]

        # 수입
        revenue_pv_y = pv_surplus_kwh * pv_price_y
        revenue_v2g_y = annual_v2g_kwh * v2g_price_y
        annual_revenue_y = revenue_pv_y + revenue_v2g_y

        # 비용
        om_y = annual_om_cost
        capex_y = capex_total if i == 0 else 0

        # 순현금/누적
        cf = annual_revenue_y - om_y - capex_y
        cum += cf

        # 기록
        yearly_cash.append(cf)
        cumulative.append(cum)
        pv_revenues.append(revenue_pv_y)
        v2g_revenues.append(revenue_v2g_y)
        om_costs.append(om_y)
        capex_list.append(capex_y)
        pv_kwh_year_list.append(pv_kwh)
        pv_surplus_kwh_year_list.append(pv_surplus_kwh)

    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        # 에너지/탄소 계산용
        "pv_kwh_year": pv_kwh_year_list,
        "pv_surplus_kwh_year": pv_surplus_kwh_year_list,
        "annual_v2g_kwh": annual_v2g_kwh,   # (연도 고정)
    }


# =========================
# 6) Streamlit App
# =========================
def main():
    st.title("V2G 투자 대비 연도별/누적 현금흐름 (CSV 기반 PV 자동 반영)")

    params = make_v2g_model_params()

    # ----- 사이드바 입력 (CSV 경로는 노출하지 않음) -----
    st.sidebar.header("시뮬레이션 입력")
    install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
    current_year = st.sidebar.number_input("마지막 연도", value=2045, step=1, min_value=install_year)

    params["num_v2g_chargers"] = st.sidebar.number_input("V2G 충전기 대수", value=params["num_v2g_chargers"], step=1, min_value=1)
    params["v2g_daily_discharge_per_charger_kwh"] = st.sidebar.number_input(
        "1대당 일일 방전량 (kWh)", value=params["v2g_daily_discharge_per_charger_kwh"], step=1, min_value=1
    )
    params["v2g_operating_days"] = st.sidebar.number_input(
        "연간 운영일 수", value=params["v2g_operating_days"], step=10, min_value=1, max_value=365
    )
    # ⚠️ pv_annual_kwh 입력칸 제거! (CSV 자동 반영)
    params["self_use_ratio"] = st.sidebar.slider("PV 자가소비 비율", min_value=0.0, max_value=1.0, value=params["self_use_ratio"], step=0.05)
    params["pv_base_price"] = st.sidebar.number_input("PV 기준단가 (원/kWh)", value=params["pv_base_price"], step=5, min_value=0)
    params["price_cagr"] = st.sidebar.number_input("전력단가 연평균 상승률", value=params["price_cagr"], step=0.001, format="%.3f")

    # ★ 재무 지표용 할인율
    discount_rate = st.sidebar.number_input(
        "할인율(연)", value=0.08, min_value=0.0, max_value=0.5, step=0.005, format="%.3f", help="NPV/할인회수기간 계산에 사용"
    )

    # ----- CSV에서 연도별 PV 연간 발전량(kWh) 자동 로드 -----
    pv_kwh_by_year = load_pv_kwh_by_year_from_csv()  # dict | None

    # ----- 계산 -----
    cf = build_yearly_cashflows(install_year, current_year, params, pv_kwh_by_year=pv_kwh_by_year)
    years = cf["years"]
    yearly_cash = cf["yearly_cash"]
    cumulative = cf["cumulative"]

    # 손익분기 연도(회계적): 누적이 0 이상 최초 연도
    break_even_year = None
    for y, cum in zip(years, cumulative):
        if cum >= 0:
            break_even_year = y
            break

    # 탄소절감량 (kgCO2e) — 연도별 합산(연 PV 잉여 + V2G)
    total_clean_kwh = float(np.sum(cf["pv_surplus_kwh_year"])) + len(years) * cf["annual_v2g_kwh"]
    total_co2_kg = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH

    # ----- 재무지표 -----
    npv_val = npv(discount_rate, yearly_cash)
    irr_val = irr_bisection(yearly_cash)  # None 가능
    dpp_val = discounted_payback(yearly_cash, discount_rate)  # None 가능

    # ----- KPI (1행: 재무지표) -----
    k1, k2, k3, sp1 = st.columns([1, 1, 1, 0.4])
    with k1:
        st.markdown('<div style="font-size:0.85rem;color:#666;">NPV</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{npv_val:,.0f} 원</div>', unsafe_allow_html=True)
        st.caption(f"할인율 {discount_rate*100:.1f}%")

    with k2:
        st.markdown('<div style="font-size:0.85rem;color:#666;">IRR</div>', unsafe_allow_html=True)
        irr_txt = f"{irr_val*100:.2f} %" if irr_val is not None else "정의 불가"
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{irr_txt}</div>', unsafe_allow_html=True)

    with k3:
        st.markdown('<div style="font-size:0.85rem;color:#666;">할인 회수기간</div>', unsafe_allow_html=True)
        dpp_txt = f"{dpp_val:.2f} 년" if dpp_val is not None else "미회수"
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{dpp_txt}</div>', unsafe_allow_html=True)

    # ----- KPI (2행: 기존 지표) -----
    r1, r2, r3, sp2 = st.columns([1, 1, 1, 0.4])
    with r1:
        be_text = f"{break_even_year}년" if break_even_year else "아직 미도달"
        st.markdown('<div style="font-size:0.85rem;color:#666;">손익분기 연도</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{be_text}</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div style="font-size:0.85rem;color:#666;">마지막 연도 누적</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{cumulative[-1]:,.0f} 원</div>', unsafe_allow_html=True)

    with r3:
        st.markdown('<div style="font-size:0.85rem;color:#666;">누적 탄소절감량</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:1.3rem;font-weight:600;">{total_co2_kg:,.0f} kgCO₂e</div>', unsafe_allow_html=True)

    # ----- 누적 라인 차트 (matplotlib) -----
    st.subheader("누적 현금흐름")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cumulative, marker="o", linewidth=2.2)
    ax.set_xlabel("연도"); ax.set_ylabel("누적 금액(원)")
    ax.yaxis.set_major_formatter(FuncFormatter(won_formatter))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("V2G + CSV기반 PV 누적 현금흐름")
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(break_even_year, 0, f"손익분기 {break_even_year}", color="green", va="bottom", ha="left")
    st.pyplot(fig)

    # ----- 누적 막대 (연도별 누적) -----
    st.subheader("연도별 순현금흐름 (누적)")
    x_labels = [f"{y}년" for y in years]
    colors = ["red" if cum < 0 else "royalblue" for cum in cumulative]
    bar_fig = go.Figure(
        data=[go.Bar(x=x_labels, y=cumulative, marker=dict(color=colors),
                     text=[f"{v:,.0f}원" for v in cumulative], textposition="outside")]
    )
    if break_even_year is not None:
        be_label = f"{break_even_year}년"
        bar_fig.add_shape(type="line", x0=be_label, x1=be_label, y0=0, y1=1,
                          xref="x", yref="paper", line=dict(color="green", width=2, dash="dash"))
        bar_fig.add_annotation(x=be_label, y=1, xref="x", yref="paper",
                               text=f"손익분기 {break_even_year}년", showarrow=False, yanchor="bottom",
                               font=dict(color="green"))
    bar_fig.update_layout(title="연도별 순현금흐름 (누적)", yaxis=dict(tickformat=","), bargap=0.25)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ----- 표 -----
    st.subheader("연도별 금액 확인")
    df_table = pd.DataFrame({
        "연도": years,
        "순현금흐름(원)": yearly_cash,
        "누적(원)": cumulative,
        "PV 수입(원)": cf["pv_revenues"],
        "V2G 수입(원)": cf["v2g_revenues"],
        "O&M 비용(원)": cf["om_costs"],
        "CAPEX(원)": cf["capex_list"],
    })
    st.dataframe(df_table, use_container_width=True)

    # (선택) CSV에서 읽은 연도별 PV kWh를 확인하고 싶다면 아래 주석 해제
    # if pv_kwh_by_year:
    #     st.write("CSV 기반 연도별 PV 연간 발전량(kWh) 샘플:", dict(list(pv_kwh_by_year.items())[:5]))


if __name__ == "__main__":
    main()
