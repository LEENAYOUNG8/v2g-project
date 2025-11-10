#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# -----------------------------
# 상수
# -----------------------------
EMISSION_FACTOR_KG_PER_KWH = 0.495   # 국내 전력 1kWh 생산시 약 0.495 kgCO2e
MJ_PER_M2_TO_KWH_PER_M2 = 0.27778    # MJ/m² → kWh/m² 변환 계수

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
# 2) 재무 유틸
# =========================
def npv(rate: float, cashflows: list[float]) -> float:
    if rate <= -0.999999:
        return float("nan")
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows)))

def irr_bisection(cashflows: list[float], lo=-0.99, hi=3.0, tol=1e-7, max_iter=200):
    def f(r):
        try:
            return npv(r, cashflows)
        except Exception:
            return np.nan
    f_lo, f_hi = f(lo), f(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return None
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
    disc = []
    cum = 0.0
    for t, cf in enumerate(cashflows):
        val = cf / ((1 + rate) ** t)
        cum += val
        disc.append(cum)
    for k, v in enumerate(disc):
        if v >= 0:
            if k == 0:
                return 0.0
            prev = disc[k - 1]
            frac = 0.0 if v == prev else (-prev) / (v - prev)
            return (k - 1) + max(0.0, min(1.0, frac))
    return None

def won_formatter(x, pos):
    return f"{int(x):,}"

# =========================
# 3) CSV → 연도별 PV kWh 계산 (일사합)
# =========================
def load_irradiance_csv(csv_path: str) -> pd.DataFrame:
    """
    연도별 '일사합(MJ/m²)' CSV를 읽어 표준 컬럼으로 정리:
      - year  : ['연도', 'year', '일시'] 중 하나
      - ghi_mj_m2 : ['일사합(mj/m2)', '일사합(mj/m²)', 'ghi_mj', 'ghi(mj/m2)' ...]
    반환 DF: ['year', 'ghi_mj_m2']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV가 없음: {csv_path}")

    tried = []
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            tried.append((enc, str(e)))
            df = None
    if df is None:
        raise ValueError(f"CSV 읽기 실패: {tried}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    # 연도
    year_candidates = ["연도", "year", "일시"]
    year_col = next((c for c in year_candidates if c in df.columns), None)
    if year_col is None:
        raise ValueError(f"[연도 컬럼 없음] 실제 열: {list(df.columns)}")

    # 일사합
    ghi_candidates = [
        "일사합(mj/m2)", "일사합(mj/m²)", "ghi_mj", "ghi(mj/m2)", "ghi(mj/m²)",
        "solar_mj", "solar(mj/m2)"
    ]
    ghi_col = next((c for c in ghi_candidates if c in df.columns), None)
    if ghi_col is None:
        maybe = [c for c in df.columns if ("mj" in c and "m" in c)]
        raise ValueError(f"[일사합 컬럼 없음] 실제 열: {list(df.columns)} / 유사: {maybe}")

    out = pd.DataFrame({
        "year": pd.to_numeric(df[year_col], errors="coerce"),
        "ghi_mj_m2": pd.to_numeric(df[ghi_col], errors="coerce")
    }).dropna()

    out["year"] = out["year"].astype(int)
    return out.sort_values("year").reset_index(drop=True)

def compute_pv_kwh_by_year(irr_df: pd.DataFrame,
                           panel_width_m=1.46,
                           panel_height_m=0.98,
                           n_modules=250,
                           pr_base=0.82, availability=0.98, soiling=0.02, inv_eff=0.97,
                           pr_manual=None):
    """
    PV 연간 발전량(kWh) = (일사합 MJ/m² × 0.27778) × 총면적(m²) × PR
    pr_manual 지정 시 그 값 사용, 아니면 고정 손실계수로 PR 산정.
    반환: (dict{year:kWh}, area_m2, PR)
    """
    area_m2 = float(panel_width_m) * float(panel_height_m) * int(n_modules)
    if pr_manual is None:
        PR = pr_base * availability * (1.0 - soiling) * inv_eff
    else:
        PR = float(pr_manual)

    ghi_kwh_m2 = irr_df["ghi_mj_m2"].astype(float) * MJ_PER_M2_TO_KWH_PER_M2
    pv_kwh = ghi_kwh_m2 * area_m2 * PR
    out = dict(zip(irr_df["year"].astype(int).tolist(), pv_kwh.astype(float).tolist()))
    return out, area_m2, PR

# =========================
# 4) SMP CSV → 연도별 단가
# =========================
def load_smp_csv(csv_path: str) -> pd.DataFrame:
    """
    SMP.csv 로드 → 표준 컬럼으로 정리
    필요한 열(케이스별 자동 탐색):
      - 날짜/시간: ['일시','날짜','date','datetime','시간','time']
      - 단가: ['smp','smp(원/kwh)','smp(원/mwh)','가격','price','단가']
    반환: ['ts'(datetime), 'smp_won_per_kwh'(float)]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SMP CSV가 없음: {csv_path}")

    last_err = None
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise ValueError(f"SMP CSV 읽기 실패: {last_err}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    # 날짜 후보
    ts_col = next((c for c in ["일시","날짜","date","datetime","시간","time"] if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"날짜/시간 컬럼을 찾지 못함. 실제 열: {list(df.columns)}")

    # 단가 후보
    price_candidates = [c for c in df.columns if ("smp" in c) or ("price" in c) or ("단가" in c) or ("가격" in c)]
    if not price_candidates:
        raise ValueError(f"SMP/가격 컬럼을 찾지 못함. 실제 열: {list(df.columns)}")
    pcol = price_candidates[0]

    # 파싱
    df["ts"] = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["ts"]).copy()

    ser = pd.to_numeric(df[pcol].astype(str).str.replace(",",""), errors="coerce")
    # 단위 추정: 값이 매우 크면 원/MWh로 보고 /1000
    median_val = np.nanmedian(ser)
    if median_val > 1000:
        ser = ser / 1000.0

    out = pd.DataFrame({
        "ts": df["ts"],
        "smp_won_per_kwh": ser
    }).dropna(subset=["smp_won_per_kwh"])
    return out.sort_values("ts").reset_index(drop=True)

def yearly_smp(smp_df: pd.DataFrame,
               method: str = "avg",
               peak_hours: tuple = (9, 21),
               smooth_ma_years: int = 0) -> pd.Series:
    """
    연도별 SMP 시리즈 생성 (원/kWh)
    method:
      - 'avg'    : 단순 연평균
      - 'peak'   : 피크시간(peak_hours 시작~종료-1시)만 평균
      - 'offpeak': 피크 제외 평균
    smooth_ma_years: 0이면 미적용, 3이면 3개년 이동평균
    반환: pd.Series(index=year(int), value=price(float))
    """
    df = smp_df.copy()
    df["year"] = df["ts"].dt.year
    df["hour"] = df["ts"].dt.hour

    if method == "peak":
        h0, h1 = peak_hours
        df = df[(df["hour"] >= h0) & (df["hour"] < h1)]
    elif method == "offpeak":
        h0, h1 = peak_hours
        df = df[(df["hour"] < h0) | (df["hour"] >= h1)]
    # else 'avg' → 그대로

    grp = df.groupby("year", as_index=True)["smp_won_per_kwh"].mean().dropna()

    if smooth_ma_years and smooth_ma_years > 1:
        grp = grp.rolling(smooth_ma_years, min_periods=1).mean()

    return grp

def get_price_for_year(year: int,
                       yearly_price: pd.Series,
                       clamp: bool = True) -> float:
    """
    해당 연도의 단가를 가져온다.
    - 동일 연도가 없으면:
      clamp=True  → 범위를 벗어나면 경계값(최초/최종년도 가격) 사용
      clamp=False → (옵션) 보간/외삽
    """
    if year in yearly_price.index:
        return float(yearly_price.loc[year])
    years = yearly_price.index.to_list()
    if clamp:
        return float(yearly_price.loc[years[0]] if year < years[0] else yearly_price.loc[years[-1]])
    # 간단 보간(연속연도 가정)
    s = yearly_price.reindex(range(min(years), max(years)+1)).interpolate()
    if year < s.index.min():
        return float(s.iloc[0])
    if year > s.index.max():
        return float(s.iloc[-1])
    return float(s.loc[year])

# =========================
# 5) 기본 파라미터
# =========================
def make_v2g_model_params():
    return {
        "self_use_ratio": 0.60,           # 자가소비 비율
        # V2G
        "num_v2g_chargers": 6,
        "v2g_charger_unit_cost": 25_000_000,
        "v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,
        "v2g_operating_days": 300,
        # (아래 3개는 이번 버전에서 미사용: SMP로 대체)
        "tariff_base_year": 2025,         # 미사용
        "pv_base_price": 160,             # 미사용
        "price_cagr": 0.043,              # 미사용
        # O&M
        "om_ratio": 0.015,
    }

# =========================
# 6) 현금흐름 빌드 (CSV 기반 PV + SMP 기반 단가)
# =========================
def build_yearly_cashflows_from_csv(install_year: int,
                                    current_year: int,
                                    p: dict,
                                    pv_kwh_by_year: dict,
                                    smp_year_price: pd.Series):
    """
    pv_kwh_by_year: {year: kWh}  (일사합 → PV 연간 발전량)
    smp_year_price: 연도별 SMP(원/kWh) 시리즈
    - PV 단가 = 그 해의 연평균 SMP
    - V2G 단가 = PV 단가 (프리미엄 미적용; 요구사항)
    - CSV 범위 밖 연도는 가격/발전량 모두 경계값 사용(clamp)
    """
    # V2G(연간 고정 kWh)
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]

    capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost = capex_total * p["om_ratio"]

    self_use = float(p["self_use_ratio"])
    years = list(range(install_year, current_year + 1))
    yearly_cash, cumulative = [], []
    pv_revenues, v2g_revenues, om_costs, capex_list = [], [], [], []

    if len(pv_kwh_by_year) == 0:
        raise ValueError("CSV에서 계산된 PV 연간 발전량이 없습니다.")
    min_y, max_y = min(pv_kwh_by_year.keys()), max(pv_kwh_by_year.keys())

    cum = 0.0
    for i, year in enumerate(years):
        # PV kWh: 범위 밖이면 경계값
        y_key = min(max(year, min_y), max_y)
        annual_pv_kwh = pv_kwh_by_year[y_key]
        annual_pv_surplus_kwh = annual_pv_kwh * (1 - self_use)

        # 가격: SMP 연평균 (V2G 프리미엄 미적용 → 동일 단가)
        pv_price_y = get_price_for_year(year, smp_year_price, clamp=True)
        v2g_price_y = pv_price_y

        # 수입
        revenue_pv_y = annual_pv_surplus_kwh * pv_price_y
        revenue_v2g_y = annual_v2g_kwh * v2g_price_y
        annual_revenue_y = revenue_pv_y + revenue_v2g_y

        # 비용
        om_y = annual_om_cost
        capex_y = capex_total if i == 0 else 0

        # 순현금흐름
        cf = annual_revenue_y - om_y - capex_y
        cum += cf

        yearly_cash.append(cf)
        cumulative.append(cum)
        pv_revenues.append(revenue_pv_y)
        v2g_revenues.append(revenue_v2g_y)
        om_costs.append(om_y)
        capex_list.append(capex_y)

    # 보고용 평균(간단화)
    avg_pv_surplus_kwh = np.mean([pv_kwh_by_year[min(max(y, min_y), max_y)] * (1 - self_use) for y in years])
    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        "annual_pv_surplus_kwh": avg_pv_surplus_kwh,
        "annual_v2g_kwh": annual_v2g_kwh,
    }

# =========================
# 7) Streamlit App
# =========================
def main():
    st.title("V2G 투자 대비 연도별/누적 현금흐름 (SMP 연평균 단가 & 일사합 기반 PV)")

    # --- 고정 경로 (레포 폴더 내) ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    irr_csv_path = os.path.join(base_dir, "jeju.csv")  # 일사합 CSV
    smp_csv_path = os.path.join(base_dir, "SMP.csv")   # SMP CSV

    # --- 사이드바: PV 시스템 파라미터 ---
    st.sidebar.header("PV 시스템 설정")
    c1, c2 = st.sidebar.columns(2)
    panel_w = c1.number_input("패널 폭(m)", value=1.46, step=0.01, format="%.2f")
    panel_h = c2.number_input("패널 높이(m)", value=0.98, step=0.01, format="%.2f")
    n_modules = st.sidebar.number_input("모듈 수(장)", value=250, step=5, min_value=1)

    st.sidebar.caption("PR은 고정 계수(기본) 또는 직접 지정(수동) 중 선택")
    use_manual_pr = st.sidebar.checkbox("PR 수동 지정", value=False)
    if use_manual_pr:
        pr_manual = st.sidebar.number_input("PR (0~1)", value=0.78, min_value=0.0, max_value=1.0, step=0.01)
    else:
        pr_manual = None
    pr_base, availability, soiling, inv_eff = 0.82, 0.98, 0.02, 0.97

    # --- 사이드바: 시뮬레이션/요금/비율 ---
    st.sidebar.header("시뮬레이션/요금")
    install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
    current_year = st.sidebar.number_input("마지막 연도", value=2045, step=1, min_value=install_year)

    params = make_v2g_model_params()

    # V2G 관련(요청대로 유지)
    params["num_v2g_chargers"] = st.sidebar.number_input(
        "V2G 충전기 대수", value=params["num_v2g_chargers"], step=1, min_value=1
    )
    params["v2g_daily_discharge_per_charger_kwh"] = st.sidebar.number_input(
        "1대당 일일 방전량 (kWh)", value=params["v2g_daily_discharge_per_charger_kwh"], step=1, min_value=1
    )
    params["v2g_operating_days"] = st.sidebar.number_input(
        "연간 운영일 수", value=params["v2g_operating_days"], step=10, min_value=1, max_value=365
    )

    params["self_use_ratio"] = st.sidebar.slider("PV 자가소비 비율", 0.0, 1.0, params["self_use_ratio"], 0.05)

    # (아래 2개는 이번 버전에서 '미사용'; SMP 단가를 그대로 씀)
    st.sidebar.number_input("PV 기준단가 (원/kWh) [미사용]", value=params["pv_base_price"], step=5, min_value=0)
    st.sidebar.number_input("전력단가 연평균 상승률 [미사용]", value=params["price_cagr"], step=0.001, format="%.3f")

    # 재무 지표용 할인율
    discount_rate = st.sidebar.number_input("할인율(연)", value=0.08, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")

    # --- CSV 로드 ---
    irr_df = load_irradiance_csv(irr_csv_path)
    pv_by_year, area_m2, PR_used = compute_pv_kwh_by_year(
        irr_df,
        panel_width_m=panel_w, panel_height_m=panel_h, n_modules=int(n_modules),
        pr_base=pr_base, availability=availability, soiling=soiling, inv_eff=inv_eff,
        pr_manual=pr_manual
    )

    smp_df = load_smp_csv(smp_csv_path)
    smp_year = yearly_smp(smp_df, method="avg", smooth_ma_years=0)  # 방법 1: 연평균 그대로 사용

    # --- 현금흐름 계산 (SMP 단가 적용) ---
    cf = build_yearly_cashflows_from_csv(install_year, current_year, params, pv_by_year, smp_year)
    years = cf["years"]; yearly_cash = cf["yearly_cash"]; cumulative = cf["cumulative"]

    # 손익분기 연도
    break_even_year = next((y for y, cum in zip(years, cumulative) if cum >= 0), None)

    # 탄소절감량(kgCO2e) – 평균 kWh로 보고
    clean_kwh_per_year = cf["annual_pv_surplus_kwh"] + cf["annual_v2g_kwh"]
    total_clean_kwh = clean_kwh_per_year * len(years)
    total_co2_kg = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH

    # 재무지표
    npv_val = npv(discount_rate, yearly_cash)
    irr_val = irr_bisection(yearly_cash)
    dpp_val = discounted_payback(yearly_cash, discount_rate)

    # --- KPI (1행: 재무) ---
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

    # --- KPI (2행: 시스템/환경) ---
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

    # --- 그래프: 누적 라인 ---
    st.subheader("누적 현금흐름")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cumulative, marker="o", linewidth=2.2)
    ax.set_xlabel("연도"); ax.set_ylabel("누적 금액(원)")
    ax.yaxis.set_major_formatter(FuncFormatter(won_formatter))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("V2G + PV 누적 현금흐름 (SMP 연평균 단가 적용)")
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(break_even_year, 0, f"손익분기 {break_even_year}", color="green", va="bottom", ha="left")
    st.pyplot(fig)

    # --- 그래프: 연도별 누적 막대 ---
    st.subheader("연도별 순현금흐름 (누적)")
    x_labels = [f"{y}년" for y in years]
    colors = ["red" if cum < 0 else "royalblue" for cum in cumulative]
    bar_fig = go.Figure(data=[go.Bar(x=x_labels, y=cumulative,
                                     marker=dict(color=colors),
                                     text=[f"{v:,.0f}원" for v in cumulative],
                                     textposition="outside")])
    if break_even_year is not None:
        be_label = f"{break_even_year}년"
        bar_fig.add_shape(type="line", x0=be_label, x1=be_label, y0=0, y1=1,
                          xref="x", yref="paper", line=dict(color="green", width=2, dash="dash"))
        bar_fig.add_annotation(x=be_label, y=1, xref="x", yref="paper",
                               text=f"손익분기 {break_even_year}년", showarrow=False, yanchor="bottom",
                               font=dict(color="green"))
    bar_fig.update_layout(title="연도별 순현금흐름 (누적)", yaxis=dict(tickformat=","), bargap=0.25)
    st.plotly_chart(bar_fig, use_container_width=True)

    # --- 표: 연도별 금액 ---
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

    # 참고 정보
    st.caption(
        f"총 모듈 면적: {area_m2:.1f} m² | 사용 PR: {PR_used:.3f} | "
        f"PV CSV 연도 범위: {min(pv_by_year.keys())}~{max(pv_by_year.keys())} | "
        f"SMP 연도 범위: {int(smp_year.index.min())}~{int(smp_year.index.max())} "
        f"(범위 밖 연도는 경계값 사용)"
    )

if __name__ == "__main__":
    main()
