# app.py
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
EMISSION_FACTOR_KG_PER_KWH = 0.495   # 국내 전력 1kWh당 약 0.495 kgCO2e
MJ_PER_M2_TO_KWH_PER_M2    = 0.27778 # MJ/m² → kWh/m²

# =========================
# 1) 한글 폰트 (NanumGothic.ttf 자동 사용)
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
def price_with_cagr(base_price, base_year, year, cagr):
    return base_price * (1 + cagr) ** (year - base_year)

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
# 3) CSV → 연도별 PV kWh 계산
# =========================
def load_irradiance_csv(csv_path: str) -> pd.DataFrame:
    """
    jeju.csv 같은 연도별 일사합(MJ/m²) 파일을 읽어 ['year','ghi_mj_m2']로 반환.
    연도 컬럼 후보: ['연도','year','일시']
    일사합 컬럼 후보: ['일사합(mj/m2)','일사합(mj/m²)','ghi_mj','ghi(mj/m2)','ghi(mj/m²)','solar_mj','solar(mj/m2)']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV가 없음: {csv_path}")

    tried = []
    df = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            tried.append((enc, str(e)))
    if df is None:
        raise ValueError(f"CSV 읽기 실패: {tried}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    year_candidates = ["연도", "year", "일시"]
    ghi_candidates  = ["일사합(mj/m2)", "일사합(mj/m²)", "ghi_mj",
                       "ghi(mj/m2)", "ghi(mj/m²)", "solar_mj", "solar(mj/m2)"]

    year_col = next((c for c in year_candidates if c in df.columns), None)
    if year_col is None:
        raise ValueError(f"[연도 컬럼 없음] 실제 열: {list(df.columns)}")

    ghi_col = next((c for c in ghi_candidates if c in df.columns), None)
    if ghi_col is None:
        maybe = [c for c in df.columns if ("mj" in c and "m" in c)]
        raise ValueError(f"[일사합 컬럼 없음] 실제 열: {list(df.columns)} / 유사 후보: {maybe}")

    out = pd.DataFrame({
        "year": pd.to_numeric(df[year_col], errors="coerce"),
        "ghi_mj_m2": pd.to_numeric(df[ghi_col], errors="coerce"),
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
    연도별 일사합(MJ/m²) → kWh/m² 변환 후
    PV 연간 발전량(kWh) = GHI_kWh/m² × 총면적(m²) × PR
    """
    area_m2 = float(panel_width_m) * float(panel_height_m) * int(n_modules)
    if pr_manual is None:
        PR = pr_base * availability * (1.0 - soiling) * inv_eff
    else:
        PR = float(pr_manual)

    ghi_kwh_m2 = irr_df["ghi_mj_m2"].astype(float) * MJ_PER_M2_TO_KWH_PER_M2
    pv_kwh     = ghi_kwh_m2 * area_m2 * PR

    out = dict(zip(irr_df["year"].astype(int).tolist(),
                   pv_kwh.astype(float).tolist()))
    return out, area_m2, PR

# =========================
# 4) V2G/PV 파라미터
# =========================
def make_v2g_model_params():
    return {
        # PV
        "self_use_ratio": 0.60,           # 자가소비 비율(예: 0.60 → 40%가 잉여)
        "v2g_share_of_surplus": 0.40,     # 잉여 중 V2G로 보내는 비율(예: 0.40)

        # V2G (용량/열화)
        "num_v2g_chargers": 6,
        "v2g_charger_unit_cost": 25_000_000,
        "v2g_daily_discharge_per_charger_kwh": 35,
        "v2g_operating_days": 300,
        "degradation_factor": 0.90,       # 초기 가용률(연간 고정 보정)
        "v2g_degradation_annual": 0.01,   # 연차 열화율(매년 1% 감소)

        # 단가(연복리 상승)
        "tariff_base_year": 2025,
        "pv_base_price": 160,
        "v2g_price_gap": 30,
        "price_cagr": 0.043,

        # O&M
        "om_ratio": 0.015,
    }

# =========================
# 5) 현금흐름 빌드 (CSV 기반 PV연간 kWh 사용 + V2G 분배/열화)
# =========================
def build_yearly_cashflows_from_csv(install_year: int, current_year: int, p: dict,
                                    pv_kwh_by_year: dict):
    """
    pv_kwh_by_year: {year: kWh}
    - PV 잉여: pv_kwh * (1 - self_use_ratio)
    - 그 중 일부(v2g_share_of_surplus)를 V2G로 보내되, 연도별 V2G 용량(연차 열화 반영)을 초과할 수 없음
    - 남은 잉여는 PV 단가로 판매
    - V2G는 (PV 단가 + gap)으로 판매
    """
    years = list(range(install_year, current_year + 1))
    if not pv_kwh_by_year:
        raise ValueError("CSV로부터 계산된 PV 연간 발전량이 없습니다.")

    min_y, max_y = min(pv_kwh_by_year.keys()), max(pv_kwh_by_year.keys())

    self_use = float(p["self_use_ratio"])
    v2g_share = float(p.get("v2g_share_of_surplus", 0.0))

    # CAPEX/O&M
    capex_total   = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost= capex_total * p["om_ratio"]

    yearly_cash, cumulative = [], []
    pv_revenues, v2g_revenues, om_costs, capex_list = [], [], [], []

    # V2G 기본 용량(연 1년차 기준)
    base_v2g_capacity = (
        p["num_v2g_chargers"]
        * p["v2g_daily_discharge_per_charger_kwh"]
        * p["v2g_operating_days"]
        * p["degradation_factor"]
    )

    cum = 0.0
    # 평균 kWh 계산용 누적
    pv_surplus_list = []
    v2g_kwh_list    = []

    for i, year in enumerate(years):
        # 연차 열화 반영한 연도별 V2G 용량
        years_elapsed = max(0, year - install_year)
        v2g_capacity_y = base_v2g_capacity * ((1.0 - p["v2g_degradation_annual"]) ** years_elapsed)

        # CSV 범위 밖 연도는 경계값 사용(보수적): 2000 미만→2000, 2025 초과→2025
        y_key = min(max(year, min_y), max_y)
        annual_pv_kwh = pv_kwh_by_year[y_key]

        # PV 잉여
        pv_surplus_y = annual_pv_kwh * (1.0 - self_use)

        # 잉여 중 V2G로 보낼 입력(상한: 잉여)
        v2g_input_y = pv_surplus_y * v2g_share

        # 실제 V2G 처리량: 입력과 용량 중 최소
        v2g_kwh_y = min(v2g_input_y, v2g_capacity_y)

        # PV로 남는 잉여 = 전체 잉여 - V2G로 보낸 양
        pv_direct_kwh_y = max(0.0, pv_surplus_y - v2g_kwh_y)

        # 단가
        pv_price_y  = price_with_cagr(p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"])
        v2g_price_y = pv_price_y + p["v2g_price_gap"]

        # 수입
        revenue_pv_y  = pv_direct_kwh_y * pv_price_y
        revenue_v2g_y = v2g_kwh_y * v2g_price_y
        annual_revenue_y = revenue_pv_y + revenue_v2g_y

        # 비용
        om_y    = annual_om_cost
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

        pv_surplus_list.append(pv_surplus_y)
        v2g_kwh_list.append(v2g_kwh_y)

    # 리포팅용 평균값
    avg_pv_surplus_kwh = float(np.mean(pv_surplus_list)) if pv_surplus_list else 0.0
    avg_v2g_kwh        = float(np.mean(v2g_kwh_list)) if v2g_kwh_list else 0.0

    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        # 탄소/요약용
        "annual_pv_surplus_kwh": avg_pv_surplus_kwh,
        "annual_v2g_kwh": avg_v2g_kwh,
    }

# =========================
# 6) Streamlit App
# =========================
def main():
    st.title("V2G + PV 경제성 (CSV 일사합 기반, V2G 분배/열화 반영)")

    # --- CSV는 사이드바에 노출하지 않고 자동 로드 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "jeju.csv")  # 레포에 함께 넣어두세요.
    try:
        irr_df = load_irradiance_csv(csv_path)
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        st.stop()

    # --- 시스템 파라미터 사이드바 ---
    st.sidebar.header("시스템/요금 설정")

    # 모듈/면적/PR
    c1, c2 = st.sidebar.columns(2)
    panel_w   = c1.number_input("패널 폭(m)", value=1.46, step=0.01, format="%.2f")
    panel_h   = c2.number_input("패널 높이(m)", value=0.98, step=0.01, format="%.2f")
    n_modules = st.sidebar.number_input("모듈 수(장)", value=250, step=5, min_value=1)

    st.sidebar.caption("PR은 고정 계수(기본) 또는 직접 지정")
    use_manual_pr = st.sidebar.checkbox("PR 수동 지정", value=False)
    if use_manual_pr:
        pr_manual = st.sidebar.number_input("PR (0~1)", value=0.78, min_value=0.0, max_value=1.0, step=0.01)
    else:
        pr_manual = None
    # 고정 계수(기본)
    pr_base, availability, soiling, inv_eff = 0.82, 0.98, 0.02, 0.97

    # 시뮬레이션 기간
    install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
    current_year = st.sidebar.number_input("마지막 연도", value=2045, step=1, min_value=install_year)

    # 요금/비율
    params = make_v2g_model_params()
    params["self_use_ratio"]       = st.sidebar.slider("PV 자가소비 비율", 0.0, 1.0, params["self_use_ratio"], 0.05)
    params["v2g_share_of_surplus"] = st.sidebar.slider("잉여 중 V2G 비율", 0.0, 1.0, params["v2g_share_of_surplus"], 0.05)
    params["pv_base_price"]        = st.sidebar.number_input("PV 기준단가 (원/kWh)", value=params["pv_base_price"], step=5, min_value=0)
    params["price_cagr"]           = st.sidebar.number_input("전력단가 연평균 상승률", value=params["price_cagr"], step=0.001, format="%.3f")

    # V2G 하드웨어/용량/열화
    v1, v2, v3 = st.sidebar.columns(3)
    params["num_v2g_chargers"] = v1.number_input("V2G 대수", value=params["num_v2g_chargers"], step=1, min_value=1)
    params["v2g_daily_discharge_per_charger_kwh"] = v2.number_input(
        "1대당 일일 방전(kWh)", value=params["v2g_daily_discharge_per_charger_kwh"], step=1, min_value=1
    )
    params["v2g_operating_days"] = v3.number_input(
        "연 운영일", value=params["v2g_operating_days"], step=5, min_value=1, max_value=365
    )
    params["degradation_factor"] = st.sidebar.number_input(
        "초기 가용률(고정 보정)", value=params["degradation_factor"], min_value=0.5, max_value=1.0, step=0.01, format="%.2f"
    )
    params["v2g_degradation_annual"] = st.sidebar.number_input(
        "V2G 연차 열화율", value=params["v2g_degradation_annual"], min_value=0.0, max_value=0.2, step=0.005, format="%.3f",
        help="해마다 V2G 용량이 (1-열화율) 만큼 감소"
    )

    # CAPEX/O&M
    params["v2g_charger_unit_cost"] = st.sidebar.number_input(
        "V2G 대당 CAPEX(원)", value=params["v2g_charger_unit_cost"], step=1_000_000, min_value=0
    )
    params["om_ratio"] = st.sidebar.number_input("O&M 비율(연)", value=params["om_ratio"], step=0.001, format="%.3f")

    # 재무 지표용 할인율
    discount_rate = st.sidebar.number_input("할인율(연)", value=0.08, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")

    # --- PV 연간 kWh 계산 ---
    pv_by_year, area_m2, PR_used = compute_pv_kwh_by_year(
        irr_df,
        panel_width_m=panel_w, panel_height_m=panel_h, n_modules=int(n_modules),
        pr_base=pr_base, availability=availability, soiling=soiling, inv_eff=inv_eff,
        pr_manual=pr_manual
    )

    # --- 현금흐름 계산 (CSV 기반 + V2G 분배/열화) ---
    cf = build_yearly_cashflows_from_csv(install_year, current_year, params, pv_by_year)
    years        = cf["years"]
    yearly_cash  = cf["yearly_cash"]
    cumulative   = cf["cumulative"]

    # 손익분기 연도
    break_even_year = next((y for y, cum in zip(years, cumulative) if cum >= 0), None)

    # 탄소절감량(kgCO2e) – 평균 연간 kWh로 추정
    clean_kwh_per_year = cf["annual_pv_surplus_kwh"] + cf["annual_v2g_kwh"]
    total_clean_kwh = clean_kwh_per_year * len(years)
    total_co2_kg    = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH

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
    ax.set_title("V2G + PV 누적 현금흐름 (CSV 기반)")
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(break_even_year, 0, f"손익분기 {break_even_year}", color="green", va="bottom", ha="left")
    st.pyplot(fig)

    # --- 그래프: 연도별 누적 막대 ---
    st.subheader("연도별 순현금흐름 (누적)")
    x_labels = [f"{y}년" for y in years]
    colors   = ["red" if cum < 0 else "royalblue" for cum in cumulative]
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
        f"총 모듈 면적: {area_m2:.1f} m² | 사용 PR: {PR_used:.3f} "
        f"| CSV 연도 범위: {min(pv_by_year.keys())}~{max(pv_by_year.keys())} "
        f"| CSV 파일: {os.path.basename(csv_path)}"
    )

if __name__ == "__main__":
    main()
