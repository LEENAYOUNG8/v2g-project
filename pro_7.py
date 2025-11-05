import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import pandas as pd

# kWh 당 탄소배출계수 (kg CO2e/kWh)
EMISSION_FACTOR_KG_PER_KWH = 0.495  # 네가 준 값

# =========================
# 1. 한글 폰트: repo에 올려둔 NanumGothic.ttf 강제 사용
# =========================
def set_korean_font():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "NanumGothic.ttf")

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
    else:
        plt.rcParams["axes.unicode_minus"] = False

set_korean_font()
# =========================


def price_with_cagr(base_price, base_year, year, cagr):
    return base_price * (1 + cagr) ** (year - base_year)


def make_v2g_model_params():
    return {
        "pv_capacity_kw": 125,
        "pv_annual_kwh": 153_300,
        "self_use_ratio": 0.60,
        "num_v2g_chargers": 6,
        "v2g_charger_unit_cost": 25_000_000,
        "v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,
        "v2g_operating_days": 300,
        "tariff_base_year": 2025,
        "pv_base_price": 160,
        "v2g_price_gap": 30,
        "price_cagr": 0.043,
        "om_ratio": 0.015,
    }


def build_yearly_cashflows(install_year: int, current_year: int, p: dict):
    # PV
    annual_pv_kwh = p["pv_annual_kwh"]
    annual_pv_surplus_kwh = annual_pv_kwh * (1 - p["self_use_ratio"])

    # V2G (연간 실제 방전량)
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = (
        daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]
    )

    # CAPEX / O&M
    capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
    annual_om_cost = capex_total * p["om_ratio"]

    years = list(range(install_year, current_year + 1))
    yearly_cash = []
    cumulative = []
    pv_revenues = []
    v2g_revenues = []
    om_costs = []
    capex_list = []

    cum = 0
    for i, year in enumerate(years):
        pv_price_y = price_with_cagr(
            p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"]
        )
        v2g_price_y = pv_price_y + p["v2g_price_gap"]

        revenue_pv_y = annual_pv_surplus_kwh * pv_price_y
        revenue_v2g_y = annual_v2g_kwh * v2g_price_y
        annual_revenue_y = revenue_pv_y + revenue_v2g_y

        om_y = annual_om_cost
        capex_y = capex_total if i == 0 else 0

        cf = annual_revenue_y - om_y - capex_y
        cum += cf

        yearly_cash.append(cf)
        cumulative.append(cum)
        pv_revenues.append(revenue_pv_y)
        v2g_revenues.append(revenue_v2g_y)
        om_costs.append(om_y)
        capex_list.append(capex_y)

    return {
        "years": years,
        "yearly_cash": yearly_cash,
        "cumulative": cumulative,
        "pv_revenues": pv_revenues,
        "v2g_revenues": v2g_revenues,
        "om_costs": om_costs,
        "capex_list": capex_list,
        # 👇 탄소계산용으로 연간 kWh도 같이 리턴
        "annual_pv_surplus_kwh": annual_pv_surplus_kwh,
        "annual_v2g_kwh": annual_v2g_kwh,
    }


def won_formatter(x, pos):
    return f"{int(x):,}"


def main():
    st.title("V2G 투자 대비 연도별/누적 현금흐름")

    params = make_v2g_model_params()

    # ===== 사이드바 입력 =====
    st.sidebar.header("시뮬레이션 입력")
    install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
    current_year = st.sidebar.number_input(
        "마지막 연도", value=2045, step=1, min_value=install_year
    )

    params["num_v2g_chargers"] = st.sidebar.number_input(
        "V2G 충전기 대수",
        value=params["num_v2g_chargers"],
        step=1,
        min_value=1,
    )
    params["v2g_daily_discharge_per_charger_kwh"] = st.sidebar.number_input(
        "1대당 일일 방전량(kWh)",
        value=params["v2g_daily_discharge_per_charger_kwh"],
        step=1,
        min_value=1,
    )
    params["v2g_operating_days"] = st.sidebar.number_input(
        "연간 운영일 수",
        value=params["v2g_operating_days"],
        step=10,
        min_value=1,
        max_value=365,
    )
    params["pv_annual_kwh"] = st.sidebar.number_input(
        "연간 PV 발전량(kWh)",
        value=params["pv_annual_kwh"],
        step=1000,
        min_value=0,
    )
    params["self_use_ratio"] = st.sidebar.slider(
        "PV 자가소비 비율",
        min_value=0.0,
        max_value=1.0,
        value=params["self_use_ratio"],
        step=0.05,
    )
    params["pv_base_price"] = st.sidebar.number_input(
        "PV 기준단가(원/kWh)",
        value=params["pv_base_price"],
        step=5,
        min_value=0,
    )
    params["price_cagr"] = st.sidebar.number_input(
        "전력단가 연평균 상승률",
        value=params["price_cagr"],
        step=0.001,
        format="%.3f",
    )

    # ===== 계산 =====
    cf_data = build_yearly_cashflows(install_year, current_year, params)
    years = cf_data["years"]
    yearly_cash = cf_data["yearly_cash"]
    cumulative = cf_data["cumulative"]

    # 손익분기 연도
    break_even_year = None
    for y, cum_val in zip(years, cumulative):
        if cum_val >= 0:
            break_even_year = y
            break

    # ===== 탄소절감량 계산 =====
    # 1년당 대체한 kWh = 남는 PV + V2G 방전
    clean_kwh_per_year = (
        cf_data["annual_pv_surplus_kwh"] + cf_data["annual_v2g_kwh"]
    )
    num_years = len(years)
    total_clean_kwh = clean_kwh_per_year * num_years
    # kg CO2e
    total_co2_kg = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH
    # ton CO2e
    total_co2_ton = total_co2_kg / 1000.0

    # ===== KPI =====
    col1, col2, col3 = st.columns(3)
    if break_even_year is not None:
        col1.metric("손익분기 연도", f"{break_even_year}년")
    else:
        col1.metric("손익분기 연도", "아직 미도달")

    val_str = "{:,.0f}".format(cumulative[-1])
    col2.metric("마지막 연도 누적", f"{val_str} 원")

    # 탄소절감은 tCO2e로 1자리만
    col3.metric("누적 탄소절감량", f"{total_co2_ton:,.1f} tCO₂e")

    # ===== 1) 누적 현금흐름 (matplotlib) =====
    st.subheader("누적 현금흐름")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, cumulative, marker="o", linewidth=2.2)
    ax.set_xlabel("연도")
    ax.set_ylabel("누적 금액(원)")
    ax.yaxis.set_major_formatter(FuncFormatter(won_formatter))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("V2G 투자 대비 연도별/누적 현금흐름")
    if break_even_year is not None:
        ax.axvline(break_even_year, color="green", linestyle="--", alpha=0.7)
        ax.text(
            break_even_year,
            0,
            f"손익분기 {break_even_year}",
            color="green",
            va="bottom",
            ha="left",
        )
    st.pyplot(fig)

    # ===== 2) 연도별 순현금흐름 (누적 막대) =====
    st.subheader("연도별 순현금흐름 (누적)")

    x_labels = [f"{y}년" for y in years]
    colors = ["red" if cum < 0 else "royalblue" for cum in cumulative]

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=x_labels,
                y=cumulative,
                marker=dict(color=colors),
                text=[f"{v:,.0f}원" for v in cumulative],
                textposition="outside",
            )
        ]
    )

    if break_even_year is not None:
        be_label = f"{break_even_year}년"
        bar_fig.add_shape(
            type="line",
            x0=be_label,
            x1=be_label,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="green", width=2, dash="dash"),
        )
        bar_fig.add_annotation(
            x=be_label,
            y=1,
            xref="x",
            yref="paper",
            text=f"손익분기 {break_even_year}년",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="green"),
        )

    bar_fig.update_layout(
        title="연도별 순현금흐름 (누적)",
        yaxis=dict(tickformat=","),
        bargap=0.25,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # ===== 표 =====
    st.subheader("연도별 금액 확인")
    df_table = pd.DataFrame(
        {
            "연도": years,
            "순현금흐름(원)": yearly_cash,
            "누적(원)": cumulative,
            "PV 수입(원)": cf_data["pv_revenues"],
            "V2G 수입(원)": cf_data["v2g_revenues"],
            "O&M 비용(원)": cf_data["om_costs"],
            "CAPEX(원)": cf_data["capex_list"],
            # 참고용으로 연간 에너지도 보여줄 수 있음
            "연간 PV 잉여(kWh)": [cf_data["annual_pv_surplus_kwh"]] * len(years),
            "연간 V2G 방전(kWh)": [cf_data["annual_v2g_kwh"]] * len(years),
        }
    )
    st.dataframe(df_table, use_container_width=True)


if __name__ == "__main__":
    main()
