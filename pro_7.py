MJ_PER_M2_TO_KWH_PER_M2 = 0.27778    # MJ/m² → kWh/m² 변환 계수

# =========================
# 1) 한글 폰트 세팅
# 1) 한글 폰트 (NanumGothic.ttf 를 앱 폴더에 넣어두면 자동 사용)
# =========================
def set_korean_font():
base_dir = os.path.dirname(os.path.abspath(__file__))
@@ -26,13 +26,13 @@ def set_korean_font():
fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# =========================
# 2) 재무 유틸
# =========================
def price_with_cagr(base_price, base_year, year, cagr):
    """기준연도 대비 연복리(cagr)로 상승한 단가"""
return base_price * (1 + cagr) ** (year - base_year)

def npv(rate: float, cashflows: list[float]) -> float:
@@ -63,7 +63,6 @@ def f(r):
return mid

def discounted_payback(cashflows: list[float], rate: float):
    """할인 회수기간 (부분연도 선형보간)"""
disc = []
cum = 0.0
for t, cf in enumerate(cashflows):
@@ -88,8 +87,8 @@ def won_formatter(x, pos):
def load_irradiance_csv(csv_path: str) -> pd.DataFrame:
"""
   연도별 '일사합(MJ/m²)' CSV를 읽어 표준 컬럼으로 정리:
      - year       : ['연도', 'year', '일시'] 중 하나
      - ghi_mj_m2  : ['일사합(mj/m2)', '일사합(mj/m²)', 'ghi_mj', 'ghi(mj/m2)'] 등
      - year  : ['연도', 'year', '일시'] 중 하나
      - ghi_mj_m2 : ['일사합(mj/m2)', '일사합(mj/m²)', 'ghi_mj', 'ghi(mj/m2)'] 등
   반환 DF: ['year', 'ghi_mj_m2']
   """
if not os.path.exists(csv_path):
@@ -155,152 +154,64 @@ def compute_pv_kwh_by_year(irr_df: pd.DataFrame,
return out, area_m2, PR

# =========================
# 4) CSV 범위 밖 연도 처리 전략
# =========================
def make_pv_forecaster_from_series(pv_by_year: dict, strategy: str = "trend"):
    """
    strategy:
      - 'clamp' : 경계값 고정(보수적)
      - 'trend' : 선형 추세 외삽 (1차 회귀)
      - 'mean5' : 최근 5년 평균 값 사용
      - 'repeat': CSV 구간을 주기로 반복
    반환: f(year: int) -> pv_kwh(float)
    """
    years = np.array(sorted(pv_by_year.keys()), dtype=int)
    values = np.array([pv_by_year[y] for y in years], dtype=float)
    y_min, y_max = years.min(), years.max()

    if strategy == "clamp":
        def f(y):
            y = int(y)
            if y <= y_min: return float(values[0])
            if y >= y_max: return float(values[-1])
            return float(pv_by_year[y])
        return f

    if strategy == "mean5":
        last5 = values[-5:] if len(values) >= 5 else values
        m5 = float(last5.mean())
        def f(y):
            y = int(y)
            if y_min <= y <= y_max: return float(pv_by_year[y])
            return m5
        return f

    if strategy == "repeat":
        seq = values.copy()
        L = len(seq)
        def f(y):
            y = int(y)
            if y_min <= y <= y_max:
                return float(pv_by_year[y])
            # 바깥 연도는 길이 L의 주기로 반복
            if y < y_min:
                idx = (y - y_min) % L
                return float(seq[idx])
            else:
                idx = (y - y_min) % L
                return float(seq[idx])
        return f

    # default: 'trend' — 1차 선형회귀로 외삽
    coeff = np.polyfit(years.astype(float), values, deg=1)  # slope, intercept
    slope, intercept = coeff[0], coeff[1]
    def f(y):
        y = int(y)
        if y_min <= y <= y_max:
            return float(pv_by_year[y])
        # 선형 추세 예측 (음수 방지)
        pred = slope * y + intercept
        return float(max(0.0, pred))
    return f

# =========================
# 5) 기본 파라미터
# 4) 기본 파라미터
# =========================
def make_v2g_model_params():
return {
        # PV
        "self_use_ratio": 0.60,            # 자가소비 비율

        # V2G (장비 스펙)
        "self_use_ratio": 0.60,           # 자가소비 비율
        # V2G
"num_v2g_chargers": 6,
"v2g_charger_unit_cost": 25_000_000,
"v2g_daily_discharge_per_charger_kwh": 35,
        "degradation_factor": 0.9,         # 가용/열화 고정 보정
        "degradation_factor": 0.9,
"v2g_operating_days": 300,

        # V2G 운영 정책(연동 파라미터)
        "v2g_share_of_pv_surplus": 0.5,    # PV 잉여 중 V2G로 활용 비율 γ
        "v2g_yearly_degradation": 0.0,     # 연차별 V2G 성능저하율

# 단가(연복리 상승)
"tariff_base_year": 2025,
        "pv_base_price": 160,              # 원/kWh
        "pv_base_price": 160,             # 원/kWh
"v2g_price_gap": 30,
        "price_cagr": 0.0,                 # 테스트를 위해 기본 0으로 둬도 됨

        "price_cagr": 0.043,
# O&M
"om_ratio": 0.015,
}

# =========================
# 6) 현금흐름 빌드 (CSV 기반 PV) — 연도별 V2G 가변화
# 5) 현금흐름 빌드 (CSV 기반 PV)
# =========================
def build_yearly_cashflows_from_csv(install_year: int, current_year: int, p: dict,
                                    pv_value_fn):
    """
    pv_value_fn(year) → 해당 연도의 PV 연간 kWh (외삽 전략 반영)
    - 연도별 PV 잉여전력에 비례해 V2G 방전량을 산정하되,
      장비 스펙이 허용하는 '목표치'를 상한으로 캡.
    """
    # 장비 스펙 기반 '기본 목표치'(연간): 변하지 않는 상한(연차별 성능저하는 아래에서 적용)
    base_target_v2g_kwh = (
        p["num_v2g_chargers"]
        * p["v2g_daily_discharge_per_charger_kwh"]
        * p["v2g_operating_days"]
        * p["degradation_factor"]
    )
                                    pv_kwh_by_year: dict):
    # V2G(연간 고정)
    daily_v2g_kwh = p["num_v2g_chargers"] * p["v2g_daily_discharge_per_charger_kwh"]
    annual_v2g_kwh = daily_v2g_kwh * p["v2g_operating_days"] * p["degradation_factor"]

capex_total = p["num_v2g_chargers"] * p["v2g_charger_unit_cost"]
annual_om_cost = capex_total * p["om_ratio"]

self_use = float(p["self_use_ratio"])
    gamma = float(p.get("v2g_share_of_pv_surplus", 0.5))
    v2g_deg = float(p.get("v2g_yearly_degradation", 0.0))

years = list(range(install_year, current_year + 1))
yearly_cash, cumulative = [], []
pv_revenues, v2g_revenues, om_costs, capex_list = [], [], [], []

    if len(pv_kwh_by_year) == 0:
        raise ValueError("CSV에서 계산된 PV 연간 발전량이 없습니다.")
    min_y, max_y = min(pv_kwh_by_year.keys()), max(pv_kwh_by_year.keys())

cum = 0.0
for i, year in enumerate(years):
        # 연도별 PV kWh (외삽 전략 함수로부터)
        annual_pv_kwh = pv_value_fn(year)
        pv_surplus_y  = annual_pv_kwh * (1 - self_use)

        # 장비 성능 저하 반영
        v2g_cap_y = base_target_v2g_kwh * ((1.0 - v2g_deg) ** i)
        # PV 잉여 중 V2G로 쓰는 비율 γ
        v2g_from_pv_y = pv_surplus_y * gamma
        # 실제 연간 V2G 방전량
        annual_v2g_kwh_y = min(v2g_cap_y, v2g_from_pv_y)

        # 단가 (cagr=0 이어도 OK — 여기선 kWh가 바뀌므로 수입이 달라짐)
        pv_price_y  = price_with_cagr(p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"])
        y_key = min(max(year, min_y), max_y)  # 범위 밖은 경계값 사용(보수적)
        annual_pv_kwh = pv_kwh_by_year[y_key]
        annual_pv_surplus_kwh = annual_pv_kwh * (1 - self_use)

        pv_price_y = price_with_cagr(p["pv_base_price"], p["tariff_base_year"], year, p["price_cagr"])
v2g_price_y = pv_price_y + p["v2g_price_gap"]

        # 수입
        revenue_pv_y  = pv_surplus_y      * pv_price_y
        revenue_v2g_y = annual_v2g_kwh_y  * v2g_price_y
        revenue_pv_y = annual_pv_surplus_kwh * pv_price_y
        revenue_v2g_y = annual_v2g_kwh * v2g_price_y
annual_revenue_y = revenue_pv_y + revenue_v2g_y

        # 비용
        om_y    = annual_om_cost
        om_y = annual_om_cost
capex_y = capex_total if i == 0 else 0

        # 순현금흐름
        cf  = annual_revenue_y - om_y - capex_y
        cf = annual_revenue_y - om_y - capex_y
cum += cf

yearly_cash.append(cf)
@@ -310,16 +221,7 @@ def build_yearly_cashflows_from_csv(install_year: int, current_year: int, p: dic
om_costs.append(om_y)
capex_list.append(capex_y)

    # 보고용 평균
    avg_pv_surplus_kwh = np.mean([pv_value_fn(y) * (1 - self_use) for y in years])
    avg_v2g_kwh = np.mean([
        min(
            base_target_v2g_kwh * ((1.0 - v2g_deg) ** i),
            pv_value_fn(y) * (1 - self_use) * gamma
        )
        for i, y in enumerate(years)
    ])

    avg_pv_surplus_kwh = np.mean([pv_kwh_by_year[min(max(y, min_y), max_y)] * (1 - self_use) for y in years])
return {
"years": years,
"yearly_cash": yearly_cash,
@@ -329,39 +231,42 @@ def build_yearly_cashflows_from_csv(install_year: int, current_year: int, p: dic
"om_costs": om_costs,
"capex_list": capex_list,
"annual_pv_surplus_kwh": avg_pv_surplus_kwh,  # 보고용 평균
        "annual_v2g_kwh": avg_v2g_kwh,                # 보고용 평균
        "annual_v2g_kwh": annual_v2g_kwh,
}

# =========================
# 7) Streamlit App
# 6) Streamlit App
# =========================
def main():
    st.title("V2G + PV 경제성 (CSV 일사합 기반, 연도별 V2G 가변)")
    st.title("V2G 투자 대비 연도별/누적 현금흐름")

    # --- 사이드바: 시스템 & 시뮬레이션 설정 ---
    st.sidebar.header("입력/시스템 설정")
    # --- 사이드바: 데이터 & 시스템 설정 ---
    st.sidebar.header("입력 데이터/시스템 설정")

    # CSV 경로는 UI에 노출하지 않고 고정(같은 폴더 jeju.csv)
    # ① CSV 경로 (사이드바 없이 내부에서 고정)
base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "jeju.csv")
    csv_path = os.path.join(base_dir, "jeju.csv")  # 레포에 올라간 파일명 그대로

    # 모듈/면적/PR
    # ② 모듈/면적/PR
c1, c2 = st.sidebar.columns(2)
    panel_w   = c1.number_input("패널 폭(m)",   value=1.46, step=0.01, format="%.2f")
    panel_h   = c2.number_input("패널 높이(m)", value=0.98, step=0.01, format="%.2f")
    panel_w = c1.number_input("패널 폭(m)", value=1.46, step=0.01, format="%.2f")
    panel_h = c2.number_input("패널 높이(m)", value=0.98, step=0.01, format="%.2f")
n_modules = st.sidebar.number_input("모듈 수(장)", value=250, step=5, min_value=1)

    st.sidebar.caption("PR은 고정 계수(기본) 또는 직접 지정")
    st.sidebar.caption("PR은 고정 계수(기본) 또는 직접 지정(수동) 중 선택")
use_manual_pr = st.sidebar.checkbox("PR 수동 지정", value=False)
    pr_manual = st.sidebar.number_input("PR (0~1)", value=0.78, min_value=0.0, max_value=1.0, step=0.01) if use_manual_pr else None
    if use_manual_pr:
        pr_manual = st.sidebar.number_input("PR (0~1)", value=0.78, min_value=0.0, max_value=1.0, step=0.01)
    else:
        pr_manual = None
pr_base, availability, soiling, inv_eff = 0.82, 0.98, 0.02, 0.97

    # 기간/요금
    # ③ 시뮬레이션 기간/요금/비율
install_year = st.sidebar.number_input("설치 연도", value=2025, step=1)
current_year = st.sidebar.number_input("마지막 연도", value=2045, step=1, min_value=install_year)

params = make_v2g_model_params()
    # V2G 장비 스펙
    # ── 여기서 빠졌던 V2G 항목 다시 추가 ──
params["num_v2g_chargers"] = st.sidebar.number_input(
"V2G 충전기 대수", value=params["num_v2g_chargers"], step=1, min_value=1
)
@@ -371,49 +276,33 @@ def main():
params["v2g_operating_days"] = st.sidebar.number_input(
"연간 운영일 수", value=params["v2g_operating_days"], step=10, min_value=1, max_value=365
)
    # V2G 운영 정책
    params["v2g_share_of_pv_surplus"] = st.sidebar.slider("PV 잉여 중 V2G 활용 비율 γ", 0.0, 1.0, params["v2g_share_of_pv_surplus"], 0.05)
    params["v2g_yearly_degradation"]  = st.sidebar.number_input("V2G 연차별 성능저하율", value=params["v2g_yearly_degradation"], min_value=0.0, max_value=0.2, step=0.01, format="%.2f")
    # (원하면 열화율도 노출 가능)
    # params["degradation_factor"] = st.sidebar.number_input("열화·가용 보정", value=params["degradation_factor"], min_value=0.0, max_value=1.0, step=0.01)

    # 요금/비율
params["self_use_ratio"] = st.sidebar.slider("PV 자가소비 비율", 0.0, 1.0, params["self_use_ratio"], 0.05)
    params["pv_base_price"]  = st.sidebar.number_input("PV 기준단가 (원/kWh)", value=params["pv_base_price"], step=5, min_value=0)
    params["price_cagr"]     = st.sidebar.number_input("전력단가 연평균 상승률", value=params["price_cagr"], step=0.001, format="%.3f")
    params["pv_base_price"] = st.sidebar.number_input("PV 기준단가 (원/kWh)", value=params["pv_base_price"], step=5, min_value=0)
    params["price_cagr"] = st.sidebar.number_input("전력단가 연평균 상승률", value=params["price_cagr"], step=0.001, format="%.3f")

# 재무 지표용 할인율
discount_rate = st.sidebar.number_input("할인율(연)", value=0.08, min_value=0.0, max_value=0.5, step=0.005, format="%.3f")

# --- CSV 로드 & PV 연간 kWh 계산 ---
irr_df = load_irradiance_csv(csv_path)
    pv_by_year_raw, area_m2, PR_used = compute_pv_kwh_by_year(
    pv_by_year, area_m2, PR_used = compute_pv_kwh_by_year(
irr_df,
panel_width_m=panel_w, panel_height_m=panel_h, n_modules=int(n_modules),
pr_base=pr_base, availability=availability, soiling=soiling, inv_eff=inv_eff,
pr_manual=pr_manual
)

    # --- CSV 범위 밖 외삽 전략 선택 ---
    strategy = st.sidebar.selectbox(
        "CSV 범위 밖 외삽 전략",
        options=["trend", "clamp", "mean5", "repeat"],
        format_func=lambda s: {
            "trend": "선형 추세 외삽(기본)",
            "clamp": "경계값 고정(보수적)",
            "mean5": "최근 5년 평균",
            "repeat":"주기 반복"
        }[s],
        index=0
    )
    pv_value_fn = make_pv_forecaster_from_series(pv_by_year_raw, strategy=strategy)

# --- 현금흐름 계산 ---
    cf = build_yearly_cashflows_from_csv(install_year, current_year, params, pv_value_fn)
    cf = build_yearly_cashflows_from_csv(install_year, current_year, params, pv_by_year)
years = cf["years"]; yearly_cash = cf["yearly_cash"]; cumulative = cf["cumulative"]

# 손익분기 연도
break_even_year = next((y for y, cum in zip(years, cumulative) if cum >= 0), None)

    # 탄소절감량(kgCO2e) — 보고용 평균 kWh로 계산
    # 탄소절감량(kgCO2e)
clean_kwh_per_year = cf["annual_pv_surplus_kwh"] + cf["annual_v2g_kwh"]
total_clean_kwh = clean_kwh_per_year * len(years)
total_co2_kg = total_clean_kwh * EMISSION_FACTOR_KG_PER_KWH
@@ -496,11 +385,7 @@ def main():
st.dataframe(df_table, use_container_width=True)

# 참고 정보
    st.caption(
        f"총 모듈 면적: {area_m2:.1f} m² | 사용 PR: {PR_used:.3f} | "
        f"CSV 연도 범위: {min(pv_by_year_raw.keys())}~{max(pv_by_year_raw.keys())} | "
        f"외삽 전략: {strategy}"
    )
    st.caption(f"총 모듈 면적: {area_m2:.1f} m² | 사용 PR: {PR_used:.3f} | CSV 연도 범위: {min(pv_by_year.keys())}~{max(pv_by_year.keys())}")

if __name__ == "__main__":
main()
