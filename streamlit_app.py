import streamlit as st
import pandas as pd
import numpy as np # 로짓-확률 변환을 위해 필요

# 최종 모델(Model 9)에 포함된 변수 목록
final_variables = [
    'Age', 'BusinessTravel', 'DistanceFromHome', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked',
    'OverTime', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# 각 변수의 비표준화 계수(B) 
coefficients = {
    '(Constant)': 0.713, # 상수항 B 값
    'Age': -0.004,
    'BusinessTravel': 0.082, # 출장 변수 B 값 (숫자 코딩 기준)
    'DistanceFromHome': 0.004,
    'EnvironmentSatisfaction': -0.040,
    'JobInvolvement': -0.065,
    'JobLevel': -0.024,
    'JobSatisfaction': -0.037,
    'NumCompaniesWorked': 0.017,
    'OverTime': 0.204, # OverTime_Num 변수 B 값
    'RelationshipSatisfaction': -0.022,
    'StockOptionLevel': -0.055,
    'TotalWorkingYears': -0.004,
    'TrainingTimesLastYear': -0.012,
    'WorkLifeBalance': -0.025,
    'YearsAtCompany': 0.006,
    'YearsInCurrentRole': -0.010,
    'YearsSinceLastPromotion': 0.012,
    'YearsWithCurrManager': -0.010
}

# --- 2. 앱 인터페이스 구성 ---
st.title("🧑‍💼 직원 이직 확률 예측")
st.write("직원의 정보를 입력하면 이직 확률을 예측합니다.")

# 사용자 입력을 받을 딕셔너리 생성
inputs = {}

st.sidebar.header("직원 정보 입력")

# 각 변수에 대한 입력 위젯 생성
inputs['Age'] = st.sidebar.slider("나이 (Age)", 18, 60, 30) # 최소, 최대, 기본값

# BusinessTravel
travel_options = {1: '출장 없음', 2: '가끔 출장', 3: '자주 출장'}
selected_travel_text = st.sidebar.selectbox(
    "출장 빈도 (BusinessTravel)",
    options=list(travel_options.values()),
    index=1 # 기본값을 '가끔 출장'으로
)
# 선택된 텍스트를 숫자로 변환
inputs['BusinessTravel'] = [k for k, v in travel_options.items() if v == selected_travel_text][0]


inputs['DistanceFromHome'] = st.sidebar.slider("집과의 거리 (km)", 1, 30, 5)
inputs['EnvironmentSatisfaction'] = st.sidebar.select_slider(
    "환경 만족도 (1: 낮음 ~ 4: 높음)", options=[1, 2, 3, 4], value=3)
inputs['JobInvolvement'] = st.sidebar.select_slider(
    "직무 몰입도 (1: 낮음 ~ 4: 높음)", options=[1, 2, 3, 4], value=3)
inputs['JobLevel'] = st.sidebar.select_slider(
    "직급 (1 ~ 5)", options=[1, 2, 3, 4, 5], value=2)
inputs['JobSatisfaction'] = st.sidebar.select_slider(
    "직무 만족도 (1: 낮음 ~ 4: 높음)", options=[1, 2, 3, 4], value=3)
inputs['NumCompaniesWorked'] = st.sidebar.slider("타 회사 근무 경력 (횟수)", 0, 10, 2)

# OverTime 처리 (숫자 코딩: 0=No, 1=Yes)
overtime_option = st.sidebar.radio("초과근무 여부 (OverTime)", ('No', 'Yes'), index=0)
inputs['OverTime'] = 1 if overtime_option == 'Yes' else 0

inputs['RelationshipSatisfaction'] = st.sidebar.select_slider(
    "관계 만족도 (1: 낮음 ~ 4: 높음)", options=[1, 2, 3, 4], value=3)
inputs['StockOptionLevel'] = st.sidebar.select_slider(
    "스톡옵션 수준 (0 ~ 3)", options=[0, 1, 2, 3], value=0)
inputs['TotalWorkingYears'] = st.sidebar.slider("총 근무 연수 (년)", 0, 40, 5)
inputs['TrainingTimesLastYear'] = st.sidebar.slider("최근 1년 교육 횟수", 0, 6, 2)
inputs['WorkLifeBalance'] = st.sidebar.select_slider(
    "워라밸 만족도 (1: 낮음 ~ 4: 높음)", options=[1, 2, 3, 4], value=3)
inputs['YearsAtCompany'] = st.sidebar.slider("현 직장 근속 년수", 0, 40, 3)
inputs['YearsInCurrentRole'] = st.sidebar.slider("현 직무 근속 년수", 0, 20, 2)
inputs['YearsSinceLastPromotion'] = st.sidebar.slider("승진 후 경과 년수", 0, 20, 1)
inputs['YearsWithCurrManager'] = st.sidebar.slider("현 관리자와 근무 년수", 0, 20, 2)


# --- 3. 이직 확률 계산 ---
logit = coefficients['(Constant)']
for var in final_variables:
    if var in inputs and var != '(Constant)': # 상수항은 이미 더했으므로 제외
        logit += coefficients[var] * inputs[var]

# 로짓을 확률로 변환 (Sigmoid 함수 사용)
probability = 1 / (1 + np.exp(-logit))

# --- 4. 결과 표시 ---
st.subheader("📊 예측 결과")
probability_percent = probability * 100
st.metric(label="이직 확률", value=f"{probability_percent:.2f}%")

# 확률에 따른 위험도 표시
if probability_percent >= 30:
    st.error("🚨 이직 위험 높음")
elif probability_percent >= 15:
    st.warning("⚠️ 이직 위험 보통")
else:
    st.success("✅ 이직 위험 낮음")
