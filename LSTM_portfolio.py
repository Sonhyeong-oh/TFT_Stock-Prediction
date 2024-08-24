# LSTM을 이용한 시가 예측 및 시각화

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
import numpy as np


import os
import warnings
import matplotlib.pyplot as plt
import torch.backends
warnings.filterwarnings("ignore") 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir("../../..")
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch

acc = 'cpu'

# 최대한의 시작 날짜 설정
start_date = '2000-01-01'
end_date = '2024-07-26'

# S&P 500 ETF 데이터 불러오기
snp500 = yf.download('SPY', start=start_date, end=end_date)['Adj Close']

# 미국 실업률 데이터 불러오기
unemployment_rate = web.DataReader('UNRATE', 'fred', start_date, end_date)

# 기준금리 데이터 불러오기
interest_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)

# 장단기 금리차 데이터 불러오기
term_spread = web.DataReader('T10Y2Y', 'fred', start_date, end_date)

# TIPS 데이터 불러오기
tips = web.DataReader('DFII10', 'fred', start_date, end_date)

# 하이일드 스프레드 데이터 불러오기
high_yield_spread = web.DataReader('BAMLH0A0HYM2', 'fred', start_date, end_date)

# WTI 원유 데이터 불러오기
wti = web.DataReader('DCOILWTICO', 'fred', start_date, end_date)




# 데이터를 거래일 기준으로 리샘플링 및 NA 값 처리
snp500 = snp500.asfreq('B').fillna(method='ffill')
unemployment_rate = unemployment_rate.asfreq('B').fillna(method='ffill')
interest_rate = interest_rate.asfreq('B').fillna(method='ffill')
term_spread = term_spread.asfreq('B').fillna(method='ffill')
tips = tips.asfreq('B').fillna(method='ffill')
high_yield_spread = high_yield_spread.asfreq('B').fillna(method='ffill')
wti = wti.asfreq('B').fillna(method='ffill')




# 원본 데이터프레임 결합 (data1)
data1 = pd.concat([snp500, unemployment_rate, interest_rate, term_spread, tips, high_yield_spread, wti], axis=1)
data1.columns = ['SNP500', 'Unemployment_Rate', 'Interest_Rate', 'Term_Spread', 'TIPS', 'High_Yield_Spread', 'WTI']
data1 = data1.dropna()


# 로그 변화율 계산 함수
def calculate_log_return(series):
    return np.log(series / series.shift(1))

# 단순 변화율 계산 함수 (S&P 500, WTI에 사용)
def calculate_percentage_change(series):
    return series.pct_change() * 100

# 단순 변화량 계산 함수 (나머지 변수에 사용)
def calculate_difference(series):
    return series.diff()

# 변화율 및 변화량 계산
data2 = pd.DataFrame(index=data1.index)
data2['SNP500_log_return'] = calculate_log_return(data1['SNP500'])  # S&P 500 로그 변화율
data2['TIPS_diff'] = calculate_difference(data1['TIPS'])  # TIPS 변화량
data2['Unemployment_Rate_diff'] = calculate_difference(data1['Unemployment_Rate'])  # 실업률 변화량
data2['Interest_Rate_diff'] = calculate_difference(data1['Interest_Rate'])  # 금리 변화량
data2['Term_Spread_diff'] = calculate_difference(data1['Term_Spread'])  # 장단기 금리차 변화량
data2['High_Yield_Spread_diff'] = calculate_difference(data1['High_Yield_Spread'])  # 하이일드 스프레드 변화량
data2['WTI_diff'] = calculate_percentage_change(data1['WTI'])  # WTI 단순 변화율

# NaN 값 제거
data2 = data2.dropna()

# 원본 데이터와 계산된 변화율, 변화량 합침
data_combined = pd.concat([data1, data2], axis=1)

# 결측값 제거
data_combined = data_combined.dropna()

# 주별 리샘플링
data_weekly = data_combined.resample('W').last()

# 주별 로그 변화율 및 단순 변화율, 변화량 다시 계산
data_weekly['SNP500_log_return'] = calculate_log_return(data_weekly['SNP500'])  # S&P 500 로그 변화율
data_weekly['TIPS_diff'] = calculate_difference(data_weekly['TIPS'])  # TIPS 변화량
data_weekly['Unemployment_Rate_diff'] = calculate_difference(data_weekly['Unemployment_Rate'])  # 실업률 변화량
data_weekly['Interest_Rate_diff'] = calculate_difference(data_weekly['Interest_Rate'])  # 금리 변화량
data_weekly['Term_Spread_diff'] = calculate_difference(data_weekly['Term_Spread'])  # 장단기 금리차 변화량
data_weekly['High_Yield_Spread_diff'] = calculate_difference(data_weekly['High_Yield_Spread'])  # 하이일드 스프레드 변화량
data_weekly['WTI_diff'] = calculate_percentage_change(data_weekly['WTI'])  # WTI 단순 변화율

# NaN 값 제거
data_weekly = data_weekly.dropna()

# 앞의 22개 데이터 잘라내기
data_weekly = data_weekly.iloc[22:]

# 노이즈 제거 및 최소-최대 정규화 적용
wv = 'coif5'
scaler = StandardScaler()
data_final = pd.DataFrame(index=data_combined.index)

for column in data_combined.columns:
    # 노이즈 제거 과정
    signal = data_combined[column].values
    N = len(signal)
    lev = pywt.dwt_max_level(N, wv)
    coeffs = pywt.wavedec(signal, wv, level=lev)
    D1 = coeffs[-1]
    sigma_med = np.median(np.abs(D1)) / 0.6745
    lambda_U = sigma_med * np.sqrt(2 * np.log(N))
    coeffs = [pywt.threshold(c, lambda_U, mode='garrote') for c in coeffs]
    denoised_signal = pywt.waverec(coeffs, wv)

    # 원래 신호의 길이에 맞추기
    denoised_signal = denoised_signal[:N]

    # 최소-최대 정규화 적용 후 data_final에 저장
    data_final[column] = scaler.fit_transform(denoised_signal.reshape(-1, 1)).flatten()


data2 = data_final

# 인덱스를 열로 복사하여 Date 열 생성
data2['Date'] = data2.index

# Date 열의 데이터 타입을 datetime으로 변환
data2['Date'] = pd.to_datetime(data2['Date'])

# 데이터 생성
data = data2
data['Date'] = pd.to_datetime(data['Date'])

data.to_excel('yfinance_data.xlsx', index = False)

# matplot 한글 출력 코드
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

data = pd.read_excel('C:/Users/yfinance_data.xlsx')
dates = pd.to_datetime(data['Date'])
original_open = data['SNP500'].values

# 훈련 데이터 추출 변수 ('Date' 제외)
cols = list(data)[0:14]

# new dataframe with only training data - 5 columns
# 새로운 훈련 데이터셋

stock_data = data[cols].astype(float)

# 시계열 분석이기 때문에 train_test_split 사용 X
n_train = int(0.6*stock_data.shape[0])
m_train = int(0.8*stock_data.shape[0])
train_data_scaled = stock_data[0: n_train]
train_dates = dates[0: n_train]
test_data_scaled = stock_data[n_train:m_train]
test_dates = dates[n_train:m_train]

new_n = int(0.2*stock_data.shape[0])
new_m = int(0.8*stock_data.shape[0])
new_train = stock_data[new_n: new_m]
new_train_dates = dates[new_n: new_m]
new_test = stock_data[new_m: ]
new_test_dates = dates[new_m: ]
# print(test_dates.head(5))

# LSTM을 위한 데이터 재구성
pred_days = 1  # 예측 일수
seq_len = 14   # 시퀀스 길이 = 미래 예측을 위한 과거 일수
input_dim = 14  # 입력 (열) 차원 = ['Open', 'High', 'Low', 'Close', 'Volume']

# X = 학습 데이터 시퀀스 / Y = 예측할 실제 값
trainX = []
trainY = []
testX = []
testY = []

print(train_data_scaled.shape[1])

for i in range(seq_len, n_train-pred_days +1):
    # scaled 데이터 중 i-seq_len 행의 모든 데이터(= 모든 열)를 리스트에 추가
    trainX.append(train_data_scaled.iloc[i - seq_len:i, 0:train_data_scaled.shape[1]])
    trainY.append(train_data_scaled.iloc[i + pred_days - 1:i + pred_days, 0])

for i in range(seq_len, len(test_data_scaled)-pred_days +1):
    testX.append(test_data_scaled.iloc[i - seq_len:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled.iloc[i + pred_days - 1:i + pred_days, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)

new_trainX = []
new_trainY = []
new_testX = []
new_testY = []

for i in range(seq_len, n_train-pred_days +1):
    # scaled 데이터 중 i-seq_len 행의 모든 데이터(= 모든 열)를 리스트에 추가
    new_trainX.append(new_train.iloc[i - seq_len:i, 0:new_train.shape[1]])
    new_trainY.append(new_train.iloc[i + pred_days - 1:i + pred_days, 0])

for i in range(seq_len, len(new_test)-pred_days +1):
    new_testX.append(new_test.iloc[i - seq_len:i, 0:new_test.shape[1]])
    new_testY.append(new_test.iloc[i + pred_days - 1:i + pred_days, 0])

new_trainX, new_trainY = np.array(new_trainX), np.array(new_trainY)
new_testX, new_testY = np.array(new_testX), np.array(new_testY)
# print(trainX.shape, trainY.shape)
# print(testX.shape, testY.shape)

# R2 점수 함수
# r2점수 = 예측된 값이 실제 값과 얼마나 잘 일치하는지 설명하는 지표, 1에 가까울 수록 좋은 모델
# 1 - (sum(관측값 - 예측값)^2 / sum(관측값 - 타겟 데이터 평균값)^2)
# 1 - (sum(test_target - target_pred)^2 / sum(test_target - np.mean(test_target))^2)
def r2_metric(trainX, trainY):
    # 실제 값과 예측 값의 평균 계산
    res = tf.keras.backend.sum(tf.keras.backend.square(trainX - trainY))  # 잔차 제곱합
    total = tf.keras.backend.sum(tf.keras.backend.square(trainX - tf.keras.backend.mean(trainX)))  # 총 제곱합
    r2 = 1 - res / (total + tf.keras.backend.epsilon())  # R2 점수 계산
    return r2

# LSTM 모델
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2]), # (seq length, input dimension)
               return_sequences=True))
model.add(tf.keras.layers.LSTM(32, return_sequences=False))
model.add(tf.keras.layers.Dense(trainY.shape[1]))

model.summary()

# 학습률 설정
learning_rate = 0.01
# 최적화 함수 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# 모델 컴파일(최적화 함수 : 아담, 손실함수 : mse, 측정값 = R제곱 점수)
# mse = 평균 제곱 오차, 실제 값과 예측 값 사이의 차이를 제곱한 값의 평균
# 값이 작을 수록 오차가 작음
model.compile(optimizer=optimizer, loss='mse', metrics = [r2_metric])


# 모델 학습

# comment : 모델 학습 후 저장된 최적화 파라미터가 있을 경우 그 값을 불러오는 코드입니다.
#           지금은 성능 확인이 목표이므로 주석처리 하였습니다.
# Try to load weights
# try:
#     model.load_weights('lstm_weights.weights.h5')
#     print("Loaded model weights from disk")
# except:
#     print("No weights found, training model from scratch")

# 모델 파라미터를 저장하기 위해 변수 지정
history = model.fit(trainX, trainY, epochs=100, batch_size=32,
                validation_split=0.1, verbose=1)
# 모델 파라미터 저장
model.save_weights('lstm_weights.weights.h5')

plt.plot(history.history['loss'], label='학습 손실(mse)', color = 'black')
plt.plot(history.history['val_loss'], label='검증 손실(mse)', color = 'gray')
plt.plot(history.history['r2_metric'], label = '학습 r2', color = 'blue')
plt.plot(history.history['val_r2_metric'], label = '검증 r2', color = 'red')
plt.legend()
plt.show()


# ------------------------------------------- 모델 학습 ------------------------------------------------
prediction = model.predict(testX)

# prediction 데이터의 시각화를 위한 평균 데이터셋 생성
# scaler.mean_ : prediction 데이터의 평균값(열 기준)으로 1차원 데이터 생성
# [np,newaixs, :] : 1차원 데이터를 (1, n) 데이터로 변환
# prediction.shape[0], axis = 0 : prediction 데이터셋의 행 수만큼 반복, 행으로 추가 (아래로 데이터 추가)
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

# prediction 데이터를 mean_values_pred의 첫번째 열로 교체
# np.squeeze : 크기가 1인 벡터를 제거 = (83,1) -> (83,)
mean_values_pred[:, 0] = np.squeeze(prediction)

# 표준화한 데이터를 원데이터로 변환
y_pred = scaler.inverse_transform(mean_values_pred)[:,0]

# testY 데이터 시각화를 위한 평균 데이터셋 생성
mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], new_testY.shape[0], axis=0)

# testY 데이터를 mean_values_testY의 첫번쨰 열로 교체
mean_values_testY[:, 0] = np.squeeze(new_testY)

# 표준화 데이터를 원데이터로 변환
testY_original = scaler.inverse_transform(mean_values_testY)[:,0]

# plotting
plt.figure(figsize=(14, 5))

# plot original 'Open' prices

# plot actual vs predicted
# plt.plot(dates, original_open, color='green', label='학습 시가')
plt.plot(test_dates[seq_len-1:], testY_original, color='blue', label='실제 시가')
plt.plot(test_dates[seq_len:], y_pred, color='red', linestyle='--', label='예측 시가')
plt.xlabel('날짜')
plt.ylabel('시가')
plt.title('학습, 실제, 예측 시가')
plt.legend()
plt.show()

# Calculate the start and end indices for the zoomed plot
zoom_start = len(test_dates) - 50
zoom_end = len(test_dates)

# Create the zoomed plot
plt.figure(figsize=(14, 5))

# Adjust the start index for the testY_original and y_pred arrays
adjusted_start = zoom_start - seq_len

plt.plot(test_dates[zoom_start:zoom_end],
         testY_original[adjusted_start:zoom_end - zoom_start + adjusted_start],
         color='blue',
         label='실제 시가')

plt.plot(test_dates[zoom_start:zoom_end],
         y_pred[adjusted_start:zoom_end - zoom_start + adjusted_start ],
         color='red',
         linestyle='--',
         label='예측 시가')

plt.xlabel('날짜')
plt.ylabel('시가')
plt.title('Zoomed In 실제 vs 예측 시가')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(testY_original[:1107], y_pred)
print(f'MSE: {mse}')

# r2점수 = 예측된 값이 실제 값과 얼마나 잘 일치하는지 설명하는 지표, 1에 가까울 수록 좋은 모델
# 1 - (sum(관측값 - 예측값)^2 / sum(관측값 - 타겟 데이터 평균값)^2)
# 1 - (sum(test_target - target_pred)^2 / sum(test_target - np.mean(test_target))^2)
from sklearn.metrics import r2_score
r2 = r2_score(testY_original[:1107], y_pred)
print('R squared score:',r2)


# ------------------------------ 포트 폴리오 예측----------------------------------
# 모델 파라미터를 저장하기 위해 변수 지정
history = model.fit(new_trainX, new_trainY, epochs=100, batch_size=32,
                validation_split=0.1, verbose=1)
# 모델 파라미터 저장
model.save_weights('lstm_weights.weights.h5')

plt.plot(history.history['loss'], label='학습 손실(mse)', color = 'black')
plt.plot(history.history['val_loss'], label='검증 손실(mse)', color = 'gray')
plt.plot(history.history['r2_metric'], label = '학습 r2', color = 'blue')
plt.plot(history.history['val_r2_metric'], label = '검증 r2', color = 'red')
plt.legend()
plt.show()


# 예측
# 예측
prediction = model.predict(new_testX)

# prediction 데이터의 시각화를 위한 평균 데이터셋 생성
# scaler.mean_ : prediction 데이터의 평균값(열 기준)으로 1차원 데이터 생성
# [np,newaixs, :] : 1차원 데이터를 (1, n) 데이터로 변환
# prediction.shape[0], axis = 0 : prediction 데이터셋의 행 수만큼 반복, 행으로 추가 (아래로 데이터 추가)
mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0)

# prediction 데이터를 mean_values_pred의 첫번째 열로 교체
# np.squeeze : 크기가 1인 벡터를 제거 = (83,1) -> (83,)
mean_values_pred[:, 0] = np.squeeze(prediction)

# 표준화한 데이터를 원데이터로 변환
y_pred = scaler.inverse_transform(mean_values_pred)[:,0]

# testY 데이터 시각화를 위한 평균 데이터셋 생성
mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], new_testY.shape[0], axis=0)

# testY 데이터를 mean_values_testY의 첫번쨰 열로 교체
mean_values_testY[:, 0] = np.squeeze(new_testY)

# 표준화 데이터를 원데이터로 변환
testY_original = scaler.inverse_transform(mean_values_testY)[:,0]

# plotting
plt.figure(figsize=(14, 5))

# plot original 'Open' prices

# plot actual vs predicted
# plt.plot(dates, original_open, color='green', label='학습 시가')
plt.plot(new_test_dates[seq_len:], testY_original, color='blue', label='실제 시가')
plt.plot(new_test_dates[seq_len:], y_pred, color='red', linestyle='--', label='예측 시가')
plt.xlabel('날짜')
plt.ylabel('시가')
plt.title('학습, 실제, 예측 시가')
plt.legend()
plt.show()

# Calculate the start and end indices for the zoomed plot
zoom_start = len(test_dates) - 50
zoom_end = len(test_dates)

# Create the zoomed plot
plt.figure(figsize=(14, 5))

# Adjust the start index for the testY_original and y_pred arrays
adjusted_start = zoom_start - seq_len

plt.plot(new_test_dates[zoom_start:zoom_end],
         testY_original[adjusted_start:zoom_end - zoom_start + adjusted_start],
         color='blue',
         label='실제 시가')

plt.plot(new_test_dates[zoom_start:zoom_end],
         y_pred[adjusted_start:zoom_end - zoom_start + adjusted_start ],
         color='red',
         linestyle='--',
         label='예측 시가')

plt.xlabel('날짜')
plt.ylabel('시가')
plt.title('Zoomed In 실제 vs 예측 시가')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(testY_original, y_pred)
print(f'MSE: {mse}')

# r2점수 = 예측된 값이 실제 값과 얼마나 잘 일치하는지 설명하는 지표, 1에 가까울 수록 좋은 모델
# 1 - (sum(관측값 - 예측값)^2 / sum(관측값 - 타겟 데이터 평균값)^2)
# 1 - (sum(test_target - target_pred)^2 / sum(test_target - np.mean(test_target))^2)
from sklearn.metrics import r2_score
r2 = r2_score(testY_original, y_pred)
print('R squared score:',r2)

# 코드 출처 : https://www.deepcampus.kr/266
# 참고 논문 : 김수현(2020), LSTM 기반 모형의 주식시장 예측 분석, Journal of the Korean Data Analysis Society