# LSTM을 이용한 시가 예측 및 시각화

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# matplot 한글 출력 코드
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# csv 파일 불러오기
stock_data = pd.read_csv("C:/Users/daily/Desktop/train.csv")

# Open 값을 데이터 시각화 시 사용하기 위해 저장
original_open = stock_data['Open'].values

# 날짜 데이터 문자열(type : str)을 datetime 객체(type: datetime64[ns])로 변환
# print(type(stock_data['Date'].iloc[0])) = <class 'str'>
# print(type(dates.iloc[0])) = <class 'pandas._libs.tslibs.timestamps.Timestamp'>
dates = pd.to_datetime(stock_data['Date'])

# 훈련 데이터 추출 변수 ('Date' 제외)
cols = list(stock_data)[1:6]

# new dataframe with only training data - 5 columns
# 새로운 훈련 데이터셋
stock_data = stock_data[cols].astype(float)

# 데이터셋 표준화
scaler = StandardScaler()
scaler = scaler.fit(stock_data)
stock_data_scaled = scaler.transform(stock_data)

# 훈련 데이터와 테스트 데이터 분리 (9:1)
# 시계열 분석이기 때문에 train_test_split 사용 X
n_train = int(0.9*stock_data_scaled.shape[0])
train_data_scaled = stock_data_scaled[0: n_train]
train_dates = dates[0: n_train]

test_data_scaled = stock_data_scaled[n_train:]
test_dates = dates[n_train:]
# print(test_dates.head(5))

# LSTM을 위한 데이터 재구성
pred_days = 1  # 예측 일수
seq_len = 14   # 시퀀스 길이 = 미래 예측을 위한 과거 일수
input_dim = 5  # 입력 (열) 차원 = ['Open', 'High', 'Low', 'Close', 'Volume']

# X = 학습 데이터 시퀀스 / Y = 예측할 실제 값
trainX = []
trainY = []
testX = []
testY = []

for i in range(seq_len, n_train-pred_days +1):
    # scaled 데이터 중 i-seq_len 행의 모든 데이터(= 모든 열)를 리스트에 추가
    trainX.append(train_data_scaled[i - seq_len:i, 0:train_data_scaled.shape[1]])
    trainY.append(train_data_scaled[i + pred_days - 1:i + pred_days, 0])

for i in range(seq_len, len(test_data_scaled)-pred_days +1):
    testX.append(test_data_scaled[i - seq_len:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled[i + pred_days - 1:i + pred_days, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)

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


# 예측
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
mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)

# testY 데이터를 mean_values_testY의 첫번쨰 열로 교체
mean_values_testY[:, 0] = np.squeeze(testY)

# 표준화 데이터를 원데이터로 변환
testY_original = scaler.inverse_transform(mean_values_testY)[:,0]

# plotting
plt.figure(figsize=(14, 5))

# plot original 'Open' prices
plt.plot(dates, original_open, color='green', label='학습 시가')

# plot actual vs predicted
plt.plot(test_dates[seq_len:], testY_original, color='blue', label='실제 시가')
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

print('정확도 : %.4f' % (model.evaluate(testX, testY)[1]))
# 코드 출처 : https://www.deepcampus.kr/266
# 데이터 출처 : https://www.kaggle.com/competitions/netflix-stock-prediction/data?select=sample_submission.csv