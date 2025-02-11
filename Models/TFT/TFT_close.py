# TFT 모델을 이용한 종가 예측
# 코드 실행 프로그램을 관리자 권한으로 실행할 것
# python 3.9.19 환경에서 실행

import os
import warnings
import matplotlib.pyplot as plt
import torch.backends

# 경고 메세지가 출력되지 않도록 만듦.
warnings.filterwarnings("ignore") 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir("../../..")

from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters



# 데이터 생성
data = pd.read_excel("C:/Users/daily/Desktop/data2_normalized.xlsx")
data['Date'] = pd.to_datetime(data['Date'])

data["time_idx"] = [i for i in range(1, len(data['Date'])+1)]

# year로 범주형 변수 생성
data['year'] = data.Date.dt.year.astype(str).astype('category')
# 월 별로 분석 시 data['month'] = data.Date.dt.month.astype(str).astype('category') 로 수정

# data['Open'] = data.Open.astype(float)
# data['High'] = data.High.astype(float)
# data['Low'] = data.Low.astype(float)
# data['Volume'] = data.Volume.astype(float)
# data['Close'] = data.Close.astype(float)

# 월 별로 분석 시 max_prediction_length와 max_encoder_length의 길이 조정
max_prediction_length = 30 # 예측 일수
max_encoder_length = 100 # 학습하는 과거 데이터 일수
training_cutoff = data["time_idx"].max() - max_prediction_length # time_idx의 최댓값 - 예측 일수 -> 

# 인코더 : 시계열 데이터의 초기 부분을 입력 받아 특징 추출
# 디코더 : 인코더가 생성한 벡터 입력 받아 원하는 출력 시퀀스 생성 (다음 단계 예측)

# training 시계열 데이터 생성
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff], # data 중 time_idx가 training_cutoff 이하인 데이터만 추출
    allow_missing_timesteps=True, # 누락된 시계열 구간 허용
    time_idx="time_idx", # 시간 인덱스 열 설정
    target="SNP500", # 타겟 열 설정
    group_ids=['year'], # 그룹화할 데이터 설정(범주형 변수일 것) / 월 별 분석 시 'month'로 변경
    min_encoder_length=max_encoder_length // 2,  # 인코더의 최소 길이 설정
    max_encoder_length=max_encoder_length, # 인코더 최대 길이 설정
    min_prediction_length=1, # 모델이 예측할 기간 -> 모델이 한 번의 예측에서 최소한으로 예측할 시간 간격
    max_prediction_length=max_prediction_length,  # 모델이 한 번의 예측에서 최대로 예측할 수 있는 시간 간격
    time_varying_known_categoricals=['year'], # 값을 알고 있는 범주형 동적변수 / 월 별 분석 시 'month'로 변경
    time_varying_known_reals=['time_idx'], # 값을 알고 있는 연속형 동적 변수 지정
    time_varying_unknown_categoricals=[], # 미지 범주형 동적변수 지정
    time_varying_unknown_reals=['Unemployment_Rate', 'Interest_Rate', 'Term_Spread', 'TIPS', 'High_Yield_Spread'], # 미지 연속형 동적변수 지정
    target_normalizer=GroupNormalizer( # groups = []로 지정한 범주형 변수에 따라 그룹을 나눠 정규화 수행
        groups=[], transformation='relu'   # transformation = 활성화 함수 지정 : 활성화 함수로 데이터 처리 후 정규화
    ),
    add_relative_time_idx=True, # 상대적인 시간 인덱스(relative time index)를 데이터셋에 추가
    add_target_scales=True, # 타겟 변수의 스케일(범위 , 크기)을 추가
    add_encoder_length=True # 인코더의 길이(encoder length)를 데이터셋에 추가
)

# 유효성 검증 세트 생성
# training 학습 데이터셋을 기반으로 새로운 데이터셋 생성
# training = 학습 데이터셋
# data = 기존 데이터, 검증 & 테스트를 위해 사용
# perdict = True : 예측용 데이터셋으로 설정
# stop_randomization = True : 데이터셋 생성 과정 중 임의화 중단, 학습 데이터 셋에는 임의화가 일어나지만 검증&테스트 단계에서는 순서 유지하여 성능 평가
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)


# create dataloaders for model
# 한 번에 모델에 입력되는 데이터 묶음 개수 (32 to 128개로 설정)
batch_size = 128
# training 데이터셋을 dataloader 형태로 변환
# train = True : 학습용 데이터로 설정
# num_workers = 데이터 로딩을 위해 사용할 병렬 작업 수 설정 -> 0 : 별도의 병렬 처리 없이 데이터 처리
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, shuffle =False)
# 검증 데이터셋을 dataloader 형태로 변환
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0, shuffle = False)


# 최적의 하이퍼 파라미터 탐색
opt_hp = optimize_hyperparameters(
    train_dataloader, # 학습 데이터
    val_dataloader, # 검증 데이터
    model_path= 'optuna_test', # 최적화 결과 저장 장소 지정
    n_trials= 10, # trial 횟수
    max_epochs= 10, # 한 trial에서 시행할 에포크 수
    # 하이퍼 파라미터 탐색 범위 지정
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 64),
    attention_head_size_range=(2, 4),
    hidden_continuous_size_range=(8, 64),
    learning_rate_range= (0.001, 0.1),
    dropout_range= (0.1, 0.3),
    # limit_train_batches = 학습 시 사용할 batch의 수 제한
    # log_every_n_steps = 로그를 기록할 간격 지정
    # accelerator = cpu 이용하여 학습
    trainer_kwargs=dict(limit_train_batches = 30, log_every_n_steps = 15, accelerator = 'cpu'), 
    reduce_on_plateau_patience = 4, # trial을 n번 시행했음에도 손실값이 개선되지 않으면 탐색 중지
    use_learning_rate_finder = False, # 학습 시 learning rate 탐색 여부 결정, learning_rate도 같이 탐색하기 때문에 실행 시간 단축을 위해 False 지정
    timeout = 5400*4 # 최대 실행 시간
)

opt_gcv = opt_hp.best_trial.params['gradient_clip_val']
opt_hs = opt_hp.best_trial.params['hidden_size']
opt_ahs = opt_hp.best_trial.params['attention_head_size']
opt_hcs = opt_hp.best_trial.params['hidden_continuous_size']
opt_lr = opt_hp.best_trial.params['learning_rate']
opt_dpt = opt_hp.best_trial.params['dropout']

print('optimized gradient clip val : %f' %opt_gcv)
print('optimized hidden size : %f' %opt_hs)
print('optimized attention head size : %f' %opt_ahs)
print('optimized hidden continuous size : %f' %opt_hcs)
print('optimized learning rate : %f' %opt_lr)
print('optimized dropout : %f' %opt_dpt)



# pl.seed_everything() = 실험의 재현성을 위해 사용되는 함수, 매번 동일한 시드값을 사용하게 함(랜덤성 제어)
pl.seed_everything(42)

import tensorflow as tf

# 학습 과정 중 검증 손실(val_loss)를 모니터링, 성능 개선 안 되었을 시 학습 중지
# min_delta = 개선된 것으로 간주할 최소 손실 변화
# patience = 성능이 개선되지 않은 상태를 얼마나 참을 수 있는지를 나타내는 숫자
# verbose = True로 설정 시 EarlyStopping이 각 조건 충족 시 메시지를 출력
# mode = 모니터링 지표의 최소화(min) 또는 최대화(max)를 나타냄
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
# 학습률 변화 기록, 출력
lr_logger = LearningRateMonitor()  # log the learning rate


# pytorch_lightning 이용 모델 학습
trainer = pl.Trainer(
    max_epochs=30, # 학습 횟수
    accelerator="cpu", # CPU 사용하여 학습
    enable_model_summary=True, # 모델의 요약 정보 출력
    gradient_clip_val=opt_gcv, # 그레디언트 클리핑 값 설정
    limit_train_batches=50,  # 학습 중 실제로 사용할 학습 batch의 비율, 전체 학습 데이터셋의 50%만 사용
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback], # 학습 중 호출 할 함수 전달 (학습률 모니터링, 조기종료)
    logger=None, # 학습 로그 기록할 로거 설정
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=opt_lr,
    hidden_size=opt_hs,
    attention_head_size=opt_ahs,
    dropout=opt_dpt,
    hidden_continuous_size=opt_hcs,
    loss=QuantileLoss(), # 주어진 분위수에 대해 예측된 값과 실제 값 사이의 오차를 측정
    log_interval=10,  # 로깅 주기 설정
    optimizer="Ranger",
    reduce_on_plateau_patience=4, #  학습 손실이 개선되지 않을 때 학습률을 조정하는 패션스(patience)를 설정, 4번의 에포크동안 학습 손실 개선 안될 시 학습률 줄임
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# trainer = 모델 학습의 주체(tft 모델 학습 시 필요한 기능 제공, 제어 담당), tft = 실제 학습할 모델 객체
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# 검증 손실에 의거한 최적 모델 로드
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


# 검증세트의 MAE 계산
predictions = best_tft.predict(val_dataloader, return_y=True, return_index= True, trainer_kwargs=dict(accelerator="cpu"))
pred_MAE = MAE()(predictions.output, predictions.y)
pred_MAPE = MAPE()(predictions.output, predictions.y)
# MAE = Mean Absolute Error(평균 절대 오차) : 값이 작을 수록 좋은 모델
# MAPE = Mean Absolute Percentage Error(평균 절대 비율 오차) : 낮은 퍼센트일수록 좋은 모델
print('MAE: {0}'.format(pred_MAE))
print('MAPE: {0}%'.format(pred_MAPE*100))

# 시각화를 위한 시간 지표 할당
index_value = predictions.index['time_idx']

# 전체 데이터와 연도별(월별) 예측 데이터를 같이 출력 (다른 데이터를 사용할 경우 슬라이싱 범위 조정 필요)
plt.title('actual vs predict')
plt.plot(data['Close'], label = 'actual', color = 'red', lw = 0.5, linestyle = '--')
plt.plot([i for i in range(index_value[0], index_value[0]+max_prediction_length)], predictions.output[0], color = 'blue')
plt.plot([i for i in range(index_value[1], index_value[1]+max_prediction_length)], predictions.output[1], color = 'blue')
plt.plot([i for i in range(index_value[2], index_value[2]+max_prediction_length)], predictions.output[2], color = 'blue')
plt.plot([i for i in range(index_value[3], index_value[3]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[4], index_value[4]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[5], index_value[5]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[6], index_value[6]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[7], index_value[7]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[8], index_value[8]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[9], index_value[9]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[10], index_value[10]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[11], index_value[11]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[12], index_value[12]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[13], index_value[13]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[14], index_value[14]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.plot([i for i in range(index_value[15], index_value[15]+max_prediction_length)], predictions.output[3], color = 'blue')
plt.legend()
plt.show()


# 예측 데이터 확대 시각화
for i in range(len(index_value)):
    plt.title('zoomed in last {0} days of {1}'.format(max_prediction_length, predictions.index['year'][i]))
    plt.plot([i for i in range(index_value[i], index_value[i]+max_prediction_length)], predictions.y[0][i], label = 'actual', color = 'red', lw = 0.5, linestyle = '--')
    plt.plot([i for i in range(index_value[i], index_value[i]+max_prediction_length)], predictions.output[i], label = 'predict1', color = 'blue')
    plt.show()


# x축 = 예측값, y축 = 실제값
# 대각선에 값이 많이 모여있을 수록 좋은 모델
plt.figure(figsize=(6, 6))
plt.scatter(predictions.output, predictions.y[0])
# plt.plot([i for i in range(0, 381)], [i for i in range(0, 381)], color = 'red', lw = 0.5)
plt.show()
