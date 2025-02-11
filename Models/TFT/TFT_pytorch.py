# TFT 모델을 이용한 SPY 로그 수익률 예측
# 코드 실행 프로그램을 관리자 권한으로 실행할 것
# 새로운 데이터가 등장했을 떄 -> 행렬 분해 svd -> 추천 시스템

import os
import warnings
import matplotlib.pyplot as plt
import torch.backends

# 경고 메세지가 출력되지 않도록 만듦.
warnings.filterwarnings("ignore") 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir("../../..")

import copy
from pathlib import Path
import warnings
import random
import shutil
import argparse
from tqdm import tqdm

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


# 데이터 생성
data = pd.read_excel("C:/Users/daily/Desktop/combined_weekly.xlsx")
data['Date'] = pd.to_datetime(data['Date'])

data["time_idx"] = data["Date"].dt.year * 12 + data["Date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# year로 범주형 변수 생성
data['month'] = data.Date.dt.month.astype(str).astype('category')

max_prediction_length = 25 # 예측 일수
max_encoder_length = 100 # 학습하는 과거 데이터 일수
training_cutoff = data["time_idx"].max() - max_prediction_length # time_idx의 최댓값 - 예측 일수 -> 

# 인코더 : 시계열 데이터의 초기 부분을 입력 받아 특징 추출
# 디코더 : 인코더가 생성한 벡터 입력 받아 원하는 출력 시퀀스 생성 (다음 단계 예측)

# training 시계열 데이터 생성
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff], # data 중 time_idx가 training_cutoff 이하인 데이터만 추출
    allow_missing_timesteps=True, # 누락된 시계열 구간 허용
    time_idx="time_idx", # 시간 인덱스 열 설정
    target="SPY_Log_Returns_Weekly", # 타겟 열 설정
    group_ids=['month'], # 그룹화할 데이터 설정(범주형 변수일 것)
    min_encoder_length=max_encoder_length // 2,  # 인코더의 최소 길이 설정
    max_encoder_length=max_encoder_length, # 인코더 최대 길이 설정
    min_prediction_length=1, # 모델이 예측할 기간 -> 모델이 한 번의 예측에서 최소한으로 예측할 시간 간격
    max_prediction_length=max_prediction_length,  # 모델이 한 번의 예측에서 최대로 예측할 수 있는 시간 간격
    time_varying_known_categoricals=[], # 값을 알고 있는 범주형 동적변수
    time_varying_known_reals=['SPY', 'VIX', 'TIPS', 'HighYieldSpread', 'TermSpread'], # 값을 알고 있는 연속형 동적 변수 지정
    time_varying_unknown_categoricals=[], # 미지 범주형 동적변수 지정
    time_varying_unknown_reals=['SPY_Log_Returns_Weekly'], # 미지 연속형 동적변수 지정
    target_normalizer=GroupNormalizer( # groups = []로 지정한 범주형 변수에 따라 그룹을 나눠 정규화 수행
        groups=['month'], transformation=None   # 정규화 변환 함수 = softplus(입력이 음수일 경우 0으로 변환되며, 양수일 경우 로그의 역함수로 변환)
    ),
    add_relative_time_idx=True, # 대적인 시간 인덱스(relative time index)를 데이터셋에 추가
    add_target_scales=True, # 타겟 변수의 스케일(범위 , 크기)을 추가
    add_encoder_length=True, # 인코더의 길이(encoder length)를 데이터셋에 추가
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
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
# 검증 데이터셋을 dataloader 형태로 변환
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# 모델이 예측을 수행할 때, 이전 시간 단계의 마지막 값으로 다음 값을 예측하고, 이를 통해 계산된 MAE를 평가
# Baseline() = 시계열 데이터의 이동 평균, 평균값 등의 간단한 통계적 기법을 사용하여 예측을 수행
# val_dataloader(검증 데이터셋) 데이터에 대한 예측 수행
# return_y = True : 실제 타겟 값도 반환
baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
# output(예측값)과 y(실제 타겟 값) 간의 평균 절대 오차를 계산
base_MAE = MAE()(baseline_predictions.output, baseline_predictions.y)
print(base_MAE)

# configure network and trainer

# pl.seed_everything() = 실험의 재현성을 위해 사용되는 함수, 매번 동일한 시드값을 사용하게 함(랜덤성 제어)
pl.seed_everything(42)

# pl.Trainer() = 모델 학습, 검증, 테스트 등의 루프를 관리
# accelerator = 학습을 수행할 디바이스 지정
# gradient_clip_val = 그래디언트 클리핑(gradient clipping) 값 설정, 그레디언트 폭주(발산) 방지
trainer = pl.Trainer(
    accelerator="cpu",
    # clipping gradients is a hyperparameter and important to prevent divergance of the gradient for recurrent neural networks
    gradient_clip_val=0.1
,
)

# TFT 모델 생성

# from_dataset() = 주어진 설정 값으로 TFT 모델 생성
tft = TemporalFusionTransformer.from_dataset(
    training, # training = TFT 모델을 학습할 시계열 데이터셋, 모델 초기화와 학습에 사용
    learning_rate=0.03, # 학습률 설정, 모델의 매개변수 업데이트 속도 조절
    hidden_size=8,  # TFT 모델의 은닉층 크기, 모델의 복잡성 & 표현력 결정 (학습률을 위한 중요한 변수)
    attention_head_size=1, # 어텐션 헤드의 크기 설정 (큰 데이터셋에는 4로 설정)
    # 어텐션 메커니즘 수행에 사용되는 매개변수
    # 어텐션 메커니즘 : 주어진 입력에 대해 각 입력 위치에 중요도를 부여하여 새로운 표현을 생성하는 방법, 쿼리(query), 키(key), 값(value)의 쌍이 사용
    # 어텐션 헤드는 이러한 쿼리와 키에 대한 연산을 병렬로 수행하는 부분
    # attention_head_size = 1 : 각 쿼리(query)에 대해 하나의 키(key)와 연결하여 어텐션 가중치(attention weight)를 계산하는 방식을 사용
    dropout=0.1,  # 드롭아웃 확률 설정 (0.1 ~ 0.3으로 설정)
    # 드롭아웃 : 학습 중 신경망의 일부 뉴런을 무작위 선택, 제외
    hidden_continuous_size=8,  # 연속 변수의 은닉층 크기 설정, 시계열 데이터의 연속적 부분 처리 (hidden_size와 같은 값으로 설정)
    loss=QuantileLoss(), # 손실 함수 설정, QuantileLoss = 특정 분위수(quantile)에 대한 예측 정확도를 향상
    optimizer="Ranger", # 최적화 알고리즘 설정, Ranger = 최적화의 속도와 안정을 개선 위한 발전된 기법 사용한 최적화 알고리즘
    # reduce learning rate if no improvement in validation loss after x epochs
    # reduce_on_plateau_patience=1000,
)
# tft 신경망 모델의 파라미터 수를 출력
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


from lightning.pytorch.tuner import Tuner

# Tuner = 다양한 모델 파라미터를 튜닝하고 최적의 학습률을 찾음
# lr_find = 학습률의 범위를 찾음
res = Tuner(trainer).lr_find(
    tft, # 튜닝할 대상 모델
    train_dataloaders=train_dataloader, # 학습 데이터 로드
    val_dataloaders=val_dataloader, # 검증 데이터 로드
    max_lr=10.0, # 탐색할 최대 학습률
    min_lr=1e-6, # 탐색할 최소 학습률
)

# 학습률 출력
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
# fig = plt.figure()
plt.show()
res_lr = res.suggestion()

import tensorflow as tf
import tensorboard as tb

# 학습 과정 중 검증 손실(val_loss)를 모니터링, 성능 개선 안 되었을 시 학습 중지
# min_delta = 개선된 것으로 간주할 최소 손실 변화
# patience = 성능이 개선되지 않은 상태를 얼마나 참을 수 있는지를 나타내는 숫자
# verbose = True로 설정 시 EarlyStopping이 각 조건 충족 시 메시지를 출력
# mode = 모니터링 지표의 최소화(min) 또는 최대화(max)를 나타냄
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
# 학습률 변화 기록, 출력
lr_logger = LearningRateMonitor()  # log the learning rate
# 학습 로그 기록, 시각화
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard


# pytorch_lightning 이용 모델 학습
trainer = pl.Trainer(
    max_epochs=10, # 학습 횟수
    accelerator="cpu", # CPU 사용하여 학습
    enable_model_summary=True, # 모델의 요약 정보 출력
    gradient_clip_val=0.1, # 그레디언트 클리핑 값 설정
    limit_train_batches=50,  # 학습 중 실제로 사용할 학습 batch의 비율, 전체 학습 데이터셋의 50%만 사용
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback], # 학습 중 호출 할 함수 전달 (학습률 모니터링, 조기종료)
    logger=logger, # 학습 로그 기록할 로거 설정
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=res_lr,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
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
predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
pred_MAE = MAE()(predictions.output, predictions.y)
print(pred_MAE)

pred_value = []
act_value = []

raw_predictions = best_tft.predict(training)

plt.title('actual vs predict (SPY Log Returns)')
plt.plot(raw_predictions, label = 'prediction', color = 'blue', lw = 1)
plt.plot(data["SPY_Log_Returns_Weekly"], label = 'actual', color = 'red', lw = 1)
plt.xlabel('time')
plt.show()
