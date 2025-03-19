import pywt
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import PowerTransformer



# TFT 모델을 이용한 종가 예측
# 코드 실행 프로그램을 관리자 권한으로 실행할 것
# python 3.9.19 환경에서 실행

import os
import warnings
import matplotlib.pyplot as plt
import torch.backends
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
from pytorch_forecasting.metrics import MAE, MAPE, QuantileLoss, RMSE, MASE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import SMAPE
from lightning.pytorch.tuner import Tuner
from sklearn.decomposition import PCA



# 최대한의 시작 날짜 설정
start_date = '1970-01-01'
end_date = '2024-07-26'

# S&P 500 ETF 데이터 불러오기
snp500 = yf.download('SPY', start=start_date, end=end_date)['Adj Close']

# 미국 실업률 데이터 불러오기
unemployment_rate = web.DataReader('UNRATE', 'fred', start_date, end_date)

# 기준금리 데이터 불러오기
interest_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)

# 장단기 금리차 데이터 불러오기
term_spread = web.DataReader('T10Y2Y', 'fred', start_date, end_date)

# 하이일드 스프레드 데이터 불러오기
high_yield_spread = web.DataReader('BAMLH0A0HYM2', 'fred', start_date, end_date)

# WTI 원유 데이터 불러오기
wti = web.DataReader('DCOILWTICO', 'fred', start_date, end_date)

# 기대인플레이션
ten_year_yield = web.DataReader('T10YIE', 'fred', start_date, end_date)

# VIX 데이터 불러오기
vix = yf.download('^VIX', start=start_date, end=end_date)['Adj Close']

# 10년물 국채금리 데이터 불러오기
ten_year_treasury = web.DataReader('DGS10', 'fred', start_date, end_date)


# 데이터를 거래일 기준으로 리샘플링 및 NA 값 처리
snp500 = snp500.asfreq('B').fillna(method='ffill')
unemployment_rate = unemployment_rate.asfreq('B').fillna(method='ffill')
interest_rate = interest_rate.asfreq('B').fillna(method='ffill')
term_spread = term_spread.asfreq('B').fillna(method='ffill')
high_yield_spread = high_yield_spread.asfreq('B').fillna(method='ffill')
wti = wti.asfreq('B').fillna(method='ffill')
ten_year_yield = ten_year_yield.asfreq('B').fillna(method='ffill')
vix = vix.asfreq('B').fillna(method='ffill')
ten_year_treasury = ten_year_treasury.asfreq('B').fillna(method='ffill')

# 원본 데이터프레임 결합 (data1)
data1 = pd.concat([snp500, unemployment_rate, interest_rate, term_spread, 
                   high_yield_spread, wti, ten_year_yield, vix, ten_year_treasury], axis=1)

data1.columns = ['SNP500', 'Unemployment_Rate', 'Interest_Rate', 'Term_Spread', 
                 'High_Yield_Spread', 'WTI', 'Ten_Year_Yield', 'VIX', 'Ten_Year_Treasury']

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
data2['Unemployment_Rate_diff'] = calculate_difference(data1['Unemployment_Rate'])  # 실업률 변화량
data2['Interest_Rate_diff'] = calculate_difference(data1['Interest_Rate'])  # 금리 변화량
data2['Term_Spread_diff'] = calculate_difference(data1['Term_Spread'])  # 장단기 금리차 변화량
data2['High_Yield_Spread_diff'] = calculate_difference(data1['High_Yield_Spread'])  # 하이일드 스프레드 변화량
data2['WTI_diff'] = calculate_percentage_change(data1['WTI'])  # WTI 단순 변화율
data2['Ten_Year_Yield_diff'] = calculate_difference(data1['Ten_Year_Yield'])  # 기대인플레이션 변화량
data2['VIX_log_PC'] = calculate_log_return(data1['VIX'])  # VIX 로그 변화율
data2['Ten_Year_Treasury_diff'] = calculate_difference(data1['Ten_Year_Treasury'])  # 10년물 국채금리 변화량

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
data_weekly['Unemployment_Rate_diff'] = calculate_difference(data_weekly['Unemployment_Rate'])  # 실업률 변화량
data_weekly['Interest_Rate_diff'] = calculate_difference(data_weekly['Interest_Rate'])  # 금리 변화량
data_weekly['Term_Spread_diff'] = calculate_difference(data_weekly['Term_Spread'])  # 장단기 금리차 변화량
data_weekly['High_Yield_Spread_diff'] = calculate_difference(data_weekly['High_Yield_Spread'])  # 하이일드 스프레드 변화량
data_weekly['WTI_diff'] = calculate_percentage_change(data_weekly['WTI'])  # WTI 단순 변화율
data_weekly['Ten_Year_Yield_diff'] = calculate_difference(data_weekly['Ten_Year_Yield'])  # 기대인플레이션 변화량
data_weekly['VIX_log_PC'] = calculate_log_return(data_weekly['VIX'])  # VIX 로그 변화율
data_weekly['Ten_Year_Treasury_diff'] = calculate_difference(data_weekly['Ten_Year_Treasury'])  # 10년물 국채금리 변화량

# NaN 값 제거
data_weekly = data_weekly.dropna()

# 월별 리샘플링
data_monthly = data_combined.resample('M').last()

# 월별 로그 변화율 및 단순 변화율, 변화량 다시 계산
data_monthly['SNP500_log_return'] = calculate_log_return(data_monthly['SNP500'])  # S&P 500 로그 변화율
data_monthly['Unemployment_Rate_diff'] = calculate_difference(data_monthly['Unemployment_Rate'])  # 실업률 변화량
data_monthly['Interest_Rate_diff'] = calculate_difference(data_monthly['Interest_Rate'])  # 금리 변화량
data_monthly['Term_Spread_diff'] = calculate_difference(data_monthly['Term_Spread'])  # 장단기 금리차 변화량
data_monthly['High_Yield_Spread_diff'] = calculate_difference(data_monthly['High_Yield_Spread'])  # 하이일드 스프레드 변화량
data_monthly['WTI_diff'] = calculate_percentage_change(data_monthly['WTI'])  # WTI 단순 변화율
data_monthly['Ten_Year_Yield_diff'] = calculate_difference(data_monthly['Ten_Year_Yield'])  # 기대인플레이션 변화량
data_monthly['VIX_log_PC'] = calculate_log_return(data_monthly['VIX'])  # VIX 로그 변화율
data_monthly['Ten_Year_Treasury_diff'] = calculate_difference(data_monthly['Ten_Year_Treasury'])  # 10년물 국채금리 변화량

# NaN 값 제거
data_monthly = data_monthly.dropna()

# 주간 데이터를 data2로 설정 (분석에 사용할 수 있도록)
data2 = data_weekly

data2['SNP500'] = np.log(data2['SNP500'])



wv = 'coif5'

for column in data2.columns:
    # 노이즈 제거 과정
    signal = data2[column].values
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

    # 노이즈 제거된 신호를 다시 data2에 저장
    data2[column] = denoised_signal


plt.plot(data2.index, data2['SNP500'], color='blue')
plt.show()


# 노이즈 제거된 신호를 이전 시점과 차분
data2['SNP500_diff'] = data2['SNP500'].diff().dropna()


# 차분된 데이터를 사용하여 그래프 그리기
plt.plot(data2.index, data2['SNP500_diff'], color='blue')
plt.title("Differenced and Denoised SNP500")
plt.xlabel("Date")
plt.ylabel("Differenced Value")
plt.show()

data2= data2.dropna()

# 인덱스를 열로 복사하여 Date 열 생성
data2['Date'] = data2.index

# Date 열의 데이터 타입을 datetime으로 변환
data2['Date'] = pd.to_datetime(data2['Date'])



# 데이터 생성
data = data2
data['Date'] = pd.to_datetime(data['Date'])
data["time_idx"] = [i for i in range(1, len(data['Date'])+1)]

# year로 범주형 변수 생성
#data['year'] = data.Date.dt.year.astype(str).astype('int')
# 월 별로 분석 시 
data['month'] = data.Date.dt.month.astype(str).astype('str')
data['week'] = data['Date'].dt.strftime('%W').astype(str)
data['group_id'] = 'group'


# 인코더 : 시계열 데이터의 초기 부분을 입력 받아 특징 추출
# 디코더 : 인코더가 생성한 벡터 입력 받아 원하는 출력 시퀀스 생성 (다음 단계 예측)

# 데이터 분할
train_size = int(len(data) * 0.5)
val_size = int(len(data) * 0.25)
test_size = len(data) - train_size - val_size

train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


# 표준화를 적용할 열 선택 (제외할 열들을 제외한 나머지 열들)
cols_to_scale = [col for col in train_data.columns if col not in ['Date', 'time_idx', 'year', 'month','week', 'group_id',
                                                                  "SNP500_log_return",
                                                                  "SNP500",
                                                                  'Interest_Rate',
                                                                  'Unemployment_Rate','SNP500_diff'
                                                                  #'WTI'
                                                                  ]]

# 스케일러 생성
scaler = StandardScaler()

# train_data 표준화
train_data[cols_to_scale] = scaler.fit_transform(train_data[cols_to_scale])

# val_data 표준화
val_data[cols_to_scale] = scaler.transform(val_data[cols_to_scale])

# test_data 표준화
test_data[cols_to_scale] = scaler.transform(test_data[cols_to_scale])

train_data['SNP500_log_return']

val_data['SNP500_log_return']
test_data['SNP500_log_return']

'''

# Yeo-Johnson 변환기 생성
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson')

# train_data에 대해 Yeo-Johnson 변환 학습 및 적용
train_data['SNP500_log_return'] = yeo_johnson_transformer.fit_transform(train_data[['SNP500_log_return']])

# val_data와 test_data에 동일한 Yeo-Johnson 변환 적용
val_data['SNP500_log_return'] = yeo_johnson_transformer.transform(val_data[['SNP500_log_return']])
test_data['SNP500_log_return'] = yeo_johnson_transformer.transform(test_data[['SNP500_log_return']])
'''



# 유지할 변수들의 리스트
variables_to_keep = ['train_data', 'val_data', 'test_data']

# 현재 워크스페이스에 있는 모든 변수들의 리스트를 가져오기
all_variables = list(globals().keys())

# 유지할 변수들을 제외한 나머지 변수들을 삭제
for var in all_variables:
    if var not in variables_to_keep:
        del globals()[var]

import pywt
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import PowerTransformer



# TFT 모델을 이용한 종가 예측
# 코드 실행 프로그램을 관리자 권한으로 실행할 것
# python 3.9.19 환경에서 실행

import os
import warnings
import matplotlib.pyplot as plt
import torch.backends
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
from pytorch_forecasting.metrics import MAE, MAPE, QuantileLoss, RMSE, MASE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import SMAPE
from lightning.pytorch.tuner import Tuner
from sklearn.decomposition import PCA


acc = 'gpu'

# 월 별로 분석 시 max_prediction_length와 max_encoder_length의 길이 조정
max_prediction_length = 4# 예측 데이터수
max_encoder_length = 52# 학습하는 과거 데이터 수
training_cutoff = 800
#training_cutoff = data["time_idx"].max() - max_prediction_length # time_idx의 최댓값 - 예측 일수 -> 




# training 데이터셋
training = TimeSeriesDataSet(
    train_data,  
    time_idx="time_idx",
    target="SNP500_diff",
    group_ids=['group_id'],
    max_encoder_length=max_encoder_length,
    min_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_categoricals=['month','week'
                                     ],
    time_varying_known_reals=[],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        #'Interest_Rate',
        #'Unemployment_Rate',
        'Term_Spread',
        'High_Yield_Spread',
        #'Term_Spread_diff',
        #'High_Yield_Spread_diff',
        #"SNP500_log_return",
        #'WTI_diff',
        #'WTI',
        'SNP500',
        #'Ten_Year_Yield_diff',
        'Ten_Year_Yield',
        'VIX', 
        'Ten_Year_Treasury',
        'SNP500_diff'
    ],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    predict_mode=False,)






# validation 데이터셋
validation = TimeSeriesDataSet.from_dataset(
    training, 
    val_data,
    stop_randomization=True,predict= False)


# test 데이터셋
test = TimeSeriesDataSet.from_dataset(
    training, 
    test_data,
    stop_randomization=True,predict= False)





batch_size = 64
lstm_layers = 1
loss= SMAPE()

# DataLoader 생성
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0,shuffle=False)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0,shuffle=False)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0,shuffle=False)

'''

# 최적의 하이퍼 파라미터 탐색
opt_hp = optimize_hyperparameters(
    train_dataloader, # 학습 데이터
    val_dataloader, # 검증 데이터
    model_path= 'optuna_test', # 최적화 결과 저장 장소 지정
   
    n_trials= 30, # trial 횟수
    max_epochs= 30, # 한 trial에서 시행할 에포크 수
    
    hidden_size_range=(64, 500),
    hidden_continuous_size_range=(1, 265),
    attention_head_size_range=(1,36),
    gradient_clip_val_range=(0.01, 10),
    dropout_range= (0.1, 0.5),
    learning_rate_range= (0.00001, 1),
    use_learning_rate_finder = False, # 학습 시 learning rate 탐색 여부 결정, learning_rate도 같이 탐색하기 때문에 실행 시간 단축을 위해 False 지정

    # limit_train_batches = 학습 시 사용할 batch의 수 제한
    # log_every_n_steps = 로그를 기록할 간격 지정
    # accelerator = 학습장치
    trainer_kwargs=dict(limit_train_batches = batch_size
                        ,log_every_n_steps = 15, accelerator = acc),
    
    reduce_on_plateau_patience = 20, # trial을 n번 시행했음에도 손실값이 개선되지 않으면 탐색 중지
    timeout = 5400*9, # 최대 실행 시간
    optimizer = "Ranger",
    lstm_layers = lstm_layers,
    
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

'''


# pl.seed_everything() = 실험의 재현성을 위해 사용되는 함수, 매번 동일한 시드값을 사용하게 함(랜덤성 제어)
pl.seed_everything(65)

import tensorflow as tf

# 학습 과정 중 검증 손실(val_loss)를 모니터링, 성능 개선 안 되었을 시 학습 중지
# min_delta = 개선된 것으로 간주할 최소 손실 변화
# patience = 성능이 개선되지 않은 상태를 얼마나 참을 수 있는지를 나타내는 숫자
# verbose = True로 설정 시 EarlyStopping이 각 조건 충족 시 메시지를 출력
# mode = 모니터링 지표의 최소화(min) 또는 최대화(max)를 나타냄
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0., patience=100,
                                    verbose=False, mode="min")
# 학습률 변화 기록, 출력
lr_logger = LearningRateMonitor()  # log the learning rate



# pytorch_lightning 이용 모델 학습
trainer = pl.Trainer(
    max_epochs=100, # 학습 횟수
    accelerator=acc, # CPU 사용하여 학습
    enable_model_summary=True, # 모델의 요약 정보 출력
    gradient_clip_val=0.1, # 그레디언트 클리핑 값 설정
    # limit_train_batches=64,
    # 학습 중 실제로 사용할 학습 batch의 비율, 전체 학습 데이터셋의 50%만 사용
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback], # 학습 중 호출 할 함수 전달 (학습률 모니터링, 조기종료)
    logger=None, # 학습 로그 기록할 로거 설정
    val_check_interval= 0.1
        )




tft = TemporalFusionTransformer.from_dataset(
    training,
   
    learning_rate=0.01,
    hidden_size=512,
    #attention_head_size=516,
    dropout=0.2,
    #hidden_continuous_size=8,
    loss=loss, 
    lstm_layers = lstm_layers,

    log_interval=10,  # 로깅 주기 설정
    optimizer="Ranger",
    reduce_on_plateau_patience=10#  학습 손실이 개선되지 않을 때 학습률을 조정하는 패션스(patience)를 설정, 4번의 에포크동안 학습 손실 개선 안될 시 학습률 줄임
    )



print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")



# trainer = 모델 학습의 주체(tft 모델 학습 시 필요한 기능 제공, 제어 담당), tft = 실제 학습할 모델 객체
trainer.fit( 
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)
# 검증 손실에 의거한 최적 모델 로드
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)


# 검증세트의 MAE 계산
predictions = best_tft.predict(val_dataloader,trainer_kwargs=dict(accelerator=acc), return_y= True, batch_size= 64)

a = predictions.output
aa = predictions.y[0].reshape(-1, max_prediction_length)

pred_MAE = MAE()(a, aa)
pred_MAPE = MAPE()(a, aa)
pred_SMAPE = SMAPE()(a, aa)

# MAE = Mean Absolute Error(평균 절대 오차) : 값이 작을 수록 좋은 모델
# MAPE = Mean Absolute Percentage Error(평균 절대 비율 오차) : 낮은 퍼센트일수록 좋은 모델
print('MAE: {0}'.format(pred_MAE))
print('MAPE: {0}%'.format(pred_MAPE*100))
print('SMAPE: {0}%'.format(pred_SMAPE*100))


# 테스트세트의 MAE 계산
test_predictions = best_tft.predict(test_dataloader, trainer_kwargs=dict(accelerator=acc), return_y= True,batch_size= 128)


b = test_predictions.output
bb = test_predictions.y[0].reshape(-1, max_prediction_length)

test_MAE = MAE()(b, bb)
test_MAPE = MAPE()(b, bb)
test_SMAPE = SMAPE()(b, bb)

# 결과 출력
print('Test MAE: {0}'.format(test_MAE))
print('Test MAPE: {0}%'.format(test_MAPE * 100))
print('Test SMAPE: {0}%'.format(test_SMAPE * 100))



z = a.cpu().numpy()
zz = aa.cpu().numpy()


y = b.cpu().numpy()
yy = bb.cpu().numpy()

#plt.plot(z.T[0],zz.T[0])




# 각 열을 비교하는 플롯 생성
plt.figure(figsize=(12, 8))

for i in range(max_prediction_length):
    plt.subplot(2, 2, i + 1)  # 2x2 서브플롯 설정
    plt.plot(z[:, i], label=f'z Column {i+1}', color='blue')
    plt.plot(zz[:, i], label=f'zz Column {i+1}', color='red')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Comparison of z and zz - Column {i+1}')
    plt.legend()

plt.tight_layout()
plt.show()



# 각 열을 비교하는 플롯 생성
plt.figure(figsize=(12, 8))

for i in range(max_prediction_length):
    plt.subplot(2, 2, i + 1)  # 2x2 서브플롯 설정
    plt.plot(y[:, i], label=f'z Column {i+1}', color='blue')
    plt.plot(yy[:, i], label=f'zz Column {i+1}', color='red')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Comparison of z and zz - Column {i+1}')
    plt.legend()

plt.tight_layout()
plt.show()














import matplotlib.pyplot as plt

# 가로세로 범위를 동일하게 설정
xy_min = min(z.min(), zz.min(), y.min(), yy.min())
xy_max = max(z.max(), zz.max(), y.max(), yy.max())

# 각 열을 비교하는 플롯 생성
plt.figure(figsize=(12, 8))

for i in range(max_prediction_length):
    plt.subplot(2, 2, i + 1)  # 2x2 서브플롯 설정
    plt.scatter(zz[:, i], z[:, i], label=f'Column {i+1}', color='blue')
    plt.xlabel('zz (X-axis)')
    plt.ylabel('z (Y-axis)')
    plt.title(f'Comparison of z vs zz - Column {i+1}')
    plt.xlim([xy_min, xy_max])
    plt.ylim([xy_min, xy_max])
    plt.legend()

plt.tight_layout()
plt.show()

# 각 열을 비교하는 두 번째 플롯 생성
plt.figure(figsize=(12, 8))

for i in range(max_prediction_length):
    plt.subplot(2, 2, i + 1)  # 2x2 서브플롯 설정
    plt.scatter(yy[:, i], y[:, i], label=f'Column {i+1}', color='blue')
    plt.xlabel('yy (X-axis)')
    plt.ylabel('y (Y-axis)')
    plt.title(f'Comparison of y vs yy - Column {i+1}')
    plt.xlim([xy_min, xy_max])
    plt.ylim([xy_min, xy_max])
    plt.legend()

plt.tight_layout()
plt.show()