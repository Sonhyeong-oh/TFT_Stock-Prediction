# lightGBM을 이용한 종가 예측

import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


data = pd.read_csv("C:/Users/daily/Desktop/train.csv")
data1 = data.drop('Date', axis = 1)

# 데이터 분할 (훈련데이터 9 : 테스트 데이터 1)
n_train = int(0.9*data1.shape[0])
train = data1.iloc[:n_train]
test = data1.iloc[n_train:]

train_input = train.drop('Close', axis = 1)
train_target = train['Close']
test_input = test.drop('Close', axis = 1)
test_target = test['Close']

# 하이퍼 파라미터 설정
params = {
    'max_depth' : -1, # 각 트리의 최대 깊이 설정
    'boosting_type': 'gbdt',  # 트리 부스팅 타입 설정
    'objective': 'regression',    # 목적 함수 설정
    'metric': 'mse',  # 평가 지표 설정
    'num_leaves': 24,  # 각 트리의 최대 잎 노드 수
    'learning_rate': 0.1,     # 학습률(학습 정도 조절)
    'feature_fraction': 0.8,  # 각 트리마다 선택할 특성의 비율
    'bagging_fraction': 0.8,  # 각 트리마다 선택할 데이터의 비율
    'bagging_freq': 5,        # 데이터 샘플링 빈도
    'verbose': 1              # 학습 중 메시지 출력 여부 설정
}

# 데이터를 DataFrame -> Dataset으로 변환
train_data = lgb.Dataset(train_input, label=train_target)
test_data = lgb.Dataset(test_input, label=test_target)

# LightGBM 모델 학습
# train(파라미터 설정, 학습 데이터, 학습 반복 횟수, 검증 데이터셋 설정)
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=test_data)

# [LightGBM] [Warning] No further splits with positive gain, best gain: -inf 오류 해결을 위한 leaf 개수 탐색 함수
def leaf_count(model):
    tree_infos = model.dump_model()["tree_info"]

    # 각 트리의 잎 개수 저장 리스트
    leaves_list = []
    # 각 트리의 잎 개수 추출, 저장
    for tree in tree_infos:
        leaves_list.append(tree["num_leaves"])

    return min(leaves_list)

print('minimum of leaf:',leaf_count(model))

# target_pred = test_input으로 학습하여 예측한 값
# num_iteration = 트리의 반복 횟수 , model.best_iteration = 최적의 반복 횟수로 설정
target_pred = model.predict(test_input, num_iteration=model.best_iteration)

# 모델 성능 평가
# mse = 평균 제곱 오차, 실제 값과 예측 값 사이의 차이를 제곱한 값의 평균
# 값이 작을 수록 오차가 작음
mse = mean_squared_error(test_target, target_pred)
print(f'MSE: {mse}')


# r2점수 = 예측된 값이 실제 값과 얼마나 잘 일치하는지 설명하는 지표, 1에 가까울 수록 좋은 모델
# 1 - (sum(관측값 - 예측값)^2 / sum(관측값 - 타겟 데이터 평균값)^2)
# 1 - (sum(test_target - target_pred)^2 / sum(test_target - np.mean(test_target))^2)
from sklearn.metrics import r2_score
r2 = r2_score(test_target, target_pred)
print('R square score:',r2)

# 데이터 출처 : https://www.kaggle.com/competitions/netflix-stock-prediction/data?select=sample_submission.csv