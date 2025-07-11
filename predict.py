# 1. 필요한 모듈
import torch
import pandas as pd
import numpy as np
from collections import Counter

# 2. LSTMModel 클래스 복사
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # ✅ 이게 핵심
        )
        self.fc = torch.nn.Linear(hidden_size * 2, 5)  # ✅ 2배로 곱해줌

    def forward(self, x):  # x shape: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 타임스텝
        return self.fc(out)


# 3. 경로 설정
csv_path = r"C:\Users\이찬\dreamt_pretrain\S002_whole_df.csv"
model_path = r"C:\Users\이찬\dreamt_pretrained_4ch.pth"

# 4. 라벨 매핑 정의
LABEL_MAP = {'W':0, 'P':0, 'N1':1, 'N2':2, 'N3':3, 'R':4, 'REM':4}

label_map = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

# 역방향 매핑 (라벨명 -> 숫자)
reverse_label_map = {v: k for k, v in label_map.items()}

# 5. 전처리 및 추론 함수
def preprocess(df, start=0, window=1920):
    x = df.iloc[start:start+window][['BVP', 'ACC_X', 'ACC_Y', 'ACC_Z']].values.T
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape (1, 4, 1920)

def get_majority_label(df, start=0, window=1920):
    """
    30초(epoch) 구간의 다수결 라벨을 구합니다.
    가장 많은 라벨의 비율이 60% 이상일 경우에만 대표 라벨로 간주합니다.
    """
    # 해당 구간의 Sleep_Stage 라벨들 추출
    labels = df.iloc[start:start+window]['Sleep_Stage'].values
    
    # 라벨 개수 계산
    label_counts = Counter(labels)
    total_count = len(labels)
    
    if total_count == 0:
        return None
    
    # 가장 많은 라벨과 그 비율 계산
    most_common_label, most_common_count = label_counts.most_common(1)[0]
    ratio = most_common_count / total_count
    
    # 60% 이상일 경우에만 대표 라벨로 반환
    if ratio >= 0.6:
        return most_common_label
    else:
        return None

def predict_single_epoch(df, model, start_idx):
    """
    단일 epoch에 대한 예측을 수행합니다.
    """
    x = preprocess(df, start=start_idx)
    
    with torch.no_grad():
        y = model(x)
        pred = torch.argmax(y, dim=1).item()
        predicted_label = label_map[pred]
    
    # 실제 라벨 구하기
    raw_label = get_majority_label(df, start=start_idx)
    actual_label_num = LABEL_MAP.get(raw_label, None)
    
    if actual_label_num is not None:
        actual_label = label_map[actual_label_num]
        is_correct = predicted_label == actual_label
        return predicted_label, actual_label, is_correct, True
    else:
        return predicted_label, '(No majority)', False, False

def predict():
    df = pd.read_csv(csv_path)
    x = preprocess(df)

    model = LSTMModel(input_size=4, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y = model(x)
        pred = torch.argmax(y, dim=1).item()
        predicted_label = label_map[pred]
        print(f"Predicted sleep stage: {predicted_label}")
        
        # 실제 라벨 구하기
        raw_label = get_majority_label(df)
        actual_label_num = LABEL_MAP.get(raw_label, None)
        
        if actual_label_num is not None:
            actual_label = label_map[actual_label_num]
            print(f"Actual sleep stage: {actual_label}")
            
            # 예측 결과 비교
            if predicted_label == actual_label:
                print(f"→ OK Prediction is correct!")
            else:
                print(f"→ X Prediction is wrong.")
        else:
            print(f"Actual sleep stage: (No majority)")
            print(f"→ (No majority)")

def auto_evaluate():
    """
    전체 데이터를 30초 단위로 슬라이딩하며 자동 평가를 수행합니다.
    """
    print("=== Auto Evaluation Start ===")
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    window_size = 1920  # 30초 = 1920 샘플
    
    # 모델 로드
    model = LSTMModel(input_size=4, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 평가 변수 초기화
    total_epochs = 0
    valid_epochs = 0
    correct_predictions = 0
    
    # 슬라이딩 윈도우로 반복 추론
    for start_idx in range(0, total_samples - window_size + 1, window_size):
        total_epochs += 1
        epoch_num = total_epochs
        
        # 단일 epoch 예측
        predicted_label, actual_label, is_correct, is_valid = predict_single_epoch(df, model, start_idx)
        
        # 결과 출력
        if is_valid:
            valid_epochs += 1
            if is_correct:
                correct_predictions += 1
                result_symbol = "OK"
            else:
                result_symbol = "X"
            
            print(f"Epoch {epoch_num:3d}: Pred={predicted_label:4s}, Actual={actual_label:4s} → {result_symbol}")
        else:
            print(f"Epoch {epoch_num:3d}: Pred={predicted_label:4s}, Actual={actual_label} → (No majority)")
    
    # 전체 결과 출력
    print("\n=== Evaluation Summary ===")
    print(f"Total epochs evaluated: {total_epochs}")
    print(f"Valid epochs: {valid_epochs}")
    print(f"Correct predictions: {correct_predictions}")
    
    if valid_epochs > 0:
        accuracy = (correct_predictions / valid_epochs) * 100
        print(f"Overall accuracy: {accuracy:.2f}%")
    else:
        print("Overall accuracy: Cannot calculate (no valid epochs)")

# 6. 실행
if __name__ == "__main__":
    # 기존 단일 예측
    # predict()
    
    # 자동 평가 실행
    auto_evaluate()
