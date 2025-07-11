import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import logging
from collections import Counter

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 전역 설정 상수
SAMPLING_RATE = 64.0        # Hz
WINDOW_DURATION = 30        # seconds
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)  # 1920 samples
NUM_CLASSES = 5             # 수면 단계 클래스 수 (W, N1, N2, N3, R)
MAJORITY_RATIO = 0.6        # majority voting 비율

# 현재 사용할 신호 채널 설정 (향후 확장 가능)
CURRENT_SIGNAL_CHANNELS = ['bvp', 'acc_x', 'acc_y', 'acc_z']  # 현재 4채널
CURRENT_LABEL_COLUMN = 'sleep_stage'  # 현재 라벨 컬럼명

# 향후 확장용 신호 채널 (주석 처리)
# FUTURE_SIGNAL_CHANNELS = ['ir', 'red', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']  # 8채널
# FUTURE_LABEL_COLUMN = 'label'  # 향후 라벨 컬럼명

# 학습 설정 상수
DREAMT_DATA_DIR = "path/to/DREAMT"
ACTUAL_DATA_DIR = "path/to/actual_data"
PRETRAINED_PATH = "dreamt_pretrained.pth"
FINETUNED_PATH = "finetuned_model.pth"
EPOCHS_PRETRAIN = 30
EPOCHS_FINETUNE = 20
BATCH_SIZE = 32
LR_PRETRAIN = 1e-3
LR_FINETUNE = 1e-5

LABEL_MAP = {
    'W': 0, 'P': 0,  # Wake
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'R': 4, 'REM': 4
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def map_sleep_stage(raw_label):
    return LABEL_MAP.get(str(raw_label).strip().upper(), None)


def extract_epochs_from_df(df, majority_ratio=MAJORITY_RATIO, signal_channels=None, label_column=None):
    """
    DataFrame에서 epoch별로 raw signal과 대표 라벨 추출 (majority voting)
    
    Args:
        df (pd.DataFrame): DataFrame with signal and label columns
        majority_ratio (float): majority voting 비율 (기본값: 0.6)
        signal_channels (list): 사용할 신호 채널 리스트 (기본값: CURRENT_SIGNAL_CHANNELS)
        label_column (str): 라벨 컬럼명 (기본값: CURRENT_LABEL_COLUMN)
        
    Returns:
        tuple: (X, y) where X is np.array of shape (N, num_channels, 1920), y is np.array of shape (N,)
    """
    if signal_channels is None:
        signal_channels = CURRENT_SIGNAL_CHANNELS
    if label_column is None:
        label_column = CURRENT_LABEL_COLUMN
        
    X = []
    y = []
    n_epochs = len(df) // WINDOW_SIZE
    num_channels = len(signal_channels)
    
    logging.info(f"Processing {n_epochs} epochs from DataFrame with {len(df)} samples")
    logging.info(f"Using {num_channels} channels: {signal_channels}")
    logging.info(f"Using label column: {label_column}")
    
    for i in range(n_epochs):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        epoch_df = df.iloc[start:end]
        
        # Epoch 크기 검증
        if len(epoch_df) < WINDOW_SIZE:
            logging.warning(f"Epoch {i}: insufficient samples ({len(epoch_df)} < {WINDOW_SIZE})")
            continue
            
        # 라벨 majority voting
        labels = epoch_df[label_column].map(map_sleep_stage).dropna().astype(int).values
        if len(labels) == 0:
            logging.warning(f"Epoch {i}: no valid labels found")
            continue
            
        label_counts = Counter(labels)
        major_label, count = label_counts.most_common(1)[0]
        ratio = count / len(labels)
        
        if ratio < majority_ratio:
            logging.debug(f"Epoch {i}: majority ratio {ratio:.2f} < {majority_ratio}, skipping")
            continue
            
        # 신호 추출 (동적 채널 수)
        try:
            signals = []
            for channel in signal_channels:
                if channel not in epoch_df.columns:
                    logging.warning(f"Epoch {i}: missing channel {channel}")
                    continue
                signals.append(epoch_df[channel].values)
            
            if len(signals) != num_channels:
                logging.warning(f"Epoch {i}: expected {num_channels} channels, got {len(signals)}")
                continue
            
            # NaN 값 검사 및 처리
            signals = [np.nan_to_num(sig, nan=0.0) for sig in signals]
            signals = np.stack(signals, axis=0)  # shape: (num_channels, 1920)
            
            # 신호 품질 검사 (모든 값이 0이 아닌지 확인)
            if np.all(signals == 0):
                logging.warning(f"Epoch {i}: all signals are zero, skipping")
                continue
                
            X.append(signals)
            y.append(major_label)
            
        except Exception as e:
            logging.warning(f"Error extracting signals for epoch {i}: {e}")
            continue
    
    logging.info(f"Successfully extracted {len(X)} valid epochs")
    return np.array(X), np.array(y)


def load_dreamt_data(data_dir, majority_ratio=MAJORITY_RATIO, signal_channels=None, label_column=None):
    """
    DREAMT 데이터 로드 (raw signal 기반)
    
    Args:
        data_dir (str): DREAMT 데이터 디렉토리 경로
        majority_ratio (float): majority voting 비율 (기본값: 0.6)
        signal_channels (list): 사용할 신호 채널 리스트 (기본값: CURRENT_SIGNAL_CHANNELS)
        label_column (str): 라벨 컬럼명 (기본값: CURRENT_LABEL_COLUMN)
        
    Returns:
        tuple: (X, y) where X is np.array of shape (N, num_channels, 1920), y is np.array of shape (N,)
    """
    if signal_channels is None:
        signal_channels = CURRENT_SIGNAL_CHANNELS
    if label_column is None:
        label_column = CURRENT_LABEL_COLUMN
        
    X_list, y_list = [], []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        logging.warning(f"No CSV files found in {data_dir}")
        return np.array([]), np.array([])
        
    logging.info(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    for csv_file in csv_files:
        logging.info(f"Reading file: {os.path.basename(csv_file)}")
        try:
            df = pd.read_csv(csv_file)
            
            # 컬럼명 소문자 통일 및 공백 제거
            df.columns = [c.lower().strip() for c in df.columns]
            
            # 필수 컬럼 검증 (현재 사용할 채널 + 라벨 컬럼)
            required_cols = signal_channels + [label_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logging.warning(f"Missing columns in {os.path.basename(csv_file)}: {missing_cols}")
                continue
                
            # 데이터 타입 검증 및 변환
            for col in signal_channels:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # NaN 값이 너무 많은 경우 제외
            nan_counts = df[signal_channels].isna().sum()
            if nan_counts.max() > len(df) * 0.1:  # 10% 이상 NaN이면 제외
                logging.warning(f"Too many NaN values in {os.path.basename(csv_file)}, skipping")
                continue
                
            X, y = extract_epochs_from_df(df, majority_ratio=majority_ratio, 
                                        signal_channels=signal_channels, label_column=label_column)
            
            if len(X) > 0:
                X_list.append(X)
                y_list.append(y)
                logging.info(f"Successfully loaded {len(X)} epochs from {os.path.basename(csv_file)}")
            else:
                logging.warning(f"No valid epochs extracted from {os.path.basename(csv_file)}")
                
        except Exception as e:
            logging.warning(f"Error loading {os.path.basename(csv_file)}: {e}")
            continue
    
    if X_list:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        logging.info(f"Total loaded: {len(X)} epochs from {len(X_list)} files")
        return X, y
    else:
        logging.error("No valid data loaded from any files")
        return np.array([]), np.array([])


def load_actual_data(data_dir, majority_ratio=MAJORITY_RATIO, signal_channels=None, label_column=None):
    # 실제 데이터도 DREAMT 포맷과 동일하게 처리
    return load_dreamt_data(data_dir, majority_ratio, signal_channels, label_column)


class SleepRawDataset(Dataset):
    """Raw signal 기반 수면 데이터셋 (X: [num_channels,1920], y: int)"""
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        
        # 데이터 검증
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y lengths don't match: {len(self.X)} vs {len(self.y)}")
        if len(self.X.shape) != 3 or self.X.shape[2] != 1920:
            raise ValueError(f"Expected X shape (N, num_channels, 1920), got {self.X.shape}")
            
        self.num_channels = self.X.shape[1]
        logging.info(f"Dataset initialized with {len(self.X)} samples, shape: {self.X.shape}")
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # (num_channels,1920) -> (1920,num_channels) for LSTM (seq_len, feature_dim)
        x = torch.from_numpy(self.X[idx].T)  # shape: (1920, num_channels)
        y = torch.tensor(self.y[idx])
        return x, y


class RawLSTMClassifier(nn.Module):
    """LSTM 기반 수면 단계 분류기 (raw signal 입력)"""
    def __init__(self, input_size=None, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        
        # input_size가 None이면 CURRENT_SIGNAL_CHANNELS의 길이 사용
        if input_size is None:
            input_size = len(CURRENT_SIGNAL_CHANNELS)
            
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
        # 가중치 초기화
        self._init_weights()
        
        logging.info(f"LSTM model initialized with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
        
    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 마지막 타임스텝
        out = self.dropout(out)
        out = self.fc(out)
        return out


def pretrain_on_dreamt(data_dir, output_path, epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE, lr=LR_PRETRAIN, majority_ratio=MAJORITY_RATIO, signal_channels=None, label_column=None, class_weights=None):
    """
    DREAMT 데이터로 Pre-train
    
    Args:
        data_dir (str): DREAMT 데이터 디렉토리 경로
        output_path (str): 모델 저장 경로
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        lr (float): 학습률
        majority_ratio (float): majority voting 비율
        signal_channels (list): 사용할 신호 채널 리스트 (기본값: CURRENT_SIGNAL_CHANNELS)
        label_column (str): 라벨 컬럼명 (기본값: CURRENT_LABEL_COLUMN)
    """
    if signal_channels is None:
        signal_channels = CURRENT_SIGNAL_CHANNELS
    if label_column is None:
        label_column = CURRENT_LABEL_COLUMN
        
    logging.info("Loading DREAMT data...")
    X, y = load_dreamt_data(data_dir, majority_ratio, signal_channels, label_column)
    
    if len(X) == 0:
        logging.error("No valid data found!")
        return
        
    logging.info(f"Loaded {len(X)} samples")
    
    # 클래스 분포 확인
    unique, counts = np.unique(y, return_counts=True)
    logging.info("Class distribution:")
    for label, count in zip(unique, counts):
        logging.info(f"  {INV_LABEL_MAP[label]}: {count} ({count/len(y)*100:.1f}%)")
    
    # 데이터 분할 (stratify 사용하여 클래스 균형 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SleepRawDataset(X_train, y_train)
    test_dataset = SleepRawDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 초기화 (동적 input_size)
    input_size = len(signal_channels)
    model = RawLSTMClassifier(input_size=input_size, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES)
    
    # GPU 사용 가능시 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"Using device: {device}")
    
    # 클래스별 가중치 설정 (클래스 불균형 해결)
    if class_weights is None:
        # 자동 계산: 적은 샘플에 높은 가중치
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
    else:
        # 사용자 지정 가중치 사용
        class_weights = np.array(class_weights)
    
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logging.info(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping 변수
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logging.info("Starting pre-training...")
    
    # 학습 루프
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # 평균 손실 계산
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= 10:  # Early stopping patience
            logging.info(f'Early stopping at epoch {epoch+1}')
            break
        
        # 진행 상황 출력 (매 epoch마다)
        logging.info(f'Epoch [{epoch+1}/{epochs}]')
        logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logging.info(f'Accuracy: {accuracy:.2f}%')
    
    # 최적의 모델 상태 복원 및 저장
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, output_path)
        logging.info(f"Pre-trained model saved to {output_path}")
    
    # === 평가 메트릭 출력 ===
    logging.info("\nConfusion Matrix (Validation Set):")
    print(confusion_matrix(all_targets, all_preds))
    logging.info("\nClassification Report (Validation Set):")
    print(classification_report(all_targets, all_preds, digits=4))


def finetune_on_actual(actual_data_dir, pretrained_path, output_path, epochs=EPOCHS_FINETUNE, batch_size=BATCH_SIZE, lr=LR_FINETUNE, majority_ratio=MAJORITY_RATIO, signal_channels=None, label_column=None):
    """
    실제 데이터로 Fine-tune
    
    Args:
        actual_data_dir (str): 실제 데이터 디렉토리 경로
        pretrained_path (str): Pre-trained 모델 경로
        output_path (str): Fine-tuned 모델 저장 경로
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        lr (float): 학습률
        majority_ratio (float): majority voting 비율
        signal_channels (list): 사용할 신호 채널 리스트 (기본값: CURRENT_SIGNAL_CHANNELS)
        label_column (str): 라벨 컬럼명 (기본값: CURRENT_LABEL_COLUMN)
    """
    if signal_channels is None:
        signal_channels = CURRENT_SIGNAL_CHANNELS
    if label_column is None:
        label_column = CURRENT_LABEL_COLUMN
        
    logging.info("Loading actual data...")
    X, y = load_actual_data(actual_data_dir, majority_ratio, signal_channels, label_column)
    
    if len(X) == 0:
        logging.error("No valid data found!")
        return
        
    logging.info(f"Loaded {len(X)} samples")
    
    # 클래스 분포 확인
    unique, counts = np.unique(y, return_counts=True)
    logging.info("Class distribution:")
    for label, count in zip(unique, counts):
        logging.info(f"  {INV_LABEL_MAP[label]}: {count} ({count/len(y)*100:.1f}%)")
    
    # 데이터 분할 (stratify 사용하여 클래스 균형 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SleepRawDataset(X_train, y_train)
    test_dataset = SleepRawDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 초기화 및 pre-trained 가중치 로드 (동적 input_size)
    input_size = len(signal_channels)
    model = RawLSTMClassifier(input_size=input_size, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES)
    
    # GPU 사용 가능시 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"Using device: {device}")
    
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        logging.info(f"Loaded pre-trained weights from {pretrained_path}")
    else:
        logging.warning("Pre-trained model not found, starting from scratch")
    
    # 클래스별 가중치 계산 (클래스 불균형 해결)
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping 변수
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logging.info("Starting fine-tuning...")
    
    # 학습 루프
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # 평균 손실 계산
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= 10:  # Early stopping patience
            logging.info(f'Early stopping at epoch {epoch+1}')
            break
        
        # 진행 상황 출력 (매 epoch마다)
        logging.info(f'Epoch [{epoch+1}/{epochs}]')
        logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logging.info(f'Accuracy: {accuracy:.2f}%')
    
    # 최적의 모델 상태 복원 및 저장
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, output_path)
        logging.info(f"Fine-tuned model saved to {output_path}")


if __name__ == "__main__":
    # 현재 4채널 (BVP + ACC_X/Y/Z) 기반 학습
    pretrain_on_dreamt(
        data_dir=r"C:\dreamt_pretrain",
        output_path="dreamt_pretrained_4ch.pth"
    )
    
    # 향후 8채널 확장 예시 (주석 처리)
    # FUTURE_SIGNAL_CHANNELS = ['ir', 'red', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    # pretrain_on_dreamt(
    #     data_dir=r"C:\Users\ahrid\dreamt_pretrain",
    #     output_path="dreamt_pretrained_8ch.pth",
    #     signal_channels=FUTURE_SIGNAL_CHANNELS,
    #     label_column='label'
    # )
    
    # finetune_on_actual(
    #     actual_data_dir=ACTUAL_DATA_DIR,
    #     pretrained_path=PRETRAINED_PATH,
    #     output_path=FINETUNED_PATH
    # )
