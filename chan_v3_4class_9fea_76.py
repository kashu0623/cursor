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
from scipy.signal import welch, find_peaks  # PPG 신호 분석을 위한 scipy.signal 추가
from scipy.integrate import trapezoid  # 적분을 위한 trapezoid 함수 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 전역 설정 상수
SAMPLING_RATE = 64.0        # Hz
WINDOW_DURATION = 30        # seconds
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)  # 1920 samples
# [수정] 4클래스 분류 (W, Light, N3, R)
NUM_CLASSES = 4             # 수면 단계 클래스 수 (W, Light, N3, R)
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
    'W': 0, 'P': 0,    # Wake: 0
    'N1': 1, 'N2': 1,  # Light Sleep: 1
    'N3': 2,           # Deep Sleep: 2
    'R': 3, 'REM': 3   # REM Sleep: 3
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# [수정] 4클래스용 이름 리스트
CLASS_NAMES_4 = ['Wake', 'Light', 'N3', 'REM']


def find_ppg_peaks(bvp_signal, sampling_rate=64.0, distance_min=0.5, prominence=0.1):
    """
    PPG 신호(BVP)에서 심박수 피크를 찾는 함수
    
    Args:
        bvp_signal (np.array): BVP 신호 데이터 (1차원 배열)
        sampling_rate (float): 샘플링 레이트 (Hz, 기본값: 64.0)
        distance_min (float): 피크 간 최소 거리 (초, 기본값: 0.5)
        prominence (float): 피크의 최소 prominence (기본값: 0.1)
        
    Returns:
        tuple: (peak_indices, peak_properties)
            - peak_indices: 피크 위치 인덱스 배열
            - peak_properties: 피크 속성 딕셔너리
    """
    # 샘플링 레이트를 고려한 최소 거리 계산 (샘플 단위)
    distance_samples = int(distance_min * sampling_rate)
    
    # scipy.signal.find_peaks를 사용하여 피크 검출
    peak_indices, peak_properties = find_peaks(
        bvp_signal,
        distance=distance_samples,
        prominence=prominence,
        height=None,  # 높이 제한 없음
        width=None    # 너비 제한 없음
    )
    
    return peak_indices, peak_properties


def calculate_frequency_features(ibi_seconds):
    """
    IBI 데이터로부터 주파수 피처(VLF, LF, HF power, LF/HF ratio)를 계산하는 함수
    
    Args:
        ibi_seconds (np.array): IBI 데이터 (초 단위)
        
    Returns:
        dict: 주파수 영역 HRV 피처를 포함한 딕셔너리
            - vlf: Very Low Frequency power (0.003-0.04 Hz)
            - lf: Low Frequency power (0.04-0.15 Hz)
            - hf: High Frequency power (0.15-0.4 Hz)
            - lf_hf_ratio: LF/HF ratio
    """
    if len(ibi_seconds) < 10:
        return {'vlf': 0.0, 'lf': 0.0, 'hf': 0.0, 'lf_hf_ratio': 0.0}
    
    fs = 4.0
    steps = 1 / fs
    x_interp = np.arange(ibi_seconds.min(), ibi_seconds.max(), steps)
    y_interp = np.interp(x_interp, np.cumsum(ibi_seconds), ibi_seconds)
    
    f, Pxx = welch(y_interp, fs=fs, nperseg=len(y_interp))
    
    vlf_band, lf_band, hf_band = (0.003, 0.04), (0.04, 0.15), (0.15, 0.4)
    
    vlf_power = trapezoid(Pxx[(f >= vlf_band[0]) & (f < vlf_band[1])], f[(f >= vlf_band[0]) & (f < vlf_band[1])])
    lf_power = trapezoid(Pxx[(f >= lf_band[0]) & (f < lf_band[1])], f[(f >= lf_band[0]) & (f < lf_band[1])])
    hf_power = trapezoid(Pxx[(f >= hf_band[0]) & (f < hf_band[1])], f[(f >= hf_band[0]) & (f < hf_band[1])])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
    
    return {'vlf': vlf_power, 'lf': lf_power, 'hf': hf_power, 'lf_hf_ratio': lf_hf_ratio}


def calculate_all_features(bvp_signal, sampling_rate=64.0, distance_min=0.5, prominence=0.1):
    """
    PPG 신호에서 시간 및 주파수 영역 HRV 피처를 모두 계산하는 함수
    
    Args:
        bvp_signal (np.array): BVP 신호 데이터 (1차원 배열)
        sampling_rate (float): 샘플링 레이트 (Hz, 기본값: 64.0)
        distance_min (float): 피크 간 최소 거리 (초, 기본값: 0.5)
        prominence (float): 피크의 최소 prominence (기본값: 0.1)
        
    Returns:
        dict: 시간 및 주파수 영역 HRV 피처를 포함한 딕셔너리
            - hr: 분당 심박수 (float)
            - rmssd: RMSSD HRV 지표 (float)
            - peak_count: 검출된 피크 수 (int)
            - ibi_mean: 평균 IBI (초, float)
            - ibi_std: IBI 표준편차 (초, float)
            - vlf: Very Low Frequency power (0.003-0.04 Hz)
            - lf: Low Frequency power (0.04-0.15 Hz)
            - hf: High Frequency power (0.15-0.4 Hz)
            - lf_hf_ratio: LF/HF ratio
    """
    peak_indices, _ = find_ppg_peaks(bvp_signal, sampling_rate, distance_min, prominence)
    
    # 기본 피처 초기화
    time_features = {
        'hr': 0.0, 
        'rmssd': 0.0, 
        'peak_count': len(peak_indices), 
        'ibi_mean': 0.0, 
        'ibi_std': 0.0
    }
    freq_features = {
        'vlf': 0.0, 
        'lf': 0.0, 
        'hf': 0.0, 
        'lf_hf_ratio': 0.0
    }
    
    if len(peak_indices) >= 2:
        ibi_seconds = np.diff(peak_indices) / sampling_rate
        
        if len(ibi_seconds) > 0:
            ibi_mean = np.mean(ibi_seconds)
            time_features['hr'] = 60.0 / ibi_mean if ibi_mean > 0 else 0.0
            time_features['ibi_mean'] = ibi_mean
            time_features['ibi_std'] = np.std(ibi_seconds)
            
            if len(ibi_seconds) >= 2:
                time_features['rmssd'] = np.sqrt(np.mean(np.diff(ibi_seconds) ** 2))
            
            freq_features = calculate_frequency_features(ibi_seconds)
    
    return {**time_features, **freq_features}


def map_sleep_stage(raw_label):
    return LABEL_MAP.get(str(raw_label).strip().upper(), None)


def extract_epochs_with_features_from_df(df, majority_ratio=MAJORITY_RATIO, signal_channels=None, label_column=None):
    """
    DataFrame에서 epoch별로 raw signal, HR/HRV 피처, 대표 라벨을 추출하는 함수 (majority voting)
    
    Args:
        df (pd.DataFrame): DataFrame with signal and label columns
        majority_ratio (float): majority voting 비율 (기본값: 0.6)
        signal_channels (list): 사용할 신호 채널 리스트 (기본값: CURRENT_SIGNAL_CHANNELS)
        label_column (str): 라벨 컬럼명 (기본값: CURRENT_LABEL_COLUMN)
        
    Returns:
        tuple: (raw_signals, features, labels) 
            - raw_signals: np.array of shape (N, num_channels, 1920) - 원시 신호 데이터
            - features: np.array of shape (N, 9) - HR/HRV 피처 (HR, RMSSD, PeakCount, IBI_mean, IBI_std, VLF, LF, HF, LF/HF_ratio)
            - labels: np.array of shape (N,) - 수면 단계 라벨
    """
    if signal_channels is None:
        signal_channels = CURRENT_SIGNAL_CHANNELS
    if label_column is None:
        label_column = CURRENT_LABEL_COLUMN
        
    raw_signals = []
    features = []
    labels = []
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
        labels_epoch = epoch_df[label_column].map(map_sleep_stage).dropna().astype(int).values
        if len(labels_epoch) == 0:
            logging.warning(f"Epoch {i}: no valid labels found")
            continue
            
        label_counts = Counter(labels_epoch)
        major_label, count = label_counts.most_common(1)[0]
        ratio = count / len(labels_epoch)
        
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
            
            # 수정: 5개 HR/HRV 피처 모두 계산 (BVP 채널이 있는 경우)
            hr_feature = 0.0
            rmssd_feature = 0.0
            peak_count_feature = 0.0
            ibi_mean_feature = 0.0
            ibi_std_feature = 0.0
            
            if 'bvp' in signal_channels:
                bvp_index = signal_channels.index('bvp')
                bvp_signal = signals[bvp_index]
                
                # 9개 HR/HRV 피처 모두 계산 (시간 + 주파수 영역)
                all_features_dict = calculate_all_features(bvp_signal, sampling_rate=SAMPLING_RATE)
                hr_feature = all_features_dict['hr']
                rmssd_feature = all_features_dict['rmssd']
                peak_count_feature = all_features_dict['peak_count']
                ibi_mean_feature = all_features_dict['ibi_mean']
                ibi_std_feature = all_features_dict['ibi_std']
                vlf_feature = all_features_dict['vlf']
                lf_feature = all_features_dict['lf']
                hf_feature = all_features_dict['hf']
                lf_hf_ratio_feature = all_features_dict['lf_hf_ratio']
                
                logging.debug(f"Epoch {i}: HR={hr_feature:.1f}, RMSSD={rmssd_feature:.3f}, PeakCount={peak_count_feature}, IBI_mean={ibi_mean_feature:.3f}, IBI_std={ibi_std_feature:.3f}, VLF={vlf_feature:.3f}, LF={lf_feature:.3f}, HF={hf_feature:.3f}, LF/HF={lf_hf_ratio_feature:.3f}")
            
            # 수정: 9개 피처 모두 포함한 배열 생성 [HR, RMSSD, PeakCount, IBI_mean, IBI_std, VLF, LF, HF, LF/HF_ratio]
            feature_vector = np.array([hr_feature, rmssd_feature, peak_count_feature, ibi_mean_feature, ibi_std_feature, vlf_feature, lf_feature, hf_feature, lf_hf_ratio_feature], dtype=np.float32)
                
            raw_signals.append(signals)
            features.append(feature_vector)
            labels.append(major_label)
            
        except Exception as e:
            logging.warning(f"Error extracting signals for epoch {i}: {e}")
            continue
    
    logging.info(f"Successfully extracted {len(raw_signals)} valid epochs with HR/HRV features")
    return np.array(raw_signals), np.array(features), np.array(labels)


def load_dreamt_data(data_dir, majority_ratio=MAJORITY_RATIO, signal_channels=None, label_column=None):
    """
    DREAMT 데이터 로드 (raw signal + features + labels 모두 반환)
    
    Args:
        data_dir (str): DREAMT 데이터 디렉토리 경로
        majority_ratio (float): majority voting 비율 (기본값: 0.6)
        signal_channels (list): 사용할 신호 채널 리스트 (기본값: CURRENT_SIGNAL_CHANNELS)
        label_column (str): 라벨 컬럼명 (기본값: CURRENT_LABEL_COLUMN)
        
    Returns:
        tuple: (raw_signals, features, labels) 
            - raw_signals: np.array of shape (N, num_channels, 1920) - 원시 신호 데이터
            - features: np.array of shape (N, 9) - HR/HRV 피처 (9개 값)
            - labels: np.array of shape (N,) - 수면 단계 라벨
    """
    if signal_channels is None:
        signal_channels = CURRENT_SIGNAL_CHANNELS
    if label_column is None:
        label_column = CURRENT_LABEL_COLUMN
        
    # 수정: features 리스트 추가하여 3개 배열 모두 수집
    X_list, X_features_list, y_list = [], [], []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        logging.warning(f"No CSV files found in {data_dir}")
        # 수정: 3개 빈 배열 반환하여 형태 통일
        return np.array([]), np.array([]), np.array([])
        
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
                
            raw_signals, features, labels = extract_epochs_with_features_from_df(df, majority_ratio=majority_ratio, 
                                        signal_channels=signal_channels, label_column=label_column)
            
            if len(raw_signals) > 0:
                X_list.append(raw_signals)
                X_features_list.append(features)  # 수정: features도 수집
                y_list.append(labels)
                logging.info(f"Successfully loaded {len(raw_signals)} epochs from {os.path.basename(csv_file)}")
            else:
                logging.warning(f"No valid epochs extracted from {os.path.basename(csv_file)}")
                
        except Exception as e:
            logging.warning(f"Error loading {os.path.basename(csv_file)}: {e}")
            continue
    
    if X_list:
        raw_signals = np.concatenate(X_list, axis=0)
        # 수정: features 배열도 concatenate하여 생성
        features_all = np.concatenate(X_features_list, axis=0)
        labels = np.concatenate(y_list, axis=0)
        logging.info(f"Total loaded: {len(raw_signals)} epochs from {len(X_list)} files")
        logging.info(f"Features shape: {features_all.shape}")  # 수정: features 정보 로깅
        # 수정: 3개 배열 모두 반환
        return raw_signals, features_all, labels
    else:
        logging.error("No valid data loaded from any files")
        # 수정: 3개 빈 배열 반환하여 형태 통일
        return np.array([]), np.array([]), np.array([])


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


class SleepDualInputDataset(Dataset):
    """
    듀얼 입력 수면 데이터셋: 원시 신호와 HR/HRV 피처를 동시에 제공
    
    이 데이터셋은 두 가지 입력을 제공합니다:
    1. raw_signals: 원시 신호 데이터 (BVP, 가속도 등)
    2. features: HR/HRV 피처 벡터 (심박수, RMSSD)
    """
    def __init__(self, raw_signals, features, labels):
        """
        Args:
            raw_signals: 원시 신호 데이터 (N, num_channels, 1920)
            features: HR/HRV 피처 (N, 9) - [HR, RMSSD, PeakCount, IBI_mean, IBI_std, VLF, LF, HF, LF/HF_ratio]
            labels: 수면 단계 라벨 (N,)
        """
        self.raw_signals = raw_signals.astype(np.float32)
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        
        # 데이터 검증
        if len(self.raw_signals) != len(self.features) or len(self.raw_signals) != len(self.labels):
            raise ValueError(f"Data lengths don't match: raw_signals={len(self.raw_signals)}, "
                           f"features={len(self.features)}, labels={len(self.labels)}")
        
        if len(self.raw_signals.shape) != 3 or self.raw_signals.shape[2] != 1920:
            raise ValueError(f"Expected raw_signals shape (N, num_channels, 1920), got {self.raw_signals.shape}")
        
        # 수정: 피처 차원을 9로 변경 (HR, RMSSD, PeakCount, IBI_mean, IBI_std, VLF, LF, HF, LF/HF_ratio)
        if len(self.features.shape) != 2 or self.features.shape[1] != 9:
            raise ValueError(f"Expected features shape (N, 9), got {self.features.shape}")
        
        self.num_channels = self.raw_signals.shape[1]
        logging.info(f"SleepDualInputDataset initialized with {len(self.raw_signals)} samples")
        logging.info(f"  Raw signals shape: {self.raw_signals.shape}")
        logging.info(f"  Features shape: {self.features.shape}")
        logging.info(f"  Labels shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.raw_signals)
    
    def __getitem__(self, idx):
        """
        데이터 샘플 반환
        
        Returns:
            tuple: (x_raw, x_features, y)
                - x_raw: 원시 신호 (1920, num_channels) - LSTM용
                - x_features: HR/HRV 피처 (9,) - MLP용
                - y: 라벨 (스칼라)
        """
        # 원시 신호: (num_channels, 1920) -> (1920, num_channels) for LSTM
        x_raw = torch.from_numpy(self.raw_signals[idx].T)  # shape: (1920, num_channels)
        
        # HR/HRV 피처: (9,) -> (9,) for MLP
        x_features = torch.from_numpy(self.features[idx])   # shape: (9,)
        
        # 라벨
        y = torch.tensor(self.labels[idx])
        
        return x_raw, x_features, y


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


class DualInputLSTMClassifier(nn.Module):
    """
    듀얼 입력 LSTM 분류기: 원시 신호와 HR/HRV 피처를 동시에 처리하는 모델
    
    이 모델은 두 가지 입력을 받아 수면 단계를 분류합니다:
    1. x_raw: 원시 신호 데이터 (BVP, 가속도 등)
    2. x_features: HR/HRV 피처 벡터 (심박수, RMSSD, 피크수, IBI 평균, IBI 표준편차)
    """
    def __init__(self, raw_input_size=None, feature_input_size=9, lstm_hidden_size=64, 
                 lstm_num_layers=2, mlp_hidden_size=32, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        
        # 입력 크기 설정
        if raw_input_size is None:
            raw_input_size = len(CURRENT_SIGNAL_CHANNELS)
        self.raw_input_size = raw_input_size
        self.feature_input_size = feature_input_size
        
        # === LSTM 브랜치: 원시 신호 처리 ===
        # 양방향 LSTM으로 원시 신호의 시계열 패턴 학습
        self.lstm = nn.LSTM(
            input_size=raw_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,  # 양방향으로 문맥 정보 활용
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # LSTM 출력 차원: 양방향이므로 hidden_size * 2
        lstm_output_size = lstm_hidden_size * 2
        
        # === MLP 브랜치: HR/HRV 피처 처리 ===
        # 첫 번째 레이어: 피처 벡터를 중간 차원으로 확장
        self.mlp_layer1 = nn.Linear(feature_input_size, mlp_hidden_size)
        self.mlp_activation = nn.ReLU()
        self.mlp_dropout = nn.Dropout(dropout)
        
        # 두 번째 레이어: 중간 차원을 유지하면서 특징 추출
        self.mlp_layer2 = nn.Linear(mlp_hidden_size, mlp_hidden_size)
        
        # === 결합 및 분류 ===
        # LSTM과 MLP 출력을 결합
        combined_size = lstm_output_size + mlp_hidden_size
        
        # 최종 분류 레이어
        self.classifier = nn.Linear(combined_size, num_classes)
        
        # 가중치 초기화
        self._init_weights()
        
        logging.info(f"DualInputLSTMClassifier initialized:")
        logging.info(f"  Raw input size: {raw_input_size}")
        logging.info(f"  Feature input size: {feature_input_size}")
        logging.info(f"  LSTM hidden size: {lstm_hidden_size}")
        logging.info(f"  MLP hidden size: {mlp_hidden_size}")
        logging.info(f"  Combined size: {combined_size}")
        logging.info(f"  Num classes: {num_classes}")
    
    def _init_weights(self):
        """모든 레이어의 가중치를 적절하게 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM 가중치는 기본 초기화 사용 (PyTorch 내장)
                    continue
                else:
                    # Linear 레이어는 Xavier 초기화
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 편향은 0으로 초기화
                nn.init.constant_(param, 0)
    
    def forward(self, x_raw, x_features):
        """
        순전파 함수
        
        Args:
            x_raw: 원시 신호 데이터 (batch_size, seq_len, raw_input_size)
            x_features: HR/HRV 피처 벡터 (batch_size, feature_input_size)
            
        Returns:
            logits: 클래스별 예측 점수 (batch_size, num_classes)
        """
        batch_size = x_raw.size(0)
        
        # === LSTM 브랜치 처리 ===
        # x_raw: (batch, seq_len, raw_input_size) -> LSTM 처리
        lstm_out, _ = self.lstm(x_raw)
        # lstm_out: (batch, seq_len, lstm_hidden_size * 2)
        
        # 마지막 타임스텝의 출력만 사용 (시계열 정보의 최종 요약)
        lstm_final = lstm_out[:, -1, :]  # (batch, lstm_hidden_size * 2)
        
        # === MLP 브랜치 처리 ===
        # 첫 번째 레이어: 피처 벡터 확장
        mlp_out = self.mlp_layer1(x_features)  # (batch, mlp_hidden_size)
        mlp_out = self.mlp_activation(mlp_out)  # ReLU 활성화
        mlp_out = self.mlp_dropout(mlp_out)    # Dropout 적용
        
        # 두 번째 레이어: 특징 추출
        mlp_out = self.mlp_layer2(mlp_out)     # (batch, mlp_hidden_size)
        mlp_out = self.mlp_activation(mlp_out)  # ReLU 활성화
        mlp_out = self.mlp_dropout(mlp_out)    # Dropout 적용
        
        # === 두 브랜치 출력 결합 ===
        # torch.cat으로 LSTM과 MLP 출력을 연결
        combined_features = torch.cat([lstm_final, mlp_out], dim=1)
        # combined_features: (batch, lstm_hidden_size * 2 + mlp_hidden_size)
        
        # === 최종 분류 ===
        logits = self.classifier(combined_features)
        # logits: (batch, num_classes)
        
        return logits


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
    print(classification_report(all_targets, all_preds, digits=4, target_names=CLASS_NAMES_4))


def pretrain_on_dreamt_dual_input(data_dir, output_path, epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE, 
                                 lr=LR_PRETRAIN, majority_ratio=MAJORITY_RATIO, signal_channels=None, 
                                 label_column=None, class_weights=None):
    """
    DREAMT 데이터로 듀얼 입력 모델 Pre-train (원시 신호 + HR/HRV 피처)
    
    Args:
        data_dir (str): DREAMT 데이터 디렉토리 경로
        output_path (str): 모델 저장 경로
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        lr (float): 학습률
        majority_ratio (float): majority voting 비율
        signal_channels (list): 사용할 신호 채널 리스트 (기본값: CURRENT_SIGNAL_CHANNELS)
        label_column (str): 라벨 컬럼명 (기본값: CURRENT_LABEL_COLUMN)
        class_weights (array): 클래스별 가중치 (기본값: None, 자동 계산)
    """
    if signal_channels is None:
        signal_channels = CURRENT_SIGNAL_CHANNELS
    if label_column is None:
        label_column = CURRENT_LABEL_COLUMN
        
    logging.info("=== 듀얼 입력 모델로 DREAMT 데이터 로딩 시작 ===")
    
    # 수정: 중복된 데이터 로딩 로직 제거하고 load_dreamt_data 함수 사용
    raw_signals, features, labels = load_dreamt_data(
        data_dir, majority_ratio, signal_channels, label_column
    )
    
    if len(raw_signals) == 0:
        logging.error("No valid data loaded from any files!")
        return
    
    logging.info(f"=== 전체 데이터 로딩 완료 ===")
    logging.info(f"Total epochs: {len(raw_signals)}")
    logging.info(f"Raw signals shape: {raw_signals.shape}")
    logging.info(f"Features shape: {features.shape}")
    logging.info(f"Labels shape: {labels.shape}")
    
    # 클래스 분포 확인
    unique, counts = np.unique(labels, return_counts=True)
    logging.info("Class distribution:")
    for label, count in zip(unique, counts):
        logging.info(f"  {INV_LABEL_MAP[label]}: {count} ({count/len(labels)*100:.1f}%)")
    
    # 데이터 분할 (stratify 사용하여 클래스 균형 유지)
    raw_signals_train, raw_signals_test, features_train, features_test, labels_train, labels_test = train_test_split(
        raw_signals, features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 듀얼 입력 데이터셋 생성
    train_dataset = SleepDualInputDataset(raw_signals_train, features_train, labels_train)
    test_dataset = SleepDualInputDataset(raw_signals_test, features_test, labels_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 듀얼 입력 모델 초기화
    raw_input_size = len(signal_channels)
    # 수정: feature_input_size를 9로 변경 (HR, RMSSD, PeakCount, IBI_mean, IBI_std, VLF, LF, HF, LF/HF_ratio)
    feature_input_size = 9
    model = DualInputLSTMClassifier(
        raw_input_size=raw_input_size,
        feature_input_size=feature_input_size,
        lstm_hidden_size=64,
        lstm_num_layers=2,
        mlp_hidden_size=32,
        num_classes=NUM_CLASSES,
        dropout=0.2
    )
    
    # GPU 사용 가능시 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"Using device: {device}")
    
    # 클래스별 가중치 설정 (클래스 불균형 해결)
    if class_weights is None:
        # [수정] 클래스 가중치 계산 방식을 로그 스케일로 변경하여 정밀도-재현율 균형 조절
        class_counts = np.bincount(labels_train)
        class_weights = 1.0 / np.log1p(class_counts)
        # 무한대 값 방지
        class_weights[np.isinf(class_weights)] = 1.0
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
    
    logging.info("=== 듀얼 입력 모델 Pre-training 시작 ===")
    
    # 학습 루프
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for batch_data in train_loader:
            # 듀얼 입력 데이터 언패킹
            batch_x_raw, batch_x_features, batch_y = batch_data
            batch_x_raw, batch_x_features, batch_y = batch_x_raw.to(device), batch_x_features.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # 듀얼 입력 모델에 두 입력 전달
            outputs = model(batch_x_raw, batch_x_features)
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
            for batch_data in test_loader:
                # 듀얼 입력 데이터 언패킹
                batch_x_raw, batch_x_features, batch_y = batch_data
                batch_x_raw, batch_x_features, batch_y = batch_x_raw.to(device), batch_x_features.to(device), batch_y.to(device)
                
                # 듀얼 입력 모델에 두 입력 전달
                outputs = model(batch_x_raw, batch_x_features)
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
        logging.info(f"Dual input pre-trained model saved to {output_path}")
    
    # === 평가 메트릭 출력 ===
    logging.info("\n=== 듀얼 입력 모델 최종 성능 ===")
    logging.info("Confusion Matrix (Validation Set):")
    print(confusion_matrix(all_targets, all_preds))
    logging.info("Classification Report (Validation Set):")
    # 수정: 버그 수정 - all_targets를 all_preds로 변경하여 정확한 평가 가능
    print(classification_report(all_targets, all_preds, digits=4, target_names=CLASS_NAMES_4))
    
    return model


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
    print("=== 수면 단계 분류 모델 학습 시작 ===")
    
    try:
        # 최종 듀얼 입력 모델 학습을 실행합니다.
        pretrain_on_dreamt_dual_input(
            data_dir=r"C:\\dreamt_pretrain",
            output_path="dreamt_pretrained_4class_9features.pth"
        )
        print("\n🎉 학습이 성공적으로 완료되었습니다!")
    except Exception as e:
        print(f"\n❌ 학습 중 오류가 발생했습니다: {e}")


def test_ppg_analysis():
    """
    PPG 신호 분석 기능을 테스트하는 함수
    """
    print("=== PPG 신호 분석 기능 테스트 ===")
    
    # 테스트용 PPG 신호 생성 (가상의 심박수 패턴)
    sampling_rate = 64.0
    duration = 30  # 30초
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # 기본 심박수: 60 BPM (1초마다 피크)
    base_hr = 60
    base_period = 60.0 / base_hr  # 1초
    
    # 가상 PPG 신호 생성 (여러 주파수 성분 포함)
    ppg_signal = (
        0.5 * np.sin(2 * np.pi * base_hr / 60 * t) +  # 기본 심박수
        0.2 * np.sin(2 * np.pi * 2 * base_hr / 60 * t) +  # 2차 고조파
        0.1 * np.sin(2 * np.pi * 3 * base_hr / 60 * t) +  # 3차 고조파
        0.05 * np.random.randn(len(t))  # 노이즈
    )
    
    print(f"생성된 PPG 신호 길이: {len(ppg_signal)} 샘플")
    print(f"샘플링 레이트: {sampling_rate} Hz")
    print(f"신호 지속시간: {duration}초")
    
    # 피크 검출 테스트
    print("\n--- 피크 검출 테스트 ---")
    peak_indices, peak_properties = find_ppg_peaks(ppg_signal, sampling_rate=sampling_rate)
    print(f"검출된 피크 수: {len(peak_indices)}")
    if len(peak_indices) > 0:
        print(f"첫 번째 피크 위치: {peak_indices[0]} 샘플")
        print(f"마지막 피크 위치: {peak_indices[-1]} 샘플")
    
    # HR과 HRV 피처 계산 테스트
    print("\n--- HR/HRV 피처 계산 테스트 ---")
    hr_hrv_features = calculate_all_features(ppg_signal, sampling_rate=sampling_rate)
    
    print("계산된 피처:")
    for key, value in hr_hrv_features.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # 이론값과 비교
    expected_hr = base_hr
    actual_hr = hr_hrv_features['hr']
    hr_error = abs(actual_hr - expected_hr)
    
    print(f"\n--- 정확도 검증 ---")
    print(f"예상 심박수: {expected_hr} BPM")
    print(f"실제 심박수: {actual_hr:.1f} BPM")
    print(f"오차: {hr_error:.1f} BPM")
    
    if hr_error < 5:  # 5 BPM 이내 오차면 성공
        print("✅ 심박수 검출 성공!")
    else:
        print("❌ 심박수 검출 실패 - 파라미터 조정 필요")
    
    return ppg_signal, hr_hrv_features


def test_epoch_extraction_with_features():
    """
    새로운 epoch 추출 기능을 테스트하는 함수
    """
    print("\n=== Epoch 추출 + 피처 계산 테스트 ===")
    
    # 테스트용 DataFrame 생성
    sampling_rate = 64.0
    duration = 60  # 60초 (2개 epoch)
    n_samples = int(sampling_rate * duration)
    
    # 가상 데이터 생성
    t = np.linspace(0, duration, n_samples)
    
    # BVP 신호 (심박수 변화 포함)
    bvp_signal = (
        0.5 * np.sin(2 * np.pi * 60 / 60 * t) +  # 60 BPM
        0.2 * np.sin(2 * np.pi * 120 / 60 * t) +  # 2차 고조파
        0.1 * np.random.randn(n_samples)  # 노이즈
    )
    
    # 가속도 신호들
    acc_x = 0.1 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(n_samples)
    acc_y = 0.1 * np.cos(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(n_samples)
    acc_z = 0.1 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(n_samples)
    
    # 수면 단계 라벨 (30초마다 변화)
    sleep_stages = []
    for i in range(n_samples):
        if i < n_samples // 2:
            sleep_stages.append('W')  # 첫 30초: 깨어있음
        else:
            sleep_stages.append('N2')  # 후 30초: N2 수면
    
    # DataFrame 생성
    test_df = pd.DataFrame({
        'bvp': bvp_signal,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'sleep_stage': sleep_stages
    })
    
    print(f"테스트 DataFrame 생성 완료: {len(test_df)} 샘플")
    print(f"사용할 채널: {CURRENT_SIGNAL_CHANNELS}")
    print(f"Epoch 크기: {WINDOW_SIZE} 샘플 ({WINDOW_DURATION}초)")
    
    # 새로운 epoch 추출 함수 테스트
    try:
        raw_signals, features, labels = extract_epochs_with_features_from_df(
            test_df, 
            majority_ratio=0.6,
            signal_channels=CURRENT_SIGNAL_CHANNELS,
            label_column=CURRENT_LABEL_COLUMN
        )
        
        print(f"\n✅ Epoch 추출 성공!")
        print(f"추출된 epoch 수: {len(raw_signals)}")
        print(f"Raw signals shape: {raw_signals.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
                    # 첫 번째 epoch의 피처 확인 (9개 피처 모두 표시)
        if len(features) > 0:
            first_hr = features[0][0]
            first_rmssd = features[0][1]
            first_peak_count = features[0][2]
            first_ibi_mean = features[0][3]
            first_ibi_std = features[0][4]
            first_label = INV_LABEL_MAP.get(labels[0], 'Unknown')
            
            print(f"\n첫 번째 epoch 정보:")
            print(f"  라벨: {first_label}")
            print(f"  심박수: {first_hr:.1f} BPM")
            print(f"  RMSSD: {first_rmssd:.3f}")
            print(f"  피크 수: {first_peak_count}")
            print(f"  IBI 평균: {first_ibi_mean:.3f}초")
            print(f"  IBI 표준편차: {first_ibi_std:.3f}초")
        
        return raw_signals, features, labels
        
    except Exception as e:
        print(f"❌ Epoch 추출 실패: {e}")
        return None, None, None


def test_dual_input_model():
    """
    듀얼 입력 모델의 동작을 테스트하는 함수
    """
    print("\n=== 듀얼 입력 모델 테스트 ===")
    
    # 테스트용 데이터 생성
    batch_size = 4
    seq_len = 1920
    num_channels = 4
    # 수정: feature_size를 9로 변경 (9개 HR/HRV 피처)
    feature_size = 9
    num_classes = 4
    
    # 가상 데이터 생성
    x_raw = torch.randn(batch_size, seq_len, num_channels)      # 원시 신호
    x_features = torch.randn(batch_size, feature_size)          # HR/HRV 피처
    y = torch.randint(0, num_classes, (batch_size,))           # 라벨
    
    print(f"테스트 데이터 생성 완료:")
    print(f"  x_raw shape: {x_raw.shape}")
    print(f"  x_features shape: {x_features.shape}")
    print(f"  y shape: {y.shape}")
    
    # 듀얼 입력 모델 생성
    try:
        model = DualInputLSTMClassifier(
            raw_input_size=num_channels,
            feature_input_size=feature_size,
            lstm_hidden_size=32,  # 테스트용으로 작게 설정
            lstm_num_layers=1,    # 테스트용으로 간단하게
            mlp_hidden_size=16,   # 테스트용으로 작게 설정
            num_classes=num_classes,
            dropout=0.1
        )
        
        print(f"\n✅ 듀얼 입력 모델 생성 성공!")
        print(f"모델 구조:")
        print(f"  LSTM: {num_channels} -> {32*2} (양방향)")
        print(f"  MLP: {feature_size} -> {16} -> {16}")
        print(f"  결합: {32*2 + 16} -> {num_classes}")
        
        # 모델에 테스트 데이터 전달
        model.eval()
        with torch.no_grad():
            outputs = model(x_raw, x_features)
            
        print(f"\n✅ 순전파 성공!")
        print(f"  입력: x_raw {x_raw.shape}, x_features {x_features.shape}")
        print(f"  출력: {outputs.shape}")
        print(f"  예측 클래스: {torch.argmax(outputs, dim=1)}")
        
        # 손실 계산 테스트
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y)
        print(f"  테스트 손실: {loss.item():.4f}")
        
        return model, x_raw, x_features, y
        
    except Exception as e:
        print(f"❌ 듀얼 입력 모델 테스트 실패: {e}")
        return None, None, None, None
