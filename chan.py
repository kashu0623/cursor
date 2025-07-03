import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import pandas as pd
from scipy import signal as scipy_signal
import heartpy as hp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import mne
import glob
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 전역 설정 상수
SAMPLING_RATE = 25.0        # Hz
WINDOW_DURATION = 30        # seconds
STRIDE_DURATION = 10        # seconds
QUALITY_THRESHOLD = 0.5     # 신호 품질 임계값
NUM_CLASSES = 5             # 수면 단계 클래스 수 (W, N1, N2, N3, R)

# 계산된 상수
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)
STRIDE_SIZE = int(SAMPLING_RATE * STRIDE_DURATION)

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

def map_sleep_stage(raw_stage):
    """수면 단계 매핑 함수
    
    Args:
        raw_stage (str): 원본 수면 단계 어노테이션
        
    Returns:
        int: 매핑된 수면 단계 (0=W, 1=N1, 2=N2, 3=N3, 4=R)
    """
    if raw_stage == 'W':
        return 0   # Wake
    elif raw_stage == 'N1':
        return 1   # N1
    elif raw_stage == 'N2':
        return 2   # N2
    elif raw_stage == 'N3':
        return 3   # Deep (N3)
    elif raw_stage == 'R':
        return 4   # REM
    else:
        return None

class SleepDataPreprocessor:
    """수면 데이터 전처리 클래스
    
    PPG 신호(IR/RED)로부터 생체 신호를 추출하고 전처리하는 클래스입니다.
    심박수, HRV, 호흡률, SpO2 등의 생체 지표를 계산합니다.
    """
    
    def __init__(self, sampling_rate=SAMPLING_RATE, quality_threshold=QUALITY_THRESHOLD):
        """초기화 함수
        
        Args:
            sampling_rate (float): 데이터 샘플링 주파수 (Hz)
            quality_threshold (float): 신호 품질 임계값
        """
        self.sampling_rate = sampling_rate
        self.quality_threshold = quality_threshold
        
    def extract_ppg_features(self, ir_signal, red_signal):
        """PPG 신호로부터 생체 지표 추출
        
        Args:
            ir_signal (np.array): IR 센서 신호
            red_signal (np.array): RED 센서 신호
            
        Returns:
            tuple: (심박수, HRV, 호흡률, SpO2, 신호품질지수)
        """
        # 신호 품질 검사
        quality_index = self._calculate_signal_quality(ir_signal, red_signal)
        if quality_index < self.quality_threshold:  # 품질이 낮은 경우
            return None, None, None, None, quality_index
            
        # IR 신호만 사용하여 심박수 및 HRV 계산
        filtered_ir = self._filter_signal(ir_signal)
        
        try:
            working_data, measures = hp.process(filtered_ir, self.sampling_rate)
            hr = measures['bpm']  # 심박수 (beats per minute)
            hrv = measures['sdnn']  # 심박변이도 (Standard Deviation of NN intervals)
        except Exception:
            # HeartPy 처리 실패 시 품질 임계값으로 스킵
            return None, None, None, None, quality_index
        
        # 호흡률 추출 (IR 신호 기반)
        rr = self._extract_respiratory_rate(filtered_ir)
        
        # SpO2 계산 (IR/RED 신호 비율 기반)
        spo2 = self._calculate_spo2(ir_signal, red_signal)
        
        return hr, hrv, rr, spo2, quality_index
    
    def _filter_signal(self, input_signal):
        """신호 필터링
        
        밴드패스 필터를 사용하여 심박수 대역(0.5-4Hz)의 신호만 추출합니다.
        
        Args:
            input_signal (np.array): 입력 신호
            
        Returns:
            np.array: 필터링된 신호
        """
        # 밴드패스 필터 적용 (0.5-4Hz: 심박수 대역)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 4.0 / nyquist
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, input_signal)
    
    def _extract_respiratory_rate(self, input_signal):
        """호흡률 추출
        
        FFT를 사용하여 호흡 대역(0.1-0.4Hz)의 주파수를 찾아 호흡률을 계산합니다.
        
        Args:
            input_signal (np.array): 입력 신호
            
        Returns:
            float: 분당 호흡수
        """
        # FFT 적용
        fft = np.fft.fft(input_signal)
        freqs = np.fft.fftfreq(len(input_signal), 1/self.sampling_rate)
        
        # 호흡 대역 (0.1-0.4Hz)에서 최대 주파수 찾기
        resp_mask = (freqs >= 0.1) & (freqs <= 0.4)
        resp_freq = freqs[resp_mask][np.argmax(np.abs(fft[resp_mask]))]
        
        return abs(resp_freq) * 60  # 분당 호흡수로 변환
    
    def _calculate_spo2(self, ir_signal, red_signal):
        """SpO2 계산
        
        IR/RED 신호의 AC/DC 비율을 이용하여 산소포화도를 계산합니다.
        참고: 이는 추정치일 뿐이며, 실제 기기에서는 보정이 필요합니다.
        
        Args:
            ir_signal (np.array): IR 센서 신호
            red_signal (np.array): RED 센서 신호
            
        Returns:
            float: 산소포화도 (%)
        """
        # AC/DC 성분 분리
        ir_ac = self._extract_ac_component(ir_signal)
        ir_dc = self._extract_dc_component(ir_signal)
        red_ac = self._extract_ac_component(red_signal)
        red_dc = self._extract_dc_component(red_signal)
        
        # R 값 계산 (SpO2 계산에 사용되는 비율) - 스칼라 값으로 계산
        r_value = (red_ac/red_dc) / (ir_ac/ir_dc)
        
        # R 값을 SpO2로 변환 (근사 공식)
        # 참고: 이는 추정치일 뿐이며, 실제 기기에서는 보정이 필요합니다
        spo2 = 110 - 25 * r_value
        spo2 = np.clip(spo2, 70, 100)  # 70-100% 범위로 제한
        
        return spo2
    
    def _extract_ac_component(self, input_signal):
        """신호의 AC 성분 추출
        
        고주파 필터를 사용하여 신호의 AC(교류) 성분을 추출하고 스칼라 지표로 반환합니다.
        
        Args:
            input_signal (np.array): 입력 신호
            
        Returns:
            float: AC 성분의 peak-to-peak 값
        """
        # 고주파 성분 추출 (AC 성분)
        nyquist = self.sampling_rate / 2
        high = 0.5 / nyquist
        b, a = scipy_signal.butter(4, high, btype='high')
        ac_signal = scipy_signal.filtfilt(b, a, input_signal)
        return np.ptp(ac_signal)  # peak-to-peak 값 반환
    
    def _extract_dc_component(self, input_signal):
        """신호의 DC 성분 추출
        
        저주파 필터를 사용하여 신호의 DC(직류) 성분을 추출하고 스칼라 지표로 반환합니다.
        
        Args:
            input_signal (np.array): 입력 신호
            
        Returns:
            float: DC 성분의 평균값
        """
        # 저주파 성분 추출 (DC 성분)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        b, a = scipy_signal.butter(4, low, btype='low')
        dc_signal = scipy_signal.filtfilt(b, a, input_signal)
        return np.mean(np.abs(dc_signal))  # 평균값 반환
        
    def _calculate_signal_quality(self, ir_signal, red_signal):
        """신호 품질 지수 계산
        
        Args:
            ir_signal (np.array): IR 센서 신호
            red_signal (np.array): RED 센서 신호
            
        Returns:
            float: 신호 품질 지수 (0-1)
        """
        # 신호 대 잡음비 (SNR) 계산 - 분모 0 방지를 위해 작은 값 추가
        ir_snr = np.mean(np.abs(ir_signal)) / (np.std(ir_signal) + 1e-6)
        red_snr = np.mean(np.abs(red_signal)) / (np.std(red_signal) + 1e-6)
        
        # 신호 품질 지수 계산 (0-1 범위)
        quality = (ir_snr + red_snr) / 2
        quality = np.clip(quality / 10, 0, 1)  # 10을 기준으로 정규화
        
        return quality

class MotionFeatureExtractor:
    """움직임 특징 추출 클래스
    
    가속도계와 자이로스코프 데이터로부터 움직임 관련 특징을 추출합니다.
    """
    
    def __init__(self):
        pass
    
    def extract_features(self, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
        """움직임 특징 추출
        
        Args:
            acc_x, acc_y, acc_z (np.array): 3축 가속도계 데이터
            gyro_x, gyro_y, gyro_z (np.array): 3축 자이로스코프 데이터
            
        Returns:
            dict: 추출된 움직임 특징들
        """
        features = {}
        
        # 가속도계 특징
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        features['acc_mean'] = np.mean(acc_magnitude)  # 평균 가속도
        features['acc_std'] = np.std(acc_magnitude)    # 가속도 표준편차
        features['acc_variance'] = np.var(acc_magnitude)  # 가속도 분산
        
        # 자이로스코프 특징
        gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        features['gyro_mean'] = np.mean(gyro_magnitude)  # 평균 각속도
        features['gyro_std'] = np.std(gyro_magnitude)    # 각속도 표준편차
        features['gyro_variance'] = np.var(gyro_magnitude)  # 각속도 분산
        
        # 자세 변화 지표
        features['tilt_x'] = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2))  # X축 기울기
        features['tilt_y'] = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))  # Y축 기울기
        
        return features

class SleepDataset(Dataset):
    """수면 데이터셋 클래스"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]

class SleepStageClassifier(nn.Module):
    """수면 단계 분류 모델
    
    Bidirectional LSTM 기반의 딥러닝 모델로 수면 단계를 분류합니다.
    """
    
    def __init__(self, input_size, hidden_size=64, num_classes=NUM_CLASSES):
        """초기화 함수
        
        Args:
            input_size (int): 입력 특징의 차원
            hidden_size (int): LSTM 히든 레이어 크기
            num_classes (int): 분류할 수면 단계 수
        """
        super(SleepStageClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # bidirectional이므로 hidden_size * 2
        
    def forward(self, x):
        """순전파 함수
        
        Args:
            x (torch.Tensor): 입력 데이터 (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: 수면 단계 예측값
        """
        # LSTM 입력 차원 맞추기: (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
        x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝의 출력만 사용

def load_dreamt_data(data_dir):
    """DREAMT 데이터 로드
    
    Args:
        data_dir (str): DREAMT 데이터 디렉토리 경로
        
    Returns:
        tuple: (features, labels)
    """
    features = []
    labels = []
    
    # CSV 파일들 찾기
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        logging.warning(f"No CSV files found in {data_dir}")
        return np.array([]), np.array([])
    
    # 전처리기 초기화
    preprocessor = SleepDataPreprocessor()
    motion_extractor = MotionFeatureExtractor()
    
    for csv_file in csv_files:
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_file)
            
            # 필요한 컬럼 확인
            required_cols = ['IR', 'RED', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LABEL']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logging.warning(f"Missing columns in {csv_file}: {missing_cols}")
                continue
            
            # 30초 epoch으로 분할
            epoch_length = 30  # seconds
            samples_per_epoch = int(epoch_length * SAMPLING_RATE)
            
            # Epoch 단위로 특징 추출
            n_epochs = len(df) // samples_per_epoch
            
            for i in range(n_epochs):
                try:
                    start_idx = i * samples_per_epoch
                    end_idx = start_idx + samples_per_epoch
                    
                    # 신호 추출
                    ir_signal = df['IR'].iloc[start_idx:end_idx].values
                    red_signal = df['RED'].iloc[start_idx:end_idx].values
                    acc_x = df['ACC_X'].iloc[start_idx:end_idx].values
                    acc_y = df['ACC_Y'].iloc[start_idx:end_idx].values
                    acc_z = df['ACC_Z'].iloc[start_idx:end_idx].values
                    gyro_x = df['GYRO_X'].iloc[start_idx:end_idx].values
                    gyro_y = df['GYRO_Y'].iloc[start_idx:end_idx].values
                    gyro_z = df['GYRO_Z'].iloc[start_idx:end_idx].values
                    
                    # PPG 특징 추출
                    hr, hrv, rr, spo2, quality = preprocessor.extract_ppg_features(ir_signal, red_signal)
                    
                    if quality is not None and quality < QUALITY_THRESHOLD:
                        continue
                    
                    # 움직임 특징 추출
                    motion_features = motion_extractor.extract_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
                    
                    # 12차원 특징 벡터 생성
                    feature_vector = [
                        hr, hrv, rr, spo2,  # PPG 특징 (4개)
                        motion_features['acc_mean'], motion_features['acc_std'], 
                        motion_features['acc_variance'], motion_features['tilt_x'], 
                        motion_features['tilt_y'],  # ACC 특징 (5개)
                        motion_features['gyro_mean'], motion_features['gyro_std'], 
                        motion_features['gyro_variance']  # Gyro 특징 (3개)
                    ]
                    
                    features.append(feature_vector)
                    
                    # 라벨 매핑 (0=W, 1=N1, 2=N2, 3=N3, 4=R)
                    label = df['LABEL'].iloc[end_idx-1]
                    labels.append(label)
                    
                except Exception as e:
                    logging.warning(f"Error processing epoch {i} in {csv_file}: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Error loading {csv_file}: {e}")
            continue
    
    return np.array(features), np.array(labels)

def load_actual_data(data_dir):
    """실제 수집 데이터 로드
    
    Args:
        data_dir (str): 실제 데이터 디렉토리 경로
        
    Returns:
        tuple: (features, labels)
    """
    features = []
    labels = []
    
    # 데이터 파일들 찾기
    data_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not data_files:
        logging.warning(f"No CSV files found in {data_dir}")
        return np.array([]), np.array([])
    
    # 전처리기 초기화
    preprocessor = SleepDataPreprocessor()
    motion_extractor = MotionFeatureExtractor()
    
    for data_file in data_files:
        try:
            # CSV 파일 로드
            df = pd.read_csv(data_file)
            
            # 필요한 컬럼 확인
            required_cols = ['IR', 'RED', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LABEL']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logging.warning(f"Missing columns in {data_file}: {missing_cols}")
                continue
            
            # 30초 epoch으로 분할
            epoch_length = 30  # seconds
            samples_per_epoch = int(epoch_length * SAMPLING_RATE)
            
            # Epoch 단위로 특징 추출
            n_epochs = len(df) // samples_per_epoch
            
            for i in range(n_epochs):
                try:
                    start_idx = i * samples_per_epoch
                    end_idx = start_idx + samples_per_epoch
                    
                    # 신호 추출
                    ir_signal = df['IR'].iloc[start_idx:end_idx].values
                    red_signal = df['RED'].iloc[start_idx:end_idx].values
                    acc_x = df['ACC_X'].iloc[start_idx:end_idx].values
                    acc_y = df['ACC_Y'].iloc[start_idx:end_idx].values
                    acc_z = df['ACC_Z'].iloc[start_idx:end_idx].values
                    gyro_x = df['GYRO_X'].iloc[start_idx:end_idx].values
                    gyro_y = df['GYRO_Y'].iloc[start_idx:end_idx].values
                    gyro_z = df['GYRO_Z'].iloc[start_idx:end_idx].values
                    
                    # PPG 특징 추출
                    hr, hrv, rr, spo2, quality = preprocessor.extract_ppg_features(ir_signal, red_signal)
                    
                    if quality is not None and quality < QUALITY_THRESHOLD:
                        continue
                    
                    # 움직임 특징 추출
                    motion_features = motion_extractor.extract_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
                    
                    # 12차원 특징 벡터 생성
                    feature_vector = [
                        hr, hrv, rr, spo2,  # PPG 특징 (4개)
                        motion_features['acc_mean'], motion_features['acc_std'], 
                        motion_features['acc_variance'], motion_features['tilt_x'], 
                        motion_features['tilt_y'],  # ACC 특징 (5개)
                        motion_features['gyro_mean'], motion_features['gyro_std'], 
                        motion_features['gyro_variance']  # Gyro 특징 (3개)
                    ]
                    
                    features.append(feature_vector)
                    
                    # 라벨 매핑 (0=W, 1=N1, 2=N2, 3=N3, 4=R)
                    label = df['LABEL'].iloc[end_idx-1]
                    labels.append(label)
                    
                except Exception as e:
                    logging.warning(f"Error processing epoch {i} in {data_file}: {e}")
                    continue
                    
        except Exception as e:
            logging.warning(f"Error loading {data_file}: {e}")
            continue
    
    return np.array(features), np.array(labels)

def pretrain_on_dreamt(data_dir, output_path, epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE, lr=LR_PRETRAIN):
    """DREAMT 데이터로 Pre-train
    
    Args:
        data_dir (str): DREAMT 데이터 디렉토리 경로
        output_path (str): 모델 저장 경로
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        lr (float): 학습률
    """
    logging.info("Loading DREAMT data...")
    features, labels = load_dreamt_data(data_dir)
    
    if len(features) == 0:
        logging.error("No valid data found!")
        return
    
    logging.info(f"Loaded {len(features)} samples")
    
    # 데이터 분할 (stratify 사용하여 클래스 균형 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SleepDataset(X_train, y_train)
    test_dataset = SleepDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 초기화
    model = SleepStageClassifier(input_size=12, hidden_size=64, num_classes=NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
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
        
        # 진행 상황 출력
        if (epoch + 1) % 5 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}]')
            logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            logging.info(f'Accuracy: {accuracy:.2f}%')
    
    # 최적의 모델 상태 복원 및 저장
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, output_path)
        logging.info(f"Pre-trained model saved to {output_path}")

def finetune_on_actual(actual_data_dir, pretrained_path, output_path, epochs=EPOCHS_FINETUNE, batch_size=BATCH_SIZE, lr=LR_FINETUNE):
    """실제 데이터로 Fine-tune
    
    Args:
        actual_data_dir (str): 실제 데이터 디렉토리 경로
        pretrained_path (str): Pre-trained 모델 경로
        output_path (str): Fine-tuned 모델 저장 경로
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        lr (float): 학습률
    """
    logging.info("Loading actual data...")
    features, labels = load_actual_data(actual_data_dir)
    
    if len(features) == 0:
        logging.error("No valid data found!")
        return
    
    logging.info(f"Loaded {len(features)} samples")
    
    # 데이터 분할 (stratify 사용하여 클래스 균형 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SleepDataset(X_train, y_train)
    test_dataset = SleepDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 모델 초기화 및 pre-trained 가중치 로드
    model = SleepStageClassifier(input_size=12, hidden_size=64, num_classes=NUM_CLASSES)
    
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        logging.info(f"Loaded pre-trained weights from {pretrained_path}")
    else:
        logging.warning("Pre-trained model not found, starting from scratch")
    
    criterion = nn.CrossEntropyLoss()
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
        
        # 진행 상황 출력
        if (epoch + 1) % 5 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}]')
            logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            logging.info(f'Accuracy: {accuracy:.2f}%')
    
    # 최적의 모델 상태 복원 및 저장
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, output_path)
        logging.info(f"Fine-tuned model saved to {output_path}")

if __name__ == "__main__":
    pretrain_on_dreamt(
        data_dir=r"C:\Users\ahrid\dreamt_pretrain",
        output_path="dreamt_pretrained.pth"
    )
    
    # 2) 실제 데이터로 Fine-tune (주석 처리)
    # finetune_on_actual(
    #     actual_data_dir=ACTUAL_DATA_DIR,
    #     pretrained_path=PRETRAINED_PATH,
    #     output_path=FINETUNED_PATH
    # )
