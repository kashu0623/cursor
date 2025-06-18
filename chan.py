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

# 전역 설정 상수
SAMPLING_RATE = 25.0        # Hz
WINDOW_DURATION = 30        # seconds
STRIDE_DURATION = 10        # seconds
QUALITY_THRESHOLD = 0.5     # 신호 품질 임계값

# 계산된 상수
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)
STRIDE_SIZE = int(SAMPLING_RATE * STRIDE_DURATION)

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
        # 신호 대 잡음비 (SNR) 계산
        ir_snr = np.mean(np.abs(ir_signal)) / np.std(ir_signal)
        red_snr = np.mean(np.abs(red_signal)) / np.std(red_signal)
        
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
    
    def __init__(self, input_size, hidden_size=64, num_classes=4):
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
            x (torch.Tensor): 입력 데이터 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 수면 단계 예측값
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝의 출력만 사용

def generate_dummy_data(n_samples=1000, sequence_length=WINDOW_SIZE, sampling_rate=SAMPLING_RATE):
    """더미 데이터 생성
    
    Args:
        n_samples (int): 생성할 샘플 수
        sequence_length (int): 각 샘플의 시퀀스 길이
        sampling_rate (float): 샘플링 주파수 (Hz)
        
    Returns:
        dict: 생성된 더미 데이터
    """
    # 기본 신호 생성
    t = np.linspace(0, sequence_length/sampling_rate, sequence_length)
    
    # 심박수 변동 (60-100 BPM)
    hr_base = 80
    hr_variation = 20 * np.sin(2 * np.pi * 0.001 * t)  # 천천히 변하는 심박수
    hr = hr_base + hr_variation
    
    # PPG 신호 생성 (심박수 기반)
    ir_signals = []
    red_signals = []
    
    for i in range(n_samples):
        # 기본 PPG 파형
        ir_signal = np.zeros(sequence_length)
        red_signal = np.zeros(sequence_length)
        
        for j in range(sequence_length):
            phase = 2 * np.pi * hr[i] * t[j] / 60  # 심박수에 따른 위상
            ir_signal[j] = np.sin(phase) + 0.2 * np.sin(2 * phase)  # 기본 파형
            red_signal[j] = 0.8 * np.sin(phase) + 0.15 * np.sin(2 * phase)  # RED는 약간 다른 진폭
            
        # 호흡 변동 추가
        breathing = 0.1 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz 호흡
        ir_signal += breathing
        red_signal += 0.8 * breathing
        
        # 움직임 아티팩트 추가 (가끔 발생)
        if random.random() < 0.1:  # 10% 확률로 움직임 발생
            artifact_start = random.randint(0, sequence_length-10)
            artifact_duration = random.randint(5, 10)
            artifact = 0.5 * np.random.randn(artifact_duration)
            ir_signal[artifact_start:artifact_start+artifact_duration] += artifact
            red_signal[artifact_start:artifact_start+artifact_duration] += artifact
        
        # 노이즈 추가
        ir_signal += 0.05 * np.random.randn(sequence_length)
        red_signal += 0.05 * np.random.randn(sequence_length)
        
        ir_signals.append(ir_signal)
        red_signals.append(red_signal)
    
    # 가속도계 데이터 생성 (수면 중 움직임 반영)
    acc_x_signals = []
    acc_y_signals = []
    acc_z_signals = []
    
    for i in range(n_samples):
        # 기본 자세 (약간의 변동)
        acc_x = 0.1 * np.sin(2 * np.pi * 0.01 * t) + 0.05 * np.random.randn(sequence_length)
        acc_y = 0.1 * np.cos(2 * np.pi * 0.01 * t) + 0.05 * np.random.randn(sequence_length)
        acc_z = 1.0 + 0.1 * np.sin(2 * np.pi * 0.005 * t) + 0.05 * np.random.randn(sequence_length)
        
        # 가끔 큰 움직임 추가
        if random.random() < 0.05:  # 5% 확률로 큰 움직임
            move_start = random.randint(0, sequence_length-20)
            move_duration = random.randint(10, 20)
            acc_x[move_start:move_start+move_duration] += 0.5 * np.random.randn(move_duration)
            acc_y[move_start:move_start+move_duration] += 0.5 * np.random.randn(move_duration)
            acc_z[move_start:move_start+move_duration] += 0.5 * np.random.randn(move_duration)
        
        acc_x_signals.append(acc_x)
        acc_y_signals.append(acc_y)
        acc_z_signals.append(acc_z)
    
    # 자이로스코프 데이터 생성
    gyro_x_signals = []
    gyro_y_signals = []
    gyro_z_signals = []
    
    for i in range(n_samples):
        gyro_x = 0.1 * np.random.randn(sequence_length)
        gyro_y = 0.1 * np.random.randn(sequence_length)
        gyro_z = 0.1 * np.random.randn(sequence_length)
        
        gyro_x_signals.append(gyro_x)
        gyro_y_signals.append(gyro_y)
        gyro_z_signals.append(gyro_z)
    
    # 수면 단계 생성 (더 현실적인 패턴)
    sleep_stages = []
    current_stage = 0
    stage_duration = 0
    
    for i in range(n_samples):
        if stage_duration <= 0:
            # 다음 단계로 전환
            if current_stage == 0:  # Awake
                current_stage = 1  # Light sleep
            elif current_stage == 1:  # Light sleep
                if random.random() < 0.3:  # 30% 확률로 Deep sleep
                    current_stage = 2
                else:
                    current_stage = 3  # REM sleep
            elif current_stage == 2:  # Deep sleep
                current_stage = 1  # Light sleep
            else:  # REM sleep
                if random.random() < 0.2:  # 20% 확률로 Awake
                    current_stage = 0
                else:
                    current_stage = 1  # Light sleep
            
            # 단계 지속 시간 설정 (5-15분)
            stage_duration = random.randint(5, 15) * 60 * sampling_rate
        
        sleep_stages.append(current_stage)
        stage_duration -= 1
    
    data = {
        'ir_signals': ir_signals,
        'red_signals': red_signals,
        'acc_x_signals': acc_x_signals,
        'acc_y_signals': acc_y_signals,
        'acc_z_signals': acc_z_signals,
        'gyro_x_signals': gyro_x_signals,
        'gyro_y_signals': gyro_y_signals,
        'gyro_z_signals': gyro_z_signals,
        'sleep_stages': sleep_stages
    }
    return data

def main():
    """메인 실행 함수"""
    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 10  # Early stopping patience
    
    # 더미 데이터 생성
    data = generate_dummy_data(sampling_rate=SAMPLING_RATE)
    
    # 전처리 및 특징 추출
    preprocessor = SleepDataPreprocessor(sampling_rate=SAMPLING_RATE)
    motion_extractor = MotionFeatureExtractor()
    
    features = []
    valid_indices = []
    
    for i in range(len(data['ir_signals'])):
        # PPG 특징 추출
        hr, hrv, rr, spo2, quality = preprocessor.extract_ppg_features(
            data['ir_signals'][i],
            data['red_signals'][i]
        )
        
        # 신호 품질이 낮은 경우 건너뛰기
        if quality is not None and quality < QUALITY_THRESHOLD:
            continue
            
        # 움직임 특징 추출
        motion_features = motion_extractor.extract_features(
            data['acc_x_signals'][i],
            data['acc_y_signals'][i],
            data['acc_z_signals'][i],
            data['gyro_x_signals'][i],
            data['gyro_y_signals'][i],
            data['gyro_z_signals'][i]
        )
        
        # 모든 특징 결합
        feature_vector = [hr, hrv, rr, spo2] + list(motion_features.values())
        features.append(feature_vector)
        valid_indices.append(i)
    
    features = np.array(features)
    labels = [data['sleep_stages'][i] for i in valid_indices]
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SleepDataset(X_train, y_train)
    test_dataset = SleepDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 모델 초기화
    model = SleepStageClassifier(
        input_size=features.shape[1],
        hidden_size=64,
        num_classes=4
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Early stopping 변수
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 학습 루프
    for epoch in range(EPOCHS):
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
            
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        # 진행 상황 출력
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}]')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Accuracy: {accuracy:.2f}%')
    
    # 최적의 모델 상태 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print('학습 완료!')

if __name__ == "__main__":
    main()
