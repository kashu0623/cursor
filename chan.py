import numpy as np
import pandas as pd
from scipy import signal
import heartpy as hp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SleepDataPreprocessor:
    """수면 데이터 전처리 클래스
    
    PPG 신호(IR/RED)로부터 생체 신호를 추출하고 전처리하는 클래스입니다.
    심박수, HRV, 호흡률, SpO2 등의 생체 지표를 계산합니다.
    """
    
    def __init__(self, sampling_rate=1.0):
        """초기화 함수
        
        Args:
            sampling_rate (float): 데이터 샘플링 주파수 (Hz)
        """
        self.sampling_rate = sampling_rate
        
    def extract_ppg_features(self, ir_signal, red_signal):
        """PPG 신호로부터 생체 지표 추출
        
        Args:
            ir_signal (np.array): IR 센서 신호
            red_signal (np.array): RED 센서 신호
            
        Returns:
            tuple: (심박수, HRV, 호흡률, SpO2)
        """
        # IR 신호만 사용하여 심박수 및 HRV 계산
        filtered_ir = self._filter_signal(ir_signal)
        working_data, measures = hp.process(filtered_ir, self.sampling_rate)
        hr = measures['bpm']  # 심박수 (beats per minute)
        hrv = measures['sdnn']  # 심박변이도 (Standard Deviation of NN intervals)
        
        # 호흡률 추출 (IR 신호 기반)
        rr = self._extract_respiratory_rate(filtered_ir)
        
        # SpO2 계산 (IR/RED 신호 비율 기반)
        spo2 = self._calculate_spo2(ir_signal, red_signal)
        
        return hr, hrv, rr, spo2
    
    def _filter_signal(self, signal):
        """신호 필터링
        
        밴드패스 필터를 사용하여 심박수 대역(0.5-4Hz)의 신호만 추출합니다.
        
        Args:
            signal (np.array): 입력 신호
            
        Returns:
            np.array: 필터링된 신호
        """
        # 밴드패스 필터 적용 (0.5-4Hz: 심박수 대역)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 4.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, signal)
    
    def _extract_respiratory_rate(self, signal):
        """호흡률 추출
        
        FFT를 사용하여 호흡 대역(0.1-0.4Hz)의 주파수를 찾아 호흡률을 계산합니다.
        
        Args:
            signal (np.array): 입력 신호
            
        Returns:
            float: 분당 호흡수
        """
        # FFT 적용
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        
        # 호흡 대역 (0.1-0.4Hz)에서 최대 주파수 찾기
        resp_mask = (freqs >= 0.1) & (freqs <= 0.4)
        resp_freq = freqs[resp_mask][np.argmax(np.abs(fft[resp_mask]))]
        
        return abs(resp_freq) * 60  # 분당 호흡수로 변환
    
    def _calculate_spo2(self, ir_signal, red_signal):
        """SpO2 계산
        
        IR/RED 신호의 AC/DC 비율을 이용하여 산소포화도를 계산합니다.
        
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
        
        # R 값 계산 (SpO2 계산에 사용되는 비율)
        r_value = (red_ac/red_dc) / (ir_ac/ir_dc)
        
        # R 값을 SpO2로 변환 (근사 공식)
        # 실제 SpO2 계산은 더 복잡한 보정이 필요하지만, 여기서는 간단한 근사 사용
        spo2 = 110 - 25 * r_value
        spo2 = np.clip(spo2, 70, 100)  # 70-100% 범위로 제한
        
        return spo2
    
    def _extract_ac_component(self, signal):
        """신호의 AC 성분 추출
        
        고주파 필터를 사용하여 신호의 AC(교류) 성분을 추출합니다.
        
        Args:
            signal (np.array): 입력 신호
            
        Returns:
            np.array: AC 성분
        """
        # 고주파 성분 추출 (AC 성분)
        nyquist = self.sampling_rate / 2
        high = 0.5 / nyquist
        b, a = signal.butter(4, high, btype='high')
        return np.abs(signal.filtfilt(b, a, signal))
    
    def _extract_dc_component(self, signal):
        """신호의 DC 성분 추출
        
        저주파 필터를 사용하여 신호의 DC(직류) 성분을 추출합니다.
        
        Args:
            signal (np.array): 입력 신호
            
        Returns:
            np.array: DC 성분
        """
        # 저주파 성분 추출 (DC 성분)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        b, a = signal.butter(4, low, btype='low')
        return np.abs(signal.filtfilt(b, a, signal))

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

class SleepStageClassifier(nn.Module):
    """수면 단계 분류 모델
    
    LSTM 기반의 딥러닝 모델로 수면 단계를 분류합니다.
    """
    
    def __init__(self, input_size, hidden_size=64, num_classes=4):
        """초기화 함수
        
        Args:
            input_size (int): 입력 특징의 차원
            hidden_size (int): LSTM 히든 레이어 크기
            num_classes (int): 분류할 수면 단계 수
        """
        super(SleepStageClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """순전파 함수
        
        Args:
            x (torch.Tensor): 입력 데이터
            
        Returns:
            torch.Tensor: 수면 단계 예측값
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def generate_dummy_data(n_samples=1000, sequence_length=60):
    """더미 데이터 생성
    
    Args:
        n_samples (int): 생성할 샘플 수
        sequence_length (int): 각 샘플의 시퀀스 길이
        
    Returns:
        pd.DataFrame: 생성된 더미 데이터
    """
    data = {
        'ir_signal': np.random.normal(0, 1, (n_samples, sequence_length)),  # IR 센서 데이터
        'red_signal': np.random.normal(0, 1, (n_samples, sequence_length)),  # RED 센서 데이터
        'acc_x': np.random.normal(0, 0.1, (n_samples, sequence_length)),    # X축 가속도
        'acc_y': np.random.normal(0, 0.1, (n_samples, sequence_length)),    # Y축 가속도
        'acc_z': np.random.normal(0, 0.1, (n_samples, sequence_length)),    # Z축 가속도
        'gyro_x': np.random.normal(0, 0.05, (n_samples, sequence_length)),  # X축 각속도
        'gyro_y': np.random.normal(0, 0.05, (n_samples, sequence_length)),  # Y축 각속도
        'gyro_z': np.random.normal(0, 0.05, (n_samples, sequence_length)),  # Z축 각속도
        'sleep_stage': np.random.randint(0, 4, n_samples)  # 수면 단계 (0: Awake, 1: Light, 2: Deep, 3: REM)
    }
    return pd.DataFrame(data)

def main():
    """메인 실행 함수"""
    # 더미 데이터 생성
    data = generate_dummy_data()
    
    # 전처리 및 특징 추출
    preprocessor = SleepDataPreprocessor()
    motion_extractor = MotionFeatureExtractor()
    
    features = []
    for i in range(len(data)):
        # PPG 특징 추출
        hr, hrv, rr, spo2 = preprocessor.extract_ppg_features(
            data['ir_signal'].iloc[i].values,
            data['red_signal'].iloc[i].values
        )
        
        # 움직임 특징 추출
        motion_features = motion_extractor.extract_features(
            data['acc_x'].iloc[i].values,
            data['acc_y'].iloc[i].values,
            data['acc_z'].iloc[i].values,
            data['gyro_x'].iloc[i].values,
            data['gyro_y'].iloc[i].values,
            data['gyro_z'].iloc[i].values
        )
        
        # 모든 특징 결합 (SpO2 추가)
        feature_vector = [hr, hrv, rr, spo2] + list(motion_features.values())
        features.append(feature_vector)
    
    features = np.array(features)
    labels = data['sleep_stage'].values
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 모델 학습
    model = SleepStageClassifier(
        input_size=features.shape[1],
        hidden_size=64,
        num_classes=4
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 학습 루프
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor.unsqueeze(1))
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
