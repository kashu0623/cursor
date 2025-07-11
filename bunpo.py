import pandas as pd
import numpy as np
import glob

# CSV 파일 리스트 (glob으로 찾기)
all_csv_files = glob.glob(r'C:/dreamt_pretrain/S*.csv')
print(f'찾은 CSV 파일 수: {len(all_csv_files)}')

# 찾은 파일 목록 출력
for file in all_csv_files:
    print(f' - {file}')

all_labels = []

for file in all_csv_files:
    try:
        df = pd.read_csv(file)
        labels = df['Sleep_Stage'].values  # 'Sleep_Stage' 컬럼 이름 맞는지 꼭 확인!

        # 라벨 매핑: P → W, N3 → N2
        labels = ['W' if l == 'P' else l for l in labels]
        labels = ['N2' if l == 'N3' else l for l in labels]

        all_labels.extend(labels)
    except Exception as e:
        print(f'파일 읽기 실패: {file}, 에러: {e}')
        continue

all_labels = np.array(all_labels)

# 고유 라벨 및 개수 집계
unique, counts = np.unique(all_labels, return_counts=True)

print("\n=== 클래스 분포 (P→W, N3→N2 후) ===")
total = np.sum(counts)
for u, c in zip(unique, counts):
    percent = c / total * 100
    print(f"{u}: {c}개 ({percent:.2f}%)")
