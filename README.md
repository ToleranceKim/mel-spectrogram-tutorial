# 멜 스펙트로그램 실습 튜토리얼

> 블로그 "멜 스펙트로그램: 소리를 이미지로 바꾸는 방법"의 실습 코드

오디오 딥러닝의 핵심 전처리인 멜 스펙트로그램을 직접 만들어보는 실습 코드입니다. 블로그의 이론을 6개 모듈로 나누어, 파형 분석부터 CNN 입력 후처리까지 단계별로 실행하며 이해할 수 있습니다.

## 목적

블로그의 이론을 **직접 실행해보며** 이해할 수 있는 코드를 제공합니다.

- 각 개념을 단계별로 실행
- 파라미터를 바꿔보며 차이 확인

## 빠른 시작

```bash
git clone https://github.com/YOUR_USERNAME/mel-spectrogram-tutorial.git
cd mel-spectrogram-tutorial
pip install -r requirements.txt

# 순서대로 실습
python 01_waveform_and_frequency.py
python 02_stft_and_spectrogram.py
python 03_mel_scale.py
python 04_mel_spectrogram_pipeline.py
python 05_cnn_preprocessing.py
python 06_parameter_experiments.py

# 결과 확인
ls outputs/
```

## 실습 구성

| 모듈 | 블로그 섹션 | 학습 내용 |
|------|------------|----------|
| `01_waveform_and_frequency.py` | 1. 배경 | 파형, 주파수, FFT |
| `02_stft_and_spectrogram.py` | 2. STFT | 윈도우, 해상도 트레이드오프 |
| `03_mel_scale.py` | 3. 멜 스케일 | HTK 공식, 필터 뱅크 |
| `04_mel_spectrogram_pipeline.py` | 4. 생성 | 전체 파이프라인, dB 변환 |
| `05_cnn_preprocessing.py` | 5. 후처리 | 패딩, 정규화, 채널 |
| `06_parameter_experiments.py` | 6. 파라미터 | n_mels, htk 비교 |

## 파라미터 설정

모든 파라미터는 `config.yaml`에서 관리합니다 (블로그와 동일):

```yaml
# STFT
n_fft: 2048          # FFT 크기 — 주파수 해상도 결정
hop_length: 512      # 홉 길이 — 시간 해상도 결정
window: "hann"       # 윈도우 함수

# 멜 필터 뱅크
n_mels: 128          # 멜 빈 수
htk: true            # HTK 멜 스케일 (librosa 기본값은 false)

# dB 변환
ref: "max"           # 상대 스케일 — 최대값이 0dB
top_db: 80           # 동적 범위 제한
power: 2.0           # 파워 스펙트럼 (진폭² 기준)
```

## 블로그 주의사항 확인

이 실습에서 직접 확인할 수 있는 주의사항:

| 주의사항 | 확인 모듈 |
|---------|----------|
| HTK vs Slaney 멜 스케일 차이 | `03`, `06` |
| 시간-주파수 해상도 트레이드오프 | `02`, `06` |
| dB 도메인 0 패딩 금지 | `05` |
| center=True 정렬 이슈 | `05` |
| 정규화와 패딩 상호작용 | `05` |

## 폴더 구조

```
mel-spectrogram-tutorial/
├── 01_waveform_and_frequency.py    # 실습 1: 파형과 주파수
├── 02_stft_and_spectrogram.py      # 실습 2: STFT
├── 03_mel_scale.py                 # 실습 3: 멜 스케일
├── 04_mel_spectrogram_pipeline.py  # 실습 4: 전체 파이프라인
├── 05_cnn_preprocessing.py         # 실습 5: CNN 후처리
├── 06_parameter_experiments.py     # 실습 6: 파라미터 실험
├── utils.py                        # 공통 유틸리티
├── config.yaml                     # 파라미터 설정
├── requirements.txt                # 의존성
├── outputs/                        # 실행 결과
└── assets/                         # 샘플 오디오 (선택)
```

## 의존성

- Python 3.8+
- librosa >= 0.10.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- pyyaml >= 6.0

## 라이선스

MIT License
