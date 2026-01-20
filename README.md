# 멜 스펙트로그램 실습 튜토리얼

> 블로그 "멜 스펙트로그램: 소리를 이미지로 바꾸는 방법"의 실습 코드

오디오 딥러닝의 핵심 전처리인 멜 스펙트로그램을 직접 만들어보는 실습 코드입니다. 블로그의 이론을 6개 모듈로 나누어, 파형 분석부터 CNN 입력 후처리까지 단계별로 실행하며 이해할 수 있습니다.

## 목적

블로그의 이론을 **직접 실행해보며** 이해할 수 있는 코드를 제공합니다.

- 각 개념을 단계별로 실행
- 파라미터를 바꿔보며 차이 확인

## 시작하기 전에

### 필요한 것

- **Python 3.10 이상** - 아직 설치하지 않았다면 [python.org](https://www.python.org/downloads/)에서 다운로드
- **터미널(명령 프롬프트)** - 명령어를 입력할 수 있는 프로그램
  - macOS: `터미널` 앱 (Spotlight에서 "터미널" 검색)
  - Windows: `PowerShell` 또는 `명령 프롬프트`
  - Linux: 기본 터미널

### Python 버전 확인하기

터미널을 열고 다음 명령어를 입력하세요:

```bash
python --version
```

`Python 3.10.x` 이상이 나오면 준비 완료입니다.

> **참고**: macOS/Linux에서는 `python3 --version`으로 확인해야 할 수 있습니다.

## 빠른 시작

### 1단계: 코드 다운로드

```bash
# 프로젝트 다운로드
git clone https://github.com/YOUR_USERNAME/mel-spectrogram-tutorial.git

# 다운로드한 폴더로 이동
cd mel-spectrogram-tutorial
```

> **git이 없다면?** [GitHub에서 ZIP 다운로드](https://github.com/YOUR_USERNAME/mel-spectrogram-tutorial/archive/refs/heads/main.zip) 후 압축을 풀어주세요.

### 2단계: uv 설치 (패키지 관리자)

uv는 Python 패키지를 빠르게 설치해주는 도구입니다.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 터미널을 **새로 열어주세요** (환경변수 적용을 위해).

### 3단계: 의존성 설치

```bash
# 필요한 라이브러리 자동 설치
uv sync
```

이 명령어 하나로 librosa, numpy, matplotlib 등 필요한 모든 라이브러리가 설치됩니다.

### 4단계: 가상환경 활성화

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (명령 프롬프트):**

```cmd
.venv\Scripts\activate.bat
```

> 활성화되면 터미널 앞에 `(.venv)`가 표시됩니다.

### 5단계: 실습 실행

```bash
# 실습 1부터 순서대로 실행
python 01_waveform_and_frequency.py
python 02_stft_and_spectrogram.py
python 03_mel_scale.py
python 04_mel_spectrogram_pipeline.py
python 05_cnn_preprocessing.py
python 06_parameter_experiments.py
```

### 6단계: 결과 확인

```bash
# 생성된 이미지 파일 목록 확인
ls outputs/
```

`outputs/` 폴더에 `.png` 이미지 파일들이 생성됩니다. 이미지 뷰어로 열어서 확인하세요.

## 문제 해결

### "python: command not found" 오류

- `python3` 명령어를 대신 사용해보세요
- Python이 설치되어 있는지 확인하세요

### "uv: command not found" 오류

- 터미널을 닫고 새로 열어보세요
- uv 설치 명령어를 다시 실행해보세요

### "ModuleNotFoundError" 오류

가상환경이 활성화되지 않았을 수 있습니다:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### librosa 관련 오류

macOS에서 `libsndfile` 오류가 나면:

```bash
brew install libsndfile
```

## 실습 구성

| 모듈 | 블로그 섹션 | 학습 내용 |
|------|------------|----------|
| `01_waveform_and_frequency.py` | 1. 배경 | 파형, 주파수, FFT |
| `02_stft_and_spectrogram.py` | 2. STFT | 윈도우, 해상도 트레이드오프 |
| `03_mel_scale.py` | 3. 멜 스케일 | HTK 공식, 필터 뱅크 |
| `04_mel_spectrogram_pipeline.py` | 4. 생성 | 전체 파이프라인, dB 변환 |
| `05_cnn_preprocessing.py` | 5. 후처리 | 패딩, 정규화, 채널 |
| `06_parameter_experiments.py` | 6. 파라미터 | n_mels, htk, fmin/fmax, window 비교 |

## 파라미터 설정

모든 파라미터는 `config.yaml`에서 관리합니다 (블로그와 동일):

```yaml
# STFT
n_fft: 2048          # FFT 크기 - 주파수 해상도 결정
hop_length: 512      # 홉 길이 - 시간 해상도 결정
window: "hann"       # 윈도우 함수

# 멜 필터 뱅크
n_mels: 128          # 멜 빈 수
fmin: 0              # 최소 주파수 (Hz)
fmax: null           # 최대 주파수 (null = sr/2)
htk: true            # HTK 멜 스케일 (librosa 기본값은 false)

# dB 변환
ref: "max"           # 상대 스케일 - 최대값이 0dB
top_db: 80           # 동적 범위 제한
power: 2.0           # 파워 스펙트럼 (진폭² 기준)
```

### 기본 파라미터 vs 실험 파라미터

`config.yaml`은 두 가지 역할을 합니다:

| 섹션 | 적용 범위 | 용도 |
|-----|----------|-----|
| `audio`, `stft`, `mel`, `db` | 모든 모듈 (01~06) | 기본 파라미터 설정 |
| `comparison` | 06_parameter_experiments.py만 | 비교 실험용 값 |

**기본 파라미터 변경** - 모든 모듈에 적용됩니다:

```yaml
stft:
  n_fft: 4096        # 2048 -> 4096으로 변경
  hop_length: 1024   # 512 -> 1024로 변경
```

**실험 파라미터 변경** - 06번 모듈의 비교 실험에만 적용됩니다:

```yaml
comparison:
  n_mels_values: [32, 64, 80]           # 커스텀 n_mels 비교
  fmin_fmax_values:
    - [0, null]                          # 전체 범위
    - [50, 6000]                         # 커스텀 범위
```

## 파라미터 실험 가이드

`06_parameter_experiments.py`에서 6가지 파라미터를 비교 실험할 수 있습니다:

### 1. n_mels - 주파수 축 해상도

```yaml
comparison:
  n_mels_values: [64, 128, 256]
```

- 64: 낮은 해상도, 빠른 처리, 메모리 절약
- 128: librosa 기본값, 일반적인 선택
- 256: 높은 해상도, 세밀한 주파수 표현

### 2. n_fft / hop_length - 시간-주파수 해상도 트레이드오프

```yaml
comparison:
  n_fft_values: [512, 2048, 4096]
  hop_length_values: [256, 512, 1024]  # n_fft와 쌍으로 사용
```

- n_fft 클수록: 주파수 해상도 높아지고, 시간 해상도 낮아짐
- hop_length는 보통 n_fft의 1/4 (75% 오버랩)

### 3. htk - 멜 스케일 계산 방식

고정 비교 (True vs False):

- HTK: 전통적 음성 인식 방식, 전체 범위 로그 스케일
- Slaney: librosa 기본값, 1000Hz 기준 선형/로그 혼합

### 4. ref - dB 변환 기준값

고정 비교 (np.max vs 1.0):

- np.max: 상대적 스케일, 시각화 용도
- 1.0: 절대적 스케일, 파일 간 비교 용도

### 5. fmin/fmax - 주파수 범위 제한

```yaml
comparison:
  fmin_fmax_values:
    - [0, null]       # 전체 범위 (0 ~ sr/2)
    - [20, 8000]      # 음성 처리 권장
    - [100, 4000]     # 좁은 범위
```

- 음성 처리: [20, 8000] 권장 (저주파 노이즈 제거)
- 음악 분석: [0, null] 전체 범위

### 6. window - 윈도우 함수

```yaml
comparison:
  window_values: ["hann", "hamming", "blackman"]
```

- hann: 가장 일반적, 좋은 균형 (기본값)
- hamming: 음성 처리 전통적 선택
- blackman: 사이드로브 억제 우수

### 커스텀 실험 예시

음성 처리용 설정으로 실험하기:

```yaml
comparison:
  n_mels_values: [40, 80, 128]          # 음성용 멜 빈 수
  fmin_fmax_values:
    - [20, 8000]                         # 음성 주파수 범위
    - [50, 7000]
  window_values: ["hann", "hamming"]
```

실행:

```bash
python 06_parameter_experiments.py
ls outputs/06_*.png  # 결과 확인
```

## 블로그 주의사항 확인

이 실습에서 직접 확인할 수 있는 주의사항:

| 주의사항 | 확인 모듈 |
|---------|----------|
| HTK vs Slaney 멜 스케일 차이 | `03`, `06` |
| 시간-주파수 해상도 트레이드오프 | `02`, `06` |
| fmin/fmax 주파수 범위 설정 | `06` |
| 윈도우 함수 선택 | `02`, `06` |
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
├── pyproject.toml                  # 의존성
├── outputs/                        # 실행 결과
└── assets/                         # 샘플 오디오 (선택)
```

## 의존성

- Python >= 3.10
- librosa >= 0.10.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0

## 라이선스

MIT License
