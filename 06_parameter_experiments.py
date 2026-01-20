"""
실습 6: 파라미터 실험

블로그 섹션: 6. 코드 예시, 파라미터 가이드

학습 목표:
- 각 파라미터가 결과에 미치는 영향 체감
- 자신의 태스크에 맞는 파라미터 선택 감각 기르기

실습 내용:
1. n_mels 비교 - 주파수 축 해상도
2. n_fft / hop_length 비교 - 시간-주파수 해상도 트레이드오프
3. htk 비교 - 멜 스케일 계산 방식 (HTK vs Slaney)
4. ref 비교 - dB 변환 기준값 (np.max vs 1.0)
5. fmin/fmax 비교 - 주파수 범위 제한
6. window 비교 - 윈도우 함수 (hann vs hamming vs blackman)

파라미터 커스터마이징:
- config.yaml의 comparison 섹션에서 실험값 수정 가능
- 예: n_mels_values: [32, 64, 80] 으로 변경하면 해당 값들로 비교

출력:
- outputs/06_n_mels_comparison.png
- outputs/06_resolution_comparison.png
- outputs/06_htk_comparison.png
- outputs/06_ref_comparison.png
- outputs/06_fmin_fmax_comparison.png
- outputs/06_window_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import load_config, ensure_output_dir

def compare_n_mels(signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels_values: list[int] = None,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    fmin: float = 0,
    fmax: float = None,
    htk: bool = True,
    norm: str = "slaney",
    power: float = 2.0,
    amin: float = 1e-10,
    top_db: float = 80.0
    ) -> None:
    """
    n_mels 파라미터 비교 (실습 1)

    n_mels란?
    - 멜 필터뱅크의 필터 개수
    - 멜 스펙트로그램의 주파수 축 해상도 결정
    - 값이 클수록 주파수 방향 해상도 증가, 계산량도 증가

    비교 값:
    - 64: 낮은 해상도, 빠른 처리
    - 128: librosa 기본값, 일반적인 선택
    - 256: 높은 해상도, 세밀한 주파수 표현

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트 (Hz)
        n_fft: FFT 크기
        hop_length: 홉 길이
        n_mels_values: 비교할 n_mels 값 리스트 (기본값: [64, 128, 256])
    """
    if n_mels_values is None:
        n_mels_values = [64, 128, 256]

    print(f"\n[1] n_mels 비교 ({' vs '.join(map(str, n_mels_values))})")

    n_plots = len(n_mels_values)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, n_mels in enumerate(n_mels_values):
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)

        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[i], cmap='magma'
        )
        axes[i].set_title(f'n_mels = {n_mels}, shape = {mel_db.shape}')
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')

        print(f"    n_mels={n_mels}: shape = {mel_db.shape}")

    plt.tight_layout()
    plt.savefig('outputs/06_n_mels_comparison.png', dpi=150)
    plt.close()

    print(f"    저장: outputs/06_n_mels_comparison.png")

def compare_resolution(signal: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft_values: list[int] = None,
    hop_length_values: list[int] = None,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    fmin: float = 0,
    fmax: float = None,
    htk: bool = True,
    norm: str = "slaney",
    power: float = 2.0,
    amin: float = 1e-10,
    top_db: float = 80.0
    ) -> None:
    """
    n_fft / hop_length 비교 (실습 2)

    시간-주파수 해상도 트레이드오프:
    - 큰 n_fft (예: 4096): 주파수 해상도 높음, 시간 해상도 낮음
    - 작은 n_fft (예: 512): 주파수 해상도 낮음, 시간 해상도 높음

    비교 설정:
    - (512, 128): 시간 해상도 우선 (빠른 변화 포착)
    - (2048, 512): 균형 (librosa 기본값)
    - (4096, 1024): 주파수 해상도 우선 (세밀한 주파수)

    hop_length는 보통 n_fft의 1/4로 설정 (75% 오버랩)

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트 (Hz)
        n_mels: 멜 필터 개수
        n_fft_values: 비교할 n_fft 값 리스트 (기본값: [512, 2048, 4096])
        hop_length_values: 비교할 hop_length 값 리스트 (기본값: [128, 512, 1024])
    """
    print("\n[2] n_fft / hop_length 비교 (해상도 트레이드오프)")

    # 기본값 설정
    if n_fft_values is None:
        n_fft_values = [512, 2048, 4096]
    if hop_length_values is None:
        hop_length_values = [128, 512, 1024]

    # (n_fft, hop_length) 조합 생성
    settings = list(zip(n_fft_values, hop_length_values))

    n_plots = len(settings)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, (n_fft, hop_length) in enumerate(settings):
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)

        # 해상도 계산
        freq_res = sr / n_fft
        time_res = hop_length / sr * 1000 # ms

        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[i], cmap='magma'
        )
        axes[i].set_title(
            f'n_fft={n_fft}, hop={hop_length} | '
            f'Freq: {freq_res:.1f}Hz, Time: {time_res:.1f}ms | '
            f"shape={mel_db.shape}"
        )
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')

        print(f"    n_fft={n_fft}, hop={hop_length}: "
            f"freq={freq_res:.1f}Hz, time={time_res:.1f}ms, shape={mel_db.shape}")
        
    plt.tight_layout()
    plt.savefig('outputs/06_resolution_comparison.png', dpi=150)
    plt.close()

    print(f"    저장: outputs/06_resolution_comparison.png")

def compare_htk(signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    fmin: float = 0,
    fmax: float = None,
    norm: str = "slaney",
    power: float = 2.0,
    amin: float = 1e-10,
    top_db: float = 80.0
    ) -> None:
    """
    htk 파라미터 비교 (실습 3)

    htk 파라미터란?
    - 멜 스케일 계산 방식 선택
    - htk=True: HTK 공식 (mel = 2595 * log10(1 + hz/700))
    - htk=False: Slaney 공식 (librosa 기본값)

    두 방식의 차이:
    - HTK: 전체 주파수 범위에서 로그 스케일
    - Slaney: 1000Hz 이하 선형, 이상 로그 스케일

    실무에서:
    - 대부분 큰 차이 없음
    - 기본 모델 호환성이 주용하면 해당 설정 따르기

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트 (Hz)
        n_fft: FFT 크기
        hop_length: 홉 길이
        n_mels: 멜 필터 개수
    """
    print("\n[3] htk 파라미터 비교 (HTK vs Slaney)")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for i, htk in enumerate([True, False]):
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)

        label = "HTK" if htk else "Slaney (default)"

        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[i], cmap='magma'
        )
        axes[i].set_title(f'htk={htk} ({label})')
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')

        print(f"    htk={htk} ({label}): shape={mel_db.shape}")

    plt.tight_layout()
    plt.savefig('outputs/06_htk_comparison.png', dpi=150)
    plt.close()

    print(f"    저장: outputs/06_htk_comparison.png")

def compare_ref(signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    fmin: float = 0,
    fmax: float = None,
    htk: bool = True,
    norm: str = "slaney",
    power: float = 2.0,
    amin: float = 1e-10,
    top_db: float = 80.0
    ) -> None:
    """
    ref 파라미터 비교 (실습 4)

    ref 파라미터란?
    - power_to_db 변환 시 기준값 설정
    - dB = 10 * log10(S / ref)

    비교 값:
    - ref=np.max: 최대값 기준 (상대적), 최대값이 0dB
    - ref=1.0: 절대 기준, 실제 파워 값 반영

    사용 가이드:
    - np.max: 단일 파일 분석, 시각화 용도
    - 1.0: 여러 파일 비교, 절대적 크기 비교 필요 시

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트 (Hz)
        n_fft: FFT 크기
        hop_length: 홉 길이
        n_mels: 멜 필터 개수
    """
    print("\n[4] ref 파라미터 비교 (np.max vs 1.0)")

    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, center=center, pad_mode=pad_mode,
        power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
    )

    # 두 가지 ref 설정
    mel_db_max = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)
    mel_db_one = librosa.power_to_db(mel_spec, ref=1.0, amin=amin, top_db=top_db)

    print(f"    ref=np.max: 범위 [{mel_db_max.min():.1f}, {mel_db_max.max():.1f}] dB")
    print(f"    ref=1.0:    범위 [{mel_db_one.min():.1f}, {mel_db_one.max():.1f}] dB")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ref=np.max
    img1 = librosa.display.specshow(
        mel_db_max, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', ax=axes[0], cmap='magma'
    )
    axes[0].set_title(f'ref=np.max (relative): range [{mel_db_max.min():.1f}, {mel_db_max.max():.1f}] dB')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # ref=1.0
    img2 = librosa.display.specshow(
        mel_db_one, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', ax=axes[1], cmap='magma'
    )
    axes[1].set_title(f'ref=1.0 (absolute): range [{mel_db_one.min():.1f}, {mel_db_one.max():.1f}] dB')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig('outputs/06_ref_comparison.png', dpi=150)
    plt.close()

    print(f"    저장: outputs/06_ref_comparison.png")

def compare_fmin_fmax(signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin_fmax_values: list = None,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    htk: bool = True,
    norm: str = "slaney",
    power: float = 2.0,
    amin: float = 1e-10,
    top_db: float = 80.0
    ) -> None:
    """
    fmin/fmax 파라미터 비교 (실습 5)

    fmin/fmax란?
    - 멜 필터뱅크가 커버하는 주파수 범위 설정
    - fmin: 최소 주파수 (기본값 0Hz)
    - fmax: 최대 주파수 (기본값 sr/2, 나이퀴스트 주파수)

    사용 가이드:
    - 전체 범위 [0, sr/2]: 일반적인 음악/음향 분석
    - 음성 처리 [20, 8000]: 저주파 노이즈 제거, 불필요한 고주파 제외
    - 좁은 범위 [100, 4000]: 핵심 주파수 대역만 집중

    주의사항:
    - fmax가 sr/2보다 크면 자동으로 sr/2로 제한됨
    - 범위를 좁히면 해당 대역의 해상도가 상대적으로 높아짐

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트 (Hz)
        n_fft: FFT 크기
        hop_length: 홉 길이
        n_mels: 멜 필터 개수
        fmin_fmax_values: [fmin, fmax] 쌍의 리스트 (기본값: [[0, None], [20, 8000], [100, 4000]])
    """
    if fmin_fmax_values is None:
        fmin_fmax_values = [[0, None], [20, 8000], [100, 4000]]

    print(f"\n[5] fmin/fmax 비교 (주파수 범위 제한)")

    n_plots = len(fmin_fmax_values)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, (fmin, fmax) in enumerate(fmin_fmax_values):
        # fmax가 None이면 sr/2 사용
        actual_fmax = fmax if fmax is not None else sr // 2

        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)

        fmax_label = f"{fmax}" if fmax is not None else f"sr/2 ({sr//2})"
        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[i], cmap='magma'
        )
        axes[i].set_title(f'fmin={fmin}, fmax={fmax_label} | shape={mel_db.shape}')
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')

        print(f"    fmin={fmin}, fmax={fmax_label}: shape={mel_db.shape}")

    plt.tight_layout()
    plt.savefig('outputs/06_fmin_fmax_comparison.png', dpi=150)
    plt.close()

    print(f"    저장: outputs/06_fmin_fmax_comparison.png")

def compare_window(signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    window_values: list[str] = None,
    win_length: int = None,
    center: bool = True,
    pad_mode: str = "constant",
    fmin: float = 0,
    fmax: float = None,
    htk: bool = True,
    norm: str = "slaney",
    power: float = 2.0,
    amin: float = 1e-10,
    top_db: float = 80.0
    ) -> None:
    """
    window 파라미터 비교 (실습 6)

    윈도우 함수란?
    - STFT에서 각 프레임에 적용하는 가중치 함수
    - 스펙트럼 누설(spectral leakage)을 줄이기 위해 사용
    - 프레임 경계에서 신호를 부드럽게 감쇠시킴

    비교 윈도우:
    - hann: 가장 일반적, 좋은 주파수 해상도와 사이드로브 억제의 균형
    - hamming: 음성 처리에서 전통적으로 사용, 첫 샘플이 0이 아님
    - blackman: 사이드로브 억제 우수하지만 메인로브가 넓음

    실무에서:
    - 대부분 hann 사용 (librosa 기본값)
    - 특별한 이유가 없으면 기본값 권장

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트 (Hz)
        n_fft: FFT 크기
        hop_length: 홉 길이
        n_mels: 멜 필터 개수
        window_values: 비교할 윈도우 함수 리스트 (기본값: ["hann", "hamming", "blackman"])
    """
    if window_values is None:
        window_values = ["hann", "hamming", "blackman"]

    print(f"\n[6] window 비교 ({' vs '.join(window_values)})")

    n_plots = len(window_values)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, window in enumerate(window_values):
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, center=center, pad_mode=pad_mode,
            power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)

        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[i], cmap='magma'
        )
        default_mark = " (default)" if window == "hann" else ""
        axes[i].set_title(f'window="{window}"{default_mark} | shape={mel_db.shape}')
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')

        print(f"    window={window}: shape={mel_db.shape}")

    plt.tight_layout()
    plt.savefig('outputs/06_window_comparison.png', dpi=150)
    plt.close()

    print(f"    저장: outputs/06_window_comparison.png")

def main():
    """
    메인 실행 함수 - 각 실습을 순차 실행

    config.yaml의 comparison 섹션에서 실험 파라미터를 커스터마이징할 수 있습니다.

    기본 파라미터 변경:
        config.yaml의 stft, mel 섹션 수정 -> 모든 모듈(01~06)에 적용

    실험 파라미터 변경:
        config.yaml의 comparison 섹션 수정 -> 06_parameter_experiments.py에만 적용

    예시:
        # config.yaml
        comparison:
          n_mels_values: [32, 64, 80]  # 커스텀 n_mels 실험
          window_values: ["hann", "hamming"]  # 윈도우 2개만 비교
    """
    config = load_config()
    sr = config['audio']['sr']
    n_fft = config['stft']['n_fft']
    hop_length = config['stft']['hop_length']
    win_length = config['stft']['win_length']
    window = config['stft']['window']
    center = config['stft']['center']
    pad_mode = config['stft']['pad_mode']
    n_mels = config['mel']['n_mels']
    fmin = config['mel']['fmin']
    fmax = config['mel']['fmax']
    htk = config['mel']['htk']
    norm = config['mel']['norm']
    amin = config['db']['amin']
    top_db = config['db']['top_db']
    power = config['db']['power']

    # =========================================================================
    # comparison 설정 로드 (없으면 각 함수의 기본값 사용)
    # config.yaml에서 값을 수정하면 해당 값으로 실험 가능
    # =========================================================================
    comparison = config.get('comparison', {})
    n_mels_values = comparison.get('n_mels_values', None)
    n_fft_values = comparison.get('n_fft_values', None)
    hop_length_values = comparison.get('hop_length_values', None)
    fmin_fmax_values = comparison.get('fmin_fmax_values', None)
    window_values = comparison.get('window_values', None)

    ensure_output_dir()

    print("=" * 60)
    print("실습 6: 파라미터 실험")
    print("=" * 60)
    print("\n[설정 정보]")
    print(f"    기본 파라미터: sr={sr}, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}")
    print(f"    실험 파라미터: config.yaml의 comparison 섹션에서 로드")

    # 트럼펫 샘플 로드
    trumpet, _ = librosa.load(librosa.ex('trumpet'), sr=sr)

    # 실습 1: n_mels 비교 (주파수 축 해상도)
    compare_n_mels(trumpet, sr, n_fft, hop_length, n_mels_values,
                   win_length, window, center, pad_mode,
                   fmin, fmax, htk, norm, power, amin, top_db)

    # 실습 2: 해상도 트레이드오프 (시간-주파수)
    compare_resolution(trumpet, sr, n_mels, n_fft_values, hop_length_values,
                       win_length, window, center, pad_mode,
                       fmin, fmax, htk, norm, power, amin, top_db)

    # 실습 3: HTK vs Slaney (멜 스케일 계산 방식)
    compare_htk(trumpet, sr, n_fft, hop_length, n_mels,
                win_length, window, center, pad_mode,
                fmin, fmax, norm, power, amin, top_db)

    # 실습 4: ref 파라미터 비교 (dB 기준값)
    compare_ref(trumpet, sr, n_fft, hop_length, n_mels,
                win_length, window, center, pad_mode,
                fmin, fmax, htk, norm, power, amin, top_db)

    # 실습 5: fmin/fmax 비교 (주파수 범위 제한)
    compare_fmin_fmax(trumpet, sr, n_fft, hop_length, n_mels, fmin_fmax_values,
                      win_length, window, center, pad_mode,
                      htk, norm, power, amin, top_db)

    # 실습 6: window 비교 (윈도우 함수)
    compare_window(trumpet, sr, n_fft, hop_length, n_mels, window_values,
                   win_length, center, pad_mode,
                   fmin, fmax, htk, norm, power, amin, top_db)

    print("\n" + "=" * 60)
    print("실습 6 완료")
    print("=" * 60)
    print("\n[출력 파일]")
    print("    - outputs/06_n_mels_comparison.png")
    print("    - outputs/06_resolution_comparison.png")
    print("    - outputs/06_htk_comparison.png")
    print("    - outputs/06_ref_comparison.png")
    print("    - outputs/06_fmin_fmax_comparison.png")
    print("    - outputs/06_window_comparison.png")

if __name__ == "__main__":
    main()

    