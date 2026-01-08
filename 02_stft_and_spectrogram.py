"""
실습 2: STFT와 스펙트로그램

블로그 섹션: 2. STFT: 시간-주파수 분석

학습 목표:
- STFT가 "시간에 따른 주파수 변화"를 분석함을 이해
- 윈도우 함수의 역할 이해 (스펙트럼 누설 방지)
- 시간-주파수 해상도 트레이드오프 체감

실습 내용:
1. STFT 개념 (프레임 나누기 -> FFT -> 시간축 쌓기)
2. 윈도우 함수 비교 (사각 vs 한)
3. 해상도 트레이드오프 실험 (n_fft 변경)
4. hop_length 실험

출력:
- outputs/02_window_comparison.png
- outputs/02_spectral_leakage.png
- outputs/02_resolution_tradeoff.png
- outputs/02_spectrogram.png
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display # 스펙트로그램 시각화 전용
from scipy.signal import get_window # 윈도우 함수 생성 (hann, boxcar 등)
from utils import load_config, generate_synthetic_audio, ensure_output_dir

def compare_windows(window_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    윈도우 함수 비교 시각화 (실습 1)

    윈도우 함수가 필요한 이유:
    - FFT는 신호가 무한히 반복된다고 가정
    - 유한한 구간을 자르면 시작/끝에서 불연속 발생
    - 불연속 -> 스펙트럼 누설(spectral leakage) 문제

    사각 윈도우 (Rectangular/Boxcar):
    - 모든 값이 1인 배열: [1, 1, 1, ..., 1]
    - 원본 신호를 그대로 사용하는 것과 동일
    - 급격한 경계로 인해 스펙트럼 누설 발생

    한 윈도우 (Hann):
    - 수식: w(n) = 0.5 * (1 - cos(2*pi*n / (N-1)))
    - 코사인 기반의 종 모양 곡선
    - 양 끝이 0, 중앙이 1 -> 신호를 부드럽게 감쇠
    - librosa, scipy 등 대부분의 라이브러리 기본값

    Args:
        window_size: 윈도우 크기 (샘플 수). 기본값 256.

    Returns:
        (window_rect, window_hann): 사각 윈도우와 한 윈도우 배열 튜플
    """
    # 두 가지 윈도우 함수 생성
    window_rect = get_window('boxcar', window_size) # 사각 윈도우: 모든 값이 1
    window_hann = get_window('hann', window_size) # 한 윈도우: 양 끝이 0으로 감쇠

    # 시각화 (2행 1열)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # 상단: 사각 윈도우
    axes[0].plot(window_rect, linewidth=1.5)
    axes[0].set_title('Rectangular Window (boxcar)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-0.1, 1.2)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.98, 0.85, 'All values = 1\nAbrupt edges cause spectral leakage',
                fontsize=9, color='red', ha='right', va='top',
                transform=axes[0].transAxes)

    # 하단: 한 윈도우
    axes[1].plot(window_hann, linewidth=1.5)
    axes[1].set_title('Hann Window')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-0.1, 1.2)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.98, 0.85, 'Smooth taper to zero\nReduces spectral leakage',
                fontsize=9, color='blue', ha='right', va='top',
                transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig('outputs/02_window_comparison.png', dpi=150)
    plt.close()

    print(f"    사각 윈도우: 모든 값이 1 (급격한 경계) -> 스펙트럼 누설 발생")
    print(f"    한 윈도우: 양 끝이 0으로 부드럽게 감소 -> 누설 감소")
    print(f"    저장: outputs/02_window_comparison.png")

    return window_rect, window_hann


def demonstrate_spectral_leakage(
    sr: int,
    window_size: int,
    window_rect: np.ndarray,
    window_hann: np.ndarray
) -> None:
    """
    스펙트럼 누설 시연 (실습 2)

    스펙트럼 누설(Spectral Leakage)란?
    - FFT는 신호가 무한히 반복된다고 가정
    - 실제로는 유한한 구간만 분석하므로 신호의 시작/끝에서 불연속 발생
    - 이 불연속에 FFT에서 "가짜" 주파수 성분으로 나타남 = 스펙트럼 누설

    윈도우 함수의 역할:
    - 신호의 양 끝을 부드럽게 0으로 감쇠시킴
    - 불연속을 제거하여 스펙트럼 누설 감소
    - 단점: 메인 로브(main lobe) 폭이 넓어져 주파수 해상도 약간 감소

    실험 내용:
    - 순수 440Hz 사인파에 사각/한 윈도우 적용
    - FFT 결과 비교: 사각 윈도우는 에너지 분산, 한 윈도우는 집중

    Args:
        sr: 샘플링 레이트
        window_size: 윈도우 크기 (샘플 수)
        window_rect: 사각 윈도우 배열
        window_hann: 한 윈도우 배열
    """
    print("\n[2] 스펙트럼 누설 시연")

    # 테스트 신호: 순수 440Hz 사인파
    # sin(2 * pi * f * t) where f=440Hz
    # 이론적으로 FFT 결과는 440Hz에만 피크가 있어야 함
    t = np.arange(window_size) / sr
    sine_440 = np.sin(2 * np.pi * 440 * t)

    # 윈도우 함수 적용
    # 윈도우 적용 = 원본 신호 * 윈도우 함수 (element-wise 곱)
    sine_rect = sine_440 * window_rect
    sine_hann = sine_440 * window_hann

    # FFT 계산 후 magnitude 추출
    fft_rect = np.abs(np.fft.fft(sine_rect))
    fft_hann = np.abs(np.fft.fft(sine_hann))

    # 주파수 축 생성
    freqs = np.fft.fftfreq(window_size, 1 / sr)

    # 양의 주파수만 사용
    # FFT 결과는 대칭: [0, 양의주파수, 음의주파수]
    # 실수 신호는 양의 주파수만으로 충분
    positive_mask = freqs >= 0
    freqs_positive = freqs[positive_mask]
    fft_rect_positive = fft_rect[positive_mask] / np.max(fft_rect[positive_mask])
    fft_hann_positive = fft_hann[positive_mask] / np.max(fft_hann[positive_mask])

    # 시각화 (2행 1열)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # 상단: 사각 윈도우 FFT
    axes[0].plot(freqs_positive, fft_rect_positive, linewidth=0.8)
    axes[0].set_title('FFT with Rectangular Window')
    axes[0].set_ylabel('Magnitude (normalized)')
    axes[0].set_xlim(0, 1000)
    axes[0].axvline(x=440, color='red', linestyle='--', alpha=0.7, label='440Hz')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.98, 0.85, 'Energy spreads to\nneighboring frequencies',
                fontsize=9, color='red', ha='right', va='top',
                transform=axes[0].transAxes)

    # 하단: 한 윈도우 FFT
    axes[1].plot(freqs_positive, fft_hann_positive, linewidth=0.8)
    axes[1].set_title('FFT with Hann Window')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (normalized)')
    axes[1].set_xlim(0, 1000)
    axes[1].axvline(x=440, color='red', linestyle='--', alpha=0.7, label='440Hz')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.98, 0.85, 'Energy concentrated\nat 440Hz',
                fontsize=9, color='blue', ha='right', va='top',
                transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig('outputs/02_spectral_leakage.png', dpi=150)
    plt.close()

    print(f"    사각 윈도우: 440Hz 외 주변 주파수에 에너지 분산 (넓은 피크)")
    print(f"    한 윈도우: 440Hz에 에너지 집중 (날카로운 피크)")
    print(f"    저장: outputs/02_spectral_leakage.png")
    

def experiment_resolution_tradeoff(
    signal: np.ndarray,
    sr: int,
    n_fft_values: list[int]
) -> None:
    """
    시간-주파수 해상도 트레이드오프 실험 (실습 3)

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트
        n_fft_values: 비교할 n_fft 값들
    """
    pass

def visualize_spectrogram(
    signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int
) -> None:
    """
    스펙트로그램 시각화 (실습 4)

    Args:
        signal: 오디오 신호
        sr: 샘플링 레이트
        n_fft: FFT 크기
        hop_length: 홉 길이
    """
    pass

def main():
    """메인 실행 함수 - 각 실습을 순차 실행"""

    config = load_config()
    sr = config['audio']['sr']
    n_fft = config['stft']['n_fft']
    hop_length = config['stft']['hop_length']

    ensure_output_dir()

    print("=" * 50)
    print("실습 2: STFT와 스팩트로그램")
    print("=" * 50)

    # 실습 1: 윈도우 함수 비교
    window_size = 256
    window_rect, window_hann = compare_windows(window_size)

    # 실습 2: 스펙트럼 누설 시연
    demonstrate_spectral_leakage(sr, window_size, window_rect, window_hann)

    # 실습 3: 해상도 트레이드오프
    signal = generate_synthetic_audio(sr, duration=2.0)
    experiment_resolution_tradeoff(signal, sr, [512, 2048, 4096])

    # 실습 4: 스펙트로그램 시각화
    visualize_spectrogram(signal, sr, n_fft, hop_length)

    print("\n" + "=" * 50)
    print("실습 2 완료")
    print("=" * 50)

if __name__ == "__main__":
    main()
