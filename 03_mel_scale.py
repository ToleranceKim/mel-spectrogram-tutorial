"""
실습 3: 멜 스케일

블로그 섹션: 3. 멜 스케일: 사람처럼 듣기

학습 목표:
- 멜 스케일이 왜 필요한지 이해 (청각 특성 반영)
- HTK 공식으로 Hz ↔ Mel 변환
- 멜 필터 뱅크의 구조 이해

실습 내용:
1. 멜 스케일 변환 (HTK 공식 직접 구현)
2. HTK vs Slaney 비교
3. 멜 필터 뱅크 시각화
4. 선형 스펙트로그램 vs 멜 스펙트로그램

출력:
- outputs/03_mel_scale_curve.png
- outputs/03_htk_vs_slaney.png
- outputs/03_mel_filterbank.png
- outputs/03_linear_vs_mel.png
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import load_config, hz_to_mel, mel_to_hz, ensure_output_dir

def visualize_mel_curve() -> None:
    """
    멜 스케일 곡선 시각화 (실습 1)

    멜 스케일(Mel Scale)이란?
    - 사람의 청각 인지를 반영한 주파수 스케일
    - 저주파에서는 주파수 변화를 민감하게, 고주파에서는 둔감하게 인지
    - 예: 100Hz -> 200Hz 차이는 크게, 8000Hz -> 8100Hz 차이는 작게 느낌

    HTK 공식:
    - mel = 2595 * log10(1 + hz / 700)
    - hz = 700 * (10^(mel / 2595) - 1)

    왜 멜 스케일을 사용하는가?
    - 사람이 듣는 방식과 유사하게 주파수를 표현
    - 음성 인식, 음악 분류 등 청각 관련 태스크에 효과적
    - 고주파 영역의 불필요한 세부사항을 압축
    """
    print("\n[1] 멜 스케일 변환 곡선")

    # Hz 범위 생성 (0 ~ 11025Hz, 나이퀴스트 주파수)
    hz_values = np.linspace(0, 11025, 1000)

    # Hz -> Mel 변환 (HTK 공식)
    mel_values = hz_to_mel(hz_values)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(hz_values, mel_values, linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mel')
    plt.title('Hz to Mel Scale Conversion (HTK Formula)')
    plt.grid(True, alpha=0.3)

    plt.text(0.98, 0.15, 'Low frequencies: steep curve\n(sensitive to changes)',
            fontsize=9, color='blue', ha='right', va='bottom',
            transform=plt.gca().transAxes)
    plt.text(0.98, 0.85, 'High frequencies: flat curve\n(less sensitive)',
            fontsize=9, color='gray', ha='right', va='top',
            transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig('outputs/03_mel_scale_curve.png', dpi=150)
    plt.close()

    print(f"    HTK 공식: mel = 2595 * log10(1 + hz / 700)")
    print(f"    저장: outputs/03_mel_scale_curve.png")

def compare_htk_slaney(sr: int) -> None:
    """
    HTK vs Slaney 멜 스케일 비교 (실습 2)

    두 가지 멜 스케일 공식:
    - HTK: mel = 2595 * log10(1 + hz / 700)
        * 전통적인 음성 인식에서 사용
        * librosa에서 htk=True로 설정
    - Slaney: 1000Hz 이하는 선형, 이상은 로그 스케일
        * Auditory Toolbox에서 유래
        * librosa 기본값 (htk=False)

    실무에서 선택:
    - 대부분의 경우 큰 차이 없음
    - 기존 모델과 호환성이 중요하면 해당 모델의 설정 따르기
    - 새 프로젝트는 librosa 기본값(Slaney) 사용해도 무방

    Args:
        sr: 샘플링 레이트 (Hz)
    """
    print("\n[2] HTK vs Slaney 멜 스케일 비교")

    # 주파수 범위
    hz_values = np.linspace(0, sr // 2, 1000)

    # HTK 방식 (librosa)
    mel_htk = librosa.hz_to_mel(hz_values, htk=True)

    # Slaney 방식 (librosa)
    mel_slaney = librosa.hz_to_mel(hz_values, htk=False)

    # 시각화
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # 상단: 두 곡선 비교
    axes[0].plot(hz_values, mel_htk, label='HTK', linewidth=2)
    axes[0].plot(hz_values, mel_slaney, label='Slaney', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Mel')
    axes[0].set_title('HTK vs Slaney Mel Scale')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 하단: 차이값
    diff = mel_htk - mel_slaney
    axes[1].plot(hz_values, diff, linewidth=2, color='red')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Difference (HTK - Slaney)')
    axes[1].set_title('Difference between HTK and Slaney')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/03_htk_vs_slaney.png', dpi=150)
    plt.close()

    print(f"    HTK: 전통적 음성 인식 방식")
    print(f"    Slaney: librosa 기본값, 1000Hz 기준 선형/로그 혼합")
    print(f"    저장: outputs/03_htk_vs_slaney.png")

def visualize_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> None:
    """
    멜 필터 뱅크 시각화 (실습 3)

    멜 필터 뱅크(Mel Filter Bank)란?
    - 선형 주파수 스펙트로그램을 멜 스케일로 변환하는 필터 집합
    - 삼각형 모양의 필터들이 멜 스케일 상에서 균등하게 배치
    - 각 필터는 특정 주파수 대역의 에너지를 합산

    필터 뱅크의 특징:
    - 저주파: 필터 폭이 좁음 (세밀한 분석)
    - 고주파: 필터 폭이 넓음 (거친 분석)
    - 인접 필터끼리 50% 오버랩

    행렬 연산으로 이해:
    - 멜 필터 뱅크: (n_mels, n_fft//2+1) 크기 행렬
    - 파워 스펙트로그램: (n_fft//2+1, time_frames)
    - 멜 스펙트로그램 = 필터뱅크 @ 파워스펙트로그램

    Args:
        sr: 샘플링 레이트 (Hz)
        n_fft: FFT 크기
        n_mels: 멜 필터 개수
    """
    print("\n[3] 멜 필터 뱅크 시각화")

    # 멜 필터 뱅크 생성
    mel_filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    # 주파수 축 생성
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 시각화
    plt.figure(figsize=(12, 6))

    # 모든 필터 플롯
    for i in range(n_mels):
        plt.plot(freqs, mel_filterbank[i], linewidth=0.8, alpha=0.7)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Weight')
    plt.title(f'Mel Filter Bank ({n_mels} filters)')
    plt.xlim(0, sr // 2)
    plt.grid(True, alpha=0.3)

    plt.text(0.98, 0.95, 'Low freq: narrow filters\nHigh freq: wide filters',
            fontsize=9, color='gray', ha='right', va='top',
            transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig('outputs/03_mel_filterbank.png', dpi=150)
    plt.close()

    print(f"    필터 개수: {n_mels}")
    print(f"    필터 뱅크 shape: {mel_filterbank.shape}")
    print(f"    저장: outputs/03_mel_filterbank.png")
    