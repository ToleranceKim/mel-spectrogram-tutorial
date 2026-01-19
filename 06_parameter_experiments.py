"""
실습 6: 파라미터 실험

블로그 섹션: 6. 코드 예시, 파라미터 가이드

학습 목표:
- 각 파라미터가 결과에 미치는 영향 체감
- 자신의 태스크에 맞는 파라미터 선택 감각 기르기

실습 내용:
1. n_mels 비교 (64 vs 128 vs 256)
2. n_fft / hop_length 비교 (해상도 트레이드오프)
3. htk=True vs htk=False
4. ref 설정 비교 (np.max vs 1.0)

출력:
- outputs/06_n_mels_comparison.png
- outputs/06_resolution_comparison.png
- outputs/06_htk_comparison.png
- outputs/06_ref_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import load_config, ensure_output_dir

def compare_n_mels(signal: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int
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
    """
    print("\n[1] n_mels 비교 (64 vs 128 vs 256)")

    n_mels_values = [64, 128, 256]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for i, n_mels in enumerate(n_mels_values):
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

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
    n_mels: int
    ) -> None:
    f"""
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
    """
    print("\n[2] n_fft / hop_length 비교 (해상도 트레이드오프)")

    # (n_fft, hop_length) 조합
    settings = [
        (512, 128), # 시간 해상도 우선
        (2048, 512), # 균형 (기본값)
        (4096, 1024), # 주파수 해상도 우선
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for i, (n_fft, hop_length) in enumerate(settings):
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

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
    n_mels: int
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
            n_mels=n_mels, htk=htk
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

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
    n_mels: int
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
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # 두 가지 ref 설정
    mel_db_max = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db_one = librosa.power_to_db(mel_spec, ref=1.0)

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

def main():
    """메인 실행 함수 - 각 실습을 순차 실행"""

    config = load_config()
    sr = config['audio']['sr']
    n_fft = config['stft']['n_fft']
    hop_length = config['stft']['hop_length']
    n_mels = config['mel']['n_mels']

    ensure_output_dir()

    print("=" * 50)
    print("실습 6: 파라미터 실험")
    print("=" * 50)

    # 트럼펫 샘플 로드
    trumpet, _ = librosa.load(librosa.ex('trumpet'), sr=sr)

    # 실습 1: n_mels 비교
    compare_n_mels(trumpet, sr, n_fft, hop_length)

    # 실습 2: 해상도 트레이드오프
    compare_resolution(trumpet, sr, n_mels)

    # 실습 3: HTK vs Slaney
    compare_htk(trumpet, sr, n_fft, hop_length, n_mels)

    # 실습 4: ref 파라미터 비교
    compare_ref(trumpet, sr, n_fft, hop_length, n_mels)

    print("\n" + "=" * 50)
    print("실습 6 완료")
    print("=" * 50)

if __name__ == "__main__":
    main()

    