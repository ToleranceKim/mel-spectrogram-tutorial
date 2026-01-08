"""
실습 1: 파형과 주파수

블로그 섹션: 1. 배경: 소리와 주파수

학습 목표:
- 오디오가 숫자의 나열(파형)임을 이해
- 주파수와 배음의 개념 이해
- FFT로 주파수 성분을 분석하는 방법 이해

실습 내용:
1. 파형 시각화
2. 사인파 합성 (단일파, 복합파)
3. FFT 분석

출력:
- outputs/01_synthetic_waveform.png
- outputs/01_sine_waves.png
- outputs/01_fft_analysis.png
- outputs/01_trumpet_waveform.png
- outputs/01_trumpet_fft.png
- outputs/01_sine_single.wav
- outputs/01_sine_complex.wav
- outputs/01_synthetic.wav
- outputs/01_trumpet.wav
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from utils import (
    load_config,
    generate_sine_wave,
    generate_complex_wave,
    generate_synthetic_audio,
    ensure_output_dir
)

def main():
    """메인 실행 함수"""

    # 설정 로드
    config = load_config()
    sr = config['audio']['sr'] # 22050, 샘플링 레이트 가져오기

    # 출력 디렉토리 확인
    ensure_output_dir()

    print("=" * 50)
    print("실습 1: 파형과 주파수")
    print("=" * 50)

    # ========================================
    # 실습 1: 파형 시각화
    # ========================================
    print("\n[1] 파형 시각화")

    # 합성 오디오 생성 (배음 포함)
    duration = 0.5 # 0.5초
    signal = generate_synthetic_audio(sr, duration) # 440Hz + 배음이 있는 신호 생성

    # 시간 축 생성 (x축)
    t = np.linspace(0, duration, len(signal), endpoint=False)

    # 파형 플롯
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform - Synthetic Audio with Harmonics')
    plt.grid(True, alpha=0.3)

    # 주석 추가: 파형의 특징 설명
    plt.text(0.98, 0.95, 'Amplitude = loudness of sound',
            fontsize=9, color='blue', ha='right', va='top',
            transform=plt.gca().transAxes)
    plt.text(0.98, 0.05, 'Periodic pattern = constant frequency',
            fontsize=9, color='gray', ha='right', va='bottom',
            transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig('outputs/01_synthetic_waveform.png', dpi=150) # 그래프를 파일로 저장
    plt.close() # 메모리 해제

    print(f"    샘플링 레이트: {sr} Hz")
    print(f"    신호 길이: {len(signal)} 샘플 ({duration}초)")
    print(f"    저장: outputs/01_synthetic_waveform.png")

    # ========================================
    # 실습 2: 사인파 합성 (단일파 vs 복합파)
    # ========================================
    print("\n[2] 사인파 합성")

    duration = 0.05 # 50mx (파형 비교용 짧은 구간)

    # 단일 사인파 (440Hz)
    sine_single = generate_sine_wave(440, sr, duration)

    # 복합파 (440Hz + 880Hz + 1320Hz)
    sine_complex = generate_complex_wave(
        freqs=[440, 880, 1320],
        amps=[1.0, 0.5, 0.25],
        sr=sr,
        duration=duration
    )

    # 시간 축
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 비교 플롯 (2행 1열)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(t * 1000, sine_single, linewidth=0.8) # ms 단위
    axes[0].set_title('Single Sine Wave (440Hz)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    axes[0].text(0.98, 0.85, 'Pure Tone: single frequency only',
                fontsize=9, color='blue', ha='right', va='top',
                transform=axes[0].transAxes)

    axes[1].plot(t * 1000, sine_complex, linewidth=0.8)
    axes[1].set_title('Complex Wave (440Hz + 880Hz + 1320Hz)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    axes[1].text(0.98, 0.85, 'Complex: harmonics add richness',
                fontsize=9, color='blue', ha='right', va='top',
                transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig('outputs/01_sine_waves.png', dpi=150)
    plt.close()

    # 청취용 WAV 저장 (1초 버전)
    duration_listen = 1.0
    sine_single_listen = generate_sine_wave(440, sr, duration_listen)
    sine_complex_listen = generate_complex_wave(
        freqs=[440, 880, 1320],
        amps=[1.0, 0.5, 0.25],
        sr=sr,
        duration=duration_listen
    )
    sf.write('outputs/01_sine_single.wav', sine_single_listen, sr)
    sf.write('outputs/01_sine_complex.wav', sine_complex_listen, sr)

    print(f"    단일파: 440Hz")
    print(f"    복합파: 440Hz(기본음) + 880Hz(2배음) + 1320Hz(3배음)")
    print(f"    저장: outputs/01_sine_waves.png")

    # ========================================
    # 실습 3: FFT 분석
    # ========================================
    print("\n[3] FFT 분석")

    # 청취용 오디오 생성 (1초, 더 긴 버전)
    duration_audio = 1.0
    audio_for_listen = generate_synthetic_audio(sr, duration_audio)

    # WAV 파일 저장
    # soundfile은 numpy array를 Wav로 변환해줌
    sf.write('outputs/01_synthetic.wav', audio_for_listen, sr)
    print(f"    청취용 WAV 저장: outputs/01_synthetic.wav")

    # FFT 분석용 신호 (배음 포함)
    duration = 0.5
    signal = generate_synthetic_audio(sr, duration)

    # FFT 계산
    # FFT: 시간 영역 신호 -> 주파수 영역 변환
    # 입력: 시간에 따른 진폭 값들 [x0, x1, x2, ...]
    # 출력: 각 주파수 성분의 복소수 값 (크기 + 위상 정보 포함)
    fft_result = np.fft.fft(signal) # FFT 계산

    # 주파수 축 생성
    # fftfreq(n, d): n개 샘플, d=샘플 간격(초)
    # d = 1/sr: 샘플링 레이트가 22050이면, 샘플 간격은 1/22050초
    # 반환값: 각 FFT 빈(bin)에 해당하는 주파수 (Hz 단위)
    # 1/sr = 샘플 간격 (초 단위)
    freqs = np.fft.fftfreq(len(signal), 1/sr)   # 주파수 축 생성

    # 양의 주파수만 사용
    # FFT 결과는 대칭 구조: [0, 양의주파수들, 음의주파수들]
    # 음의 주파수는 양의 주파수와 켤레복소수 관계 (실수 신호의 특성)
    # 실제 분석에서는 양의 주파수 절반만 사용해도 충분
    positive_mask = freqs >= 0
    freqs_positive = freqs[positive_mask]

    # 복소수 -> 크기(magnitude) 변환
    # FFT 결과는 복소수: a + bj (실수부 + 허수부)
    # magnitude = sqrt(a^2 + b^2) = 해당 주파수 성분의 "세기"
    magnitude = np.abs(fft_result[positive_mask])

    # 정규화 (최대값 = 1)
    # 절대적인 크기보다 상대적인 비율이 중요
    # 기본음 대비 배음의 세기를 비교하기 위함
    magnitude = magnitude / np.max(magnitude)

    # FFT 결과 플롯
    plt.figure(figsize=(10, 4))
    plt.plot(freqs_positive, magnitude, linewidth=0.8)
    plt.xlim(0, 2500) # 0~2500Hz 범위만 표시 (배음이 이 범위 내에 있음)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (normalized)')
    plt.title('FFT Analysis - Frequency Components')
    plt.grid(True, alpha=0.3)

    # 그래프 우측에 설명 추가
    plt.text(0.98, 0.95, 'Peak height = relative\nstrength of frequency',
            fontsize=9, color='gray', ha='right', va='top',
            transform=plt.gca().transAxes)
    plt.text(0.98, 0.70, 'Fundamental (440Hz) strongest,\nharmonics decrease',
            fontsize=8, color='gray', ha='right', va='top',
            transform=plt.gca().transAxes)

    # 예상 피크 주파수 표시 (빨간 점선)
    for freq in [440, 880, 1320, 1760]:
        plt.axvline(x=freq, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        plt.text(freq, 0.9, f'{freq}Hz', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/01_fft_analysis.png', dpi=150)
    plt.close()

    print(f"    FFT로 검출된 주파수: 440Hz(기본음), 880Hz, 1320Hz, 1760Hz")
    print(f"    저장: outputs/01_fft_analysis.png")

    # ========================================
    # 실습 4: 실제 악기 소리 분석 (트럼펫)
    # ========================================

    print("\n[4] 실제 악기 소리 분석 (트럼펫)")

    # 트럼펫 샘플 로드
    trumpet_path = librosa.example('trumpet')
    trumpet, _ = librosa.load(trumpet_path, sr=sr) # 지정한 sr로 리샘플링

    # 분석용 구간 (0.5 초)
    trumpet_segment = trumpet[:int(sr * 0.5)]
    t_trumpet = np.linspace(0, 0.5, len(trumpet_segment), endpoint=False)

    # 파형 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(t_trumpet, trumpet_segment, linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.title('Trumpet Waveform (Real Instrument)')
    plt.ylabel('Amplitude')
    plt.text(0.98, 0.95, 'Real instrument: complex waveform',
            fontsize=9, color='blue', ha='right', va='top',
            transform=plt.gca().transAxes)
    plt.text(0.98, 0.05, 'Timbre changes over time',
            fontsize=9, color='gray', ha='right', va='bottom',
            transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/01_trumpet_waveform.png', dpi=150)
    plt.close()

    # FFT 분석 (트럼펫)
    # 실제 악기는 합성음과 달리:
    # - 더 많은 배음 성분
    # - 주파수 피크가 넓게 퍼짐 (노이즈, 비정수배 성분)
    # - 시간에 따라 스펙트럼이 변화
    fft_trumpet = np.fft.fft(trumpet_segment)
    
    # 주파수 축 생성
    # 1/sr = 샘플 간격 (초 단위)
    # sr=22050 -> 1/sr = 0.0000454초 (각 샘플 사이의 시간)
    # 이 정보가 있어야 FFT 결과를 Hz 단위로 해석 가능
    freqs_trumpet = np.fft.fftfreq(len(trumpet_segment), 1/sr) 

    pos_mask = freqs_trumpet >= 0
    freqs_pos = freqs_trumpet[pos_mask]
    mag_trumpet = np.abs(fft_trumpet[pos_mask])
    mag_trumpet = mag_trumpet / np.max(mag_trumpet)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_pos, mag_trumpet, linewidth=0.8)
    plt.xlim(0, 3000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (normalized)')
    plt.title('FFT Analysis - Trumpet')
    plt.text(0.98, 0.95, 'Synthetic: discrete peaks only\nReal: continuous distribution',
            fontsize=9, color='gray', ha='right', va='top',
            transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/01_trumpet_fft.png', dpi=150)
    plt.close()

    # 청취용 WAV 저장 (2초)
    sf.write('outputs/01_trumpet.wav', trumpet[:int(sr * 2)], sr)

    print(f"    트럼팻 샘플: {len(trumpet)} 샘플 ({len(trumpet)/sr:.1f}초)")
    print(f"    저장: outputs/01_trumpet_waveform.png")
    print(f"    저장: outputs/01_trumpet_fft.png")
    print(f"    저장: outputs/01_trumpet.wav")

    print("\n" + "=" * 50)
    print("실습 1 완료")
    print("=" * 50)

if __name__ == "__main__":
    main()