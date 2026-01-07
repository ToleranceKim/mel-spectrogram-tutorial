"""
공통 유틸리티 함수

이 모듈은 모든 실습에서 공통으로 사용하는 함수들을 제공합니다:
- 설정 파일 로드
- 합성 오디오 생성
- Hz ↔ Mel 변환
- 플롯 스타일 설정
"""

import os
import yaml
import numpy as np

# 설정 파일 로드 함수
def load_config(path="config.yaml"):
    """설정 파일 로드

    Args:
        path: YAML 설정 파일 경로

    Returns:
        dict: 설정 딕셔너리
    """
    # YAML 파일을 Python dict로 변환
    with open(path, "r", encoding="utf-8") as f: # 한글 주석이 있을 수 있으므로 명시
        return yaml.safe_load(f)

# 사인파 생성 함수
def generate_sine_wave(freq, sr, duration):
    """단일 주파수 사인파 생성

    Args:
        freq: 주파수(Hz)
        sr: 샘플링 레이트
        duration: 길이 (초)
        
    Returns:
        numpy array: 사인파 신호
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint =False) # 0부터 duration까지 int(sr * duration)개의 샘플을 생성
    return np.sin(2 * np.pi * freq * t) # 사인파 공식, 반환값 범위는 -1에서 1 사이

# 복합파 생성 함수
def generate_complex_wave(freqs, amps, sr, duration):
    """복합파 생성 (여러 주파수 + 진폭)

    Args:
        freqs: 주파수 리스트 (Hz)
        amps: 각 주파수의 진폭 리스트
        sr: 샘플링 레이트
        duration: 길이 (초)

    Returns:
        numpy array: 정규화된 복합파 신호 (-1 ~ 1)

    사용 예시
    # 440Hz(기본음) + 880Hz(2배음) + 1320Hz(3배음)
    wave = generate_complex_wave(
        freqs=[440, 880, 1320],
        amps=[1.0, 0.5, 0.25],
        sr=22050,
        duration=1.0)
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps)) # 주파수와 진폭을 쌍으로 묶음
    return signal / np.max(np.abs(signal)) # 최대값이 1이 되도록 정규화

# 합성 오디오 생성 함수 (배음 포함)
def generate_synthetic_audio(sr, duration=3.0):
    """배음이 있는 합성 오디오 생성

    440Hz 기본음에 2, 3, 4배음을 추가하고
    감쇠 엔베로프를 적용한 피아노 같은 소리

    Args:
        sr: 샘플링 레이트
        duration: 길이 (초)

    Returns:
        numpy array: 정규화된 신호(-1 ~ 1)
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 440 # 기본 주파수 (A4)

    # 기본음 + 배음 (진폭이 절반씩 감소)
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t) +        # 기본음
        0.25 * np.sin(2 * np.pi * 2 * f0 * t) +   # 2배음 (880Hz)
        0.125 * np.sin(2 * np.pi * 3 * f0 * t) +  # 3배음 (1320Hz)
        0.0625 * np.sin(2 * np.pi * 4 * f0 * t)   # 4배음 (1760Hz)
    )

    # 감쇠 엔벨로프 (시간이 지나면 소리가 작아짐)
    envelope = np.exp(-t * 1.5) # 지수 감쇠 (피아노 건반을 누른 후 소리가 점점 작아지는 효과)
    signal = signal * envelope

    return signal / np.max(np.abs(signal))

# Hz -> Mel 변환 함수
def hz_to_mel(f):
    """Hz -> Mel 변환 (HTK 공식)

    공식: m = 2595 * log10(1 + f/700)

    Args:
        f: 주파수 (Hz), 스칼라 또는 numpy array
    
    Returns:
        멜 값

    Example:
        >>> hz_to_mel(440)  # A4 음
        549.64
        >>> hz_to_mel(1000) # 1kHz
    """
    return 2595 * np.log10(1 + f / 700)

# Mel -> Hz 변환 함수
def mel_to_hz(m):
    """Mel -> Hz 변환 (HTK 역변환)

    공식: f = 700 * (10^(m/2595) -1)

    Args:
        m: 멜 값, 스칼라 또는 numpy array

    Returns:
        주파수 (Hz)

    Example:
        >>> mel_to_hz(549.64)
        440.0
    """
    return 700 * (10 ** (m / 2595) - 1)

# outputs 디렉토리 생성 함수
def ensure_output_dir(path="outputs"):
    """출력 디렉토리가 없으면 생성

    Args:
        path: 디렉토리 경로
    """
    os.makedirs(path, exist_ok=True)
    