"""
실습 4: 멜 스펙트로그램 파이프라인

블로그 섹션: 4. 멜 스펙트로그램 생성

원본 오디오가 어떤 과정을 거쳐 2D 이미지(멜 스펙트로그램)가 되는지 
단계별로 직접 구현하고 시각화한다.

학습 목표:
- 전체 파이프라인을 단계별로 실행
- 각 단계의 출력 shape 변화 추적
- dB 변환의 효과 체감 (범위 압축)
- 행렬 연산으로 멜 필터 적용 이해 (M @ P)

파이프라인 요약:
1. 오디오 파형 (1D) - 시간 영역 신호
2. STFT - 복소수 스펙트로그램 (2D)
3. 파워 스펙트로그램 - |STFT|^2 (2D)
4. 멜 스펙트로그램 - 멜 필터뱅크 행렬곱 (2D)
5. dB 변환 - 10 * log10 (2D) - CNN 입력으로 사용

실습 내용:
1. 파이프라인 단계별 실행 및 시각화
2. dB 변환 효과 비교 (linear vs dB)
3. 최종 멜 스펙트로그램 + 행렬 연산 설명

출력:
- outputs/04_pipeline_steps.png
- outputs/04_db_conversion.png
- outputs/04_final_melspec.png
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import load_config, ensure_output_dir

def visualize_pipeline_steps(signal: np.ndarray,
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
   norm: str = "slaney"
   ) -> None:
   f"""
   멜 스펙트로그램 파이프라인 단계별 시각화 (실습 1)

   파이프라인 5단계:
   1. 오디오 파형: 시간 영역 신호 (1D 배열)
   2. STFT: 복소수 스펙트로그램 (주파수 x 시간)
   3. 파워 스펙트로그램: |STFT|^2 (에너지)
   4. 멜 스펙트로그램: 멜 필터뱅크 적용 (M @ P)
   5. dB 스케일: 로그 스케일 변환

   각 단계의 shape 변화:
   - 파형: (samples,) 예: (22050,)
   - STFT: (n_fft//2+1, frames) 예: (1025, 87)
   - 파워: (n_fft//2+1, frames) - STFT와 동일
   - 멜: (n_mels, frames) 예: (128, 87)
   - dB: (n_mels, frames) - 멜과 동일

   Args:
      signal: 오디오 신호
      sr: 샘플링 레이트 (Hz)
      n_fft: FFT 크기
      hop_length: 홉 길이
      n_mels: 멜 필터 개수
   """
   print("\n[1] 멜 스펙트로그램 파이프라인")

   # Step 1: 오디오 파형 (이미 입력으로 받음)
   print(f"    Step 1 - 파형: shape = {signal.shape}")

   # Step 2: STFT
   stft_result = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window=window,
                              center=center, pad_mode=pad_mode)
   print(f"    Step 2 - STFT: shape = {stft_result.shape}")

   # Step 3: 파워 스펙트로그램
   power_spec = np.abs(stft_result) ** 2
   print(f"    Step 3 - 파워: shape = {power_spec.shape}")

   # Step 4: 멜 필터 적용 (행렬 곱)
   mel_filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=fmin, fmax=fmax, htk=htk, norm=norm)
   mel_spec = mel_filterbank @ power_spec
   print(f"    Step 4 - 멜: shape = {mel_spec.shape}")

   # Step 5: dB 변환
   mel_db = librosa.power_to_db(mel_spec, ref=np.max)
   print(f"    Step 5 - dB: shape = {mel_db.shape}")

   # 시각화 (5행 1열)
   fig, axes = plt.subplots(5, 1, figsize=(12, 14))

   # 1. 파형
   time_axis = np.arange(len(signal)) / sr
   axes[0].plot(time_axis, signal, linewidth=0.5)
   axes[0].set_xlabel('Time (s)')
   axes[0].set_ylabel('Amplitude')
   axes[0].set_title('Step 1: Waveform')
   axes[0].set_xlim(0, len(signal) / sr)

   # 2. STFT (magnitude)
   stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
   img2 = librosa.display.specshow(
      stft_db, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='hz', ax=axes[1], cmap='magma'
   )
   axes[1].set_title('Step 2: STFT Magnitude')
   fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

   # 3. 파워 스펙트로그램
   power_db = librosa.power_to_db(power_spec, ref=np.max)
   img3 = librosa.display.specshow(
      power_db, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='hz', ax=axes[2], cmap='magma'
   )
   axes[2].set_title('Step 3: Power Spectrogram')
   fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')

   # 4. 멜 스펙트로그램 (linear scale)
   mel_linear_db = librosa.power_to_db(mel_spec, ref=np.max)
   img4 = librosa.display.specshow(
      mel_linear_db, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', ax=axes[3], cmap='magma'
   )
   axes[3].set_title('Step 4: Mel Spectrogram')
   fig.colorbar(img4, ax=axes[3], format='%+2.0f dB')

   # 5. 최종 dB 스케일
   img5 = librosa.display.specshow(
      mel_db, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', ax=axes[4], cmap='magma'
   )
   axes[4].set_title('Step 5: Mel Spectrogram (dB)')
   fig.colorbar(img5, ax=axes[4], format='%+2.0f dB')

   plt.tight_layout()
   plt.savefig('outputs/04_pipeline_steps.png', dpi=150)
   plt.close()

   print(f"    저장: outputs/04_pipeline_steps.png")

def visualize_db_conversion(signal: np.ndarray,
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
   dB 변환 효과 시각화 (실습 2)

   왜 dB 변환이 필요한가?
   - 소리 에너지는 매우 넓은 범위를 가짐 (동적 범위가 큼)
   - 예: 조용한 소리 0.001 vs 큰 소리 1000 (백만 배 차이)
   - 선형 스케일로 보면 작은 값들이 모두 0처럼 보임

   dB 변환의 효과:
   - 로그 스케일로 변환하여 동적 범위 압축
   - 작은 값도 시각적으로 구분 가능
   - 사람의 청각도 로그 스케일에 가까움

   power_to_db 공식:
   - db = 10 * log10(power / ref)
   - ref=np.max: 최대값을 0dB로 기준
   - 결과: 0dB(최대) ~ -80dB(최소) 범위

   Args:
      signal: 오디오 신호
      sr: 샘플링 레이트 (Hz)
      n_fft: FFT 크기
      hop_length: 홉 길이
      n_mels: 멜 필터 개수
   """
   print("\n[2] dB 변환 효과")

   # 멜 스펙트로그램 생성
   mel_spec = librosa.feature.melspectrogram(
      y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
      win_length=win_length, window=window, center=center, pad_mode=pad_mode,
      power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
   )

   # dB 변환
   mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)

   # 시각화 (2행 1열)
   fig, axes = plt.subplots(2, 1, figsize=(12, 8))

   # 상단: Linear scale (dB 변환 전)
   img1 = librosa.display.specshow(
      mel_spec, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', ax=axes[0], cmap='magma'
   )
   axes[0].set_title('Mel Spectrogram (Linear Scale)')
   fig.colorbar(img1, ax=axes[0], format='%.2f')

   # 하단: dB scale (dB 변환 후)
   img2 = librosa.display.specshow(
      mel_db, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', ax=axes[1], cmap='magma'
   )
   axes[1].set_title('Mel Spectrogram (dB Scale)')
   fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

   plt.tight_layout()
   plt.savefig('outputs/04_db_conversion.png', dpi=150)
   plt.close()

   print(f"    Linear 범위: {mel_spec.min():.6f} ~ {mel_spec.max():.6f}")
   print(f"    dB 범위: {mel_db.min():.1f} ~ {mel_db.max():.1f} dB")
   print(f"    저장: outputs/04_db_conversion.png")

def visualize_final_melspec(signal: np.ndarray,
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
   최종 멜 스펙트로그램 및 행렬 연산 설명 (실습 3)

   행렬 연산으로 이해하기:
   - 멜 필터뱅크 M: (n_mels, n_fft//2+1) = (128, 1025)
   - 파워 스펙트로그램 P: (n_fft//2+1, frames) = (1025, 230)
   - 멜 스펙트로그램 = M @ P: (128, 230)

   librosa.feature.melspectrogram vs 직접 구현:
   - librosa 함수: 내부적으로 STFT + 파워 + 필터뱅크 적용
   - 직접 구현: 각 단계를 명시적으로 수행
   - 결과는 동일함

   Args:
      signal: 오디오 신호
      sr: 샘플링 레이트 (Hz)
      n_fft: FFT 크기
      hop_length: 홉 길이
      n_mels: 멜 필터 개수
   """
   print("\n[3] 최종 멜 스펙트로그램")

   # 방법 1: librosa 함수 사용
   mel_spec_librosa = librosa.feature.melspectrogram(
      y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
      win_length=win_length, window=window, center=center, pad_mode=pad_mode,
      power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
   )

   # 방법 2: 직접 행렬 연산
   stft_result = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window=window,
                              center=center, pad_mode=pad_mode)
   power_spec = np.abs(stft_result) ** 2
   mel_filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=fmin, fmax=fmax, htk=htk, norm=norm)
   mel_spec_manual = mel_filterbank @ power_spec

   # 두 방법 결과 비교
   diff = np.abs(mel_spec_librosa - mel_spec_manual).max()
   print(f"    librosa vs 직접구현 최대 차이: {diff:.10f}")

   # 행렬 shape 출력
   print(f"    멜 필터뱅크 M: {mel_filterbank.shape}")
   print(f"    파워 스펙트로그램 P: {power_spec.shape}")
   print(f"    멜 스펙트로그램 M @ P: {mel_spec_manual.shape}")

   # dB 변환
   mel_db = librosa.power_to_db(mel_spec_librosa, ref=np.max)

   # 시각화
   plt.figure(figsize=(12, 5))
   img = librosa.display.specshow(
      mel_db, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', cmap='magma'
   )
   plt.colorbar(img, format='%+2.0f dB')
   plt.title(f"Final Mel Spectrogram ({n_mels} mels)")

   plt.tight_layout()
   plt.savefig('outputs/04_final_melspec.png', dpi=150)
   plt.close()

   print(f"    저장: outputs/04_final_melspec.png")

def main():
   """메인 실행 함수 - 각 실습을 순차 실행"""

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
   db_ref = config['db']['ref']
   amin = config['db']['amin']
   top_db = config['db']['top_db']
   power = config['db']['power']

   ensure_output_dir()

   print("=" * 50)
   print("실습 4: 멜 스펙트로그램 파이프라인")
   print("=" * 50)

   # 트럼펫 샘플 로드
   trumpet, _ = librosa.load(librosa.ex('trumpet'), sr=sr)

   # 실습 1: 파이프라인 단계별 시각화
   visualize_pipeline_steps(trumpet, sr, n_fft, hop_length, n_mels,
                            win_length, window, center, pad_mode,
                            fmin, fmax, htk, norm)

   # 실습 2: dB 변환 효과
   visualize_db_conversion(trumpet, sr, n_fft, hop_length, n_mels,
                           win_length, window, center, pad_mode,
                           fmin, fmax, htk, norm, power, amin, top_db)

   # 실습 3: 최종 멜 스펙트로그램
   visualize_final_melspec(trumpet, sr, n_fft, hop_length, n_mels,
                           win_length, window, center, pad_mode,
                           fmin, fmax, htk, norm, power, amin, top_db)

   print("\n" + "=" * 50)
   print("실습 4 완료")
   print("=" * 50)

if __name__ == "__main__":
   main()