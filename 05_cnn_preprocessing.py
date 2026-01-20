"""
실습 5: CNN 입력 후처리

학습 목표:
- CNN 입력 크기 고정의 필요성 이해
- 패딩값 선택의 중요성
- 정규화 방법과 주의사항 이해

파이프라인 요약:
1. 멜 스펙트로그램 생성 (dB 스케일)
2. 크기 고정 (crop / padding)
3. 값 정규화 (Min-Max 또는 표준화)
4. 채널 차원 추가 (CNN 입력 형태)

출력:
- outputs/05_size_comparison.png
- outputs/05_padding_wrong.png
- outputs/05_padding_correct.png
- outputs/05_channel_replication.png
- outputs/05_normalization.png
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import load_config, ensure_output_dir

def visualize_size_fixing(mel_db: np.ndarray,
   sr: int,
   hop_length: int,
   target_frames: int = 128
   ) -> None:
   """
   CNN 입력 크기 고정 시각화 (실습 1)

   왜 크기 고정이 필요한가?
   - CNN은 고정된 입력 크기 필요 (예: 128x128)
   - 오디오 길이가 다르면 프레임 수가 달라짐
   - 배치 처리를 위해 동일한 shape 필요

   크기 고정 방법:
   - Crop: 긴 오디오는 잘라냄 (정보 손실)
   - Padding: 짧은 오디오는 채움 (패딩값 중요)

   Args:
      mel_db: 멜 스펙트로그램 (dB 스케일)
      sr: 샘플링 레이트 (Hz)
      hop_length: 홉 길이
      target_frames: 목표 프레임 수 (기본값: 128)
   """
   print("\n[1] 크기 고정 (Crop / Padding)")

   n_mels, n_frames = mel_db.shape
   print(f"    원본 shape: ({n_mels}, {n_frames})")
   print(f"    목표 frames: {target_frames}")

   # Crop (프레임 수가 목표보다 많은 경우)
   if n_frames >= target_frames:
      cropped = mel_db[:, :target_frames]
      print(f"    Crop 적용: ({n_mels}, {target_frames})")
   else:
      cropped = mel_db
      print(f"    Crop 불필요 (원본이 더 짧음)")

   # Padding (프레임 수가 목표보다 적은 경우)
   pad_value = mel_db.min() # dB 도메인에서 올바른 패딩값
   if n_frames < target_frames:
      pad_width = target_frames - n_frames
      padded = np.pad(mel_db, ((0, 0), (0, pad_width)),
                     mode='constant', constant_values=pad_value)
      print(f"    Padding 적용: ({n_mels}, {target_frames})")
   else:
      padded = mel_db[:, :target_frames]
      print(f"    Padding 불필요 (원본이 더 김)")

   # 시각화 (3행 1열: 원본, Crop, Padding)
   fig, axes = plt.subplots(3, 1, figsize=(12, 10))

   # 원본
   img1 = librosa.display.specshow(
      mel_db, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', ax=axes[0], cmap='magma'
   )
   axes[0].set_title(f"Original: {mel_db.shape}")
   fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

   # Crop 결과
   img2 = librosa.display.specshow(
      cropped, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', ax=axes[1], cmap='magma'
   )
   axes[1].set_title(f"Cropped: {cropped.shape}")
   fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

   # Padding 결과
   img3 = librosa.display.specshow(
      padded, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', ax=axes[2], cmap='magma'
   )
   axes[2].set_title(f"Padded: {padded.shape} (pad_value={pad_value:.1f})")
   fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')

   plt.tight_layout()
   plt.savefig('outputs/05_size_comparison.png', dpi=150)
   plt.close()

   print(f"    저장: outputs/05_size_comparison.png")

def visualize_padding_comparison(mel_db: np.ndarray,
   sr: int,
   hop_length: int,
   target_frames: int = 300
   ) -> None:
   """
   잘못된 패딩 vs 올바른 패딩 비교 (실습 2)

   dB 도메인에서 패딩값 선택이 중요한 이유:
   - dB 스케일: 0dB = 최대값, 음수 = 작은 값
   - 0으로 패딩하면 = 최대 에너지로 채움 (잘못됨)
   - 올바른 패딩값: mel_db.min() 또는 -80 (무음)

   시각적 효과:
   - 잘못된 패딩(0): 패딩 영역이 밝게 표시 (가짜 에너지)
   - 올바른 패딩(min): 패딩 영역이 어둡게 표시 (무음)

   Args:
      mel_db: 멜 스펙트로그램 (dB 스케일)
      sr: 샘플링 레이트 (Hz)
      hop_length: 홉 길이
      target_frames: 목표 프레임 수 (원본보다 크게 설정)
   """
   print("\n[2] 패딩값 비교 (잘못된 vs 올바른)")

   n_mels, n_frames = mel_db.shape
   pad_width = target_frames - n_frames
   
   if pad_width <= 0:
      print(f"    원본({n_frames})이 목표({target_frames})보다 큼, 패딩 불필요")
      return
   
   # 잘못된 패딩 (0으로 채움)
   padded_wrong = np.pad(mel_db, ((0, 0), (0, pad_width)),
                        mode='constant', constant_values=0)
   
   # 올바른 패딩 (최소값으로 채움)
   pad_value = mel_db.min()
   padded_correct = np.pad(mel_db, ((0, 0), (0, pad_width)),
                        mode='constant', constant_values=pad_value)
   
   print(f"    원본 shape: ({n_mels}, {n_frames})")
   print(f"    패딩 후 shape: ({n_mels}, {target_frames})")
   print(f"    잘못된 패딩값: 0 (dB에서 0 = 최대 에너지)")
   print(f"    올바른 패딩값: {pad_value:.1f} (무음)")

   # 잘못된 패딩 시각화
   plt.figure(figsize=(12, 5))
   img = librosa.display.specshow(
      padded_wrong, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', cmap='magma'
   )
   plt.colorbar(img, format='%+2.0f dB')
   plt.title('Wrong Padding (value=0): Bright artifacts on the right')
   plt.axvline(x=n_frames * hop_length / sr, color='white',
               linestyle='--', linewidth=2, label='Original end')
   plt.legend()
   plt.tight_layout()
   plt.savefig('outputs/05_padding_wrong.png', dpi=150)
   plt.close()
   print(f"    저장: outputs/05_padding_wrong.png")

   # 올바른 패딩 시각화
   plt.figure(figsize=(12, 5))
   img = librosa.display.specshow(
      padded_correct, sr=sr, hop_length=hop_length,
      x_axis='time', y_axis='mel', cmap='magma'
   )
   plt.colorbar(img, format='%+2.0f dB')
   plt.title(f"Correct Padding (value={pad_value:.1f}): Silent region")
   plt.axvline(x=n_frames * hop_length / sr, color='white',
            linestyle='--', linewidth=2, label='Original end')
   plt.legend()
   plt.tight_layout()
   plt.savefig('outputs/05_padding_correct.png', dpi=150)
   plt.close()
   print(f"    저장: outputs/05_padding_correct.png")

def visualize_channel_replication(mel_db: np.ndarray,
   n_mels: int
   ) -> None:
   """
   채널 복제 시각화 (실습 3)

   왜 채널 복제가 필요한가?
   - 멜 스펙트로그램: 1채널 (흑백 이미지와 유사)
   - ImageNet 사전 학습 모델: 3채널 (RGB) 입력 기대
   - 전이학습 활용을 위해 채널 복제 필요

   채널 복제 방법:
   - 단순 복제: 동일 데이터를 3번 쌓기
   - np.stack([mel_db, mel_db, mel_db], axis=0) 또는
   - np.repeat(mel_db[np.newaxis, ...], 3, axis=0)

   shape 변화:
   - 원본: (n_mels, frames) = (128, 230)
   - 복제 후: (3, n_mels, frames) = (3, 128, 230)

   Args:
      mel_db: 멜 스펙트로그램 (dB 스케일)
      n_mels: 멜 필터 개수
   """
   print("\n[3] 채널 복제 (1ch -> 3ch)")

   print(f"    원본 shape: {mel_db.shape}")

   # 채널 복제 (3채널)
   mel_3ch = np.stack([mel_db, mel_db, mel_db], axis=0)
   print(f"    복제 후 shape: {mel_3ch.shape}")

   # 시각화 (1행 3열: R, G, B 채널)
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   channel_names = ['Channel 0 (R)', 'Channel 1 (G)', 'Channel 2 (B)']
   for i, (ax, name) in enumerate(zip(axes, channel_names)):
      im = ax.imshow(mel_3ch[i], aspect='auto', origin='lower', cmap='magma')
      ax.set_title(name)
      ax.set_xlabel('Frame')
      ax.set_ylabel('Mel bin')
      fig.colorbar(im, ax=ax, format='%+2.0f dB')

   plt.suptitle('Channel Replication for Pretrained CNN (All channels identical)')
   plt.tight_layout()
   plt.savefig('outputs/05_channel_replication.png', dpi=150)
   plt.close()

   print(f"    CNN 입력 형태: (batch, 3, {n_mels}, frames)")
   print(f"    저장: outputs/05_channel_replication.png")

def visualize_normalization(mel_db: np.ndarray) -> None:
   """
   정규화 방법 비교 시각화 (실습 4)

   왜 정규화가 필요한가?
   - 신경망은 입력 범위에 민감
   - dB 값 범위: -80 ~ 0 (데이터마다 다름)
   - 일관된 범위로 변환하여 학습 안정성 향상

   정규화 방법:
   1. Min-Max: (x - min) / (max - min) -> 0~1 범위
   2. 표준화: (x - mean) / std -> 평균 0, 분산 1

   주의사항:
   - 학습 데이터의 통계(min, max, mean, std) 저장 필요
   - 테스트 시 동일한 통계 사용
   - 배치 정규화와는 다름 (모델 내부 vs 전처리)

   Args:
      mel_db: 멜 스펙트로그램 (dB 스케일)
   """
   print("\n[4] 정규화 방법 비교")

   # 원본 통계
   print(f"    원본 범위: {mel_db.min():.1f} ~ {mel_db.max():.1f}")
   print(f"    원본 평균: {mel_db.mean():.2f}, 표준편차: {mel_db.std():.2f}")

   # Min-Max 정규화 (0~1)
   mel_minmax = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
   print(f"    Min-Max 범위: {mel_minmax.min():.2f} ~ {mel_minmax.max():.2f}")

   # 표준화 (z-score)
   mel_zscore = (mel_db - mel_db.mean()) / mel_db.std()
   print(f"    표준화 범위: {mel_zscore.min():.2f} ~ {mel_zscore.max():.2f}")
   print(f"    표준화 평균: {mel_zscore.mean():.2f}, 표준편차: {mel_zscore.std():.2f}")

   # 시각화 (1행 3열)
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   # 원본 (dB)
   im1 = axes[0].imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
   axes[0].set_title(f"Origin (dB)\nRange: [{mel_db.min():.0f}, {mel_db.max():.0f}]")
   axes[0].set_xlabel('Frame')
   axes[0].set_ylabel('Mel bin')
   fig.colorbar(im1, ax=axes[0], format='%+2.0f')

   # Min-Max
   im2 = axes[1].imshow(mel_minmax, aspect='auto', origin='lower', cmap='magma')
   axes[1].set_title('Min-Max normalization\nRange: [0, 1]')
   axes[1].set_xlabel('Frame')
   axes[1].set_ylabel('Mel bin')
   fig.colorbar(im2, ax=axes[1], format='%.2f')

   # 표준화
   im3 = axes[2].imshow(mel_zscore, aspect='auto', origin='lower', cmap='magma')
   axes[2].set_title(f'Z-Score Standardization\nRange: [{mel_zscore.min():.1f}, {mel_zscore.max():.1f}]')
   axes[2].set_xlabel('Frame')
   axes[2].set_ylabel('Mel bin')
   fig.colorbar(im3, ax=axes[2], format='%.1f')
   
   plt.tight_layout()
   plt.savefig('outputs/05_normalization.png', dpi=150)
   plt.close()

   print(f"    저장: outputs/05_normalization.png")

def main():
   """
   메인 실행 - 각 실습을 순차 실행
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

   ensure_output_dir()

   print("=" * 50)
   print("실습 5: CNN 입력 후처리")
   print("=" * 50)

   # 트럼펫 샘플 로드 및 멜 스펙트로그램 생성
   trumpet, _ = librosa.load(librosa.ex('trumpet'), sr=sr)
   mel_spec = librosa.feature.melspectrogram(
      y=trumpet, sr=sr, n_fft=n_fft, hop_length=hop_length,
      win_length=win_length, window=window, center=center, pad_mode=pad_mode,
      power=power, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm
   )
   mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=amin, top_db=top_db)

   # 실습 1: 크기 고정
   visualize_size_fixing(mel_db, sr, hop_length)

   # 실습 2: 패딩값 비교
   visualize_padding_comparison(mel_db, sr, hop_length)

   # 실습 3: 채널 복제
   visualize_channel_replication(mel_db, n_mels)

   # 실습 4: 정규화
   visualize_normalization(mel_db)

   print("\n" + "=" * 50)
   print("실습 5 완료")
   print("=" * 50)

if __name__ == "__main__":
   main()
