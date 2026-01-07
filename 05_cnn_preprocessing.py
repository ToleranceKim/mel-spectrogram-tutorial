"""
실습 5: CNN 입력 후처리

블로그 섹션: 5. CNN 입력용 후처리

학습 목표:
- CNN 입력 크기 고정의 필요성 이해
- 패딩값 선택의 중요성 (특히 dB 도메인)
- 정규화 방법과 주의사항 이해

실습 내용:
1. 크기 고정 (crop / padding)
2. 패딩값 선택
   - dB 도메인에서 0 패딩 금지!
   - 올바른 패딩값: -top_db 또는 mel_db.min()
3. 패딩 왜곡 시각화
4. 채널 복제 (1채널 → 3채널)
5. 값 정규화 (Min-Max, 표준화)
6. center=True 이슈

출력:
- outputs/05_size_comparison.png
- outputs/05_padding_wrong.png
- outputs/05_padding_correct.png
- outputs/05_channel_replication.png
- outputs/05_normalization.png
"""

# TODO: 구현 예정
