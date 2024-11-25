# Boxing DQN

이 프로젝트는 Atari Boxing 게임을 Deep Q-Network (DQN)를 사용하여 학습하는 강화학습 구현입니다.

## 설치 방법

필요한 패키지를 설치하기 위해 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

## 실행 방법

다음 명령어로 학습을 시작할 수 있습니다:

```bash
python boxing_dqn.py
```

## 주요 특징

- DQN 알고리즘 구현
- Frame stacking (4 프레임)
- Experience replay
- Target network
- ε-greedy 탐험 정책
- 자동 모델 저장 (100 에피소드마다)

## 구현 세부사항

- 입력: 84x84 그레이스케일 이미지
- 프레임 스태킹: 4개의 연속된 프레임
- 컨볼루션 신경망 구조
- Adam 옵티마이저 사용
- 리플레이 메모리 크기: 100,000
- 배치 크기: 32
- 감가율(γ): 0.99
- ε-greedy 파라미터:
  - 시작 ε: 1.0
  - 최소 ε: 0.1
  - 감소율: 0.995
