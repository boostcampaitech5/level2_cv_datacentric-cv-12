# 프로젝트 개요

프로젝트 목표

- 진료비영수증에서 OCR 기술을 통해 문서의 Text를 인식한다. 영수증에 포함된 정보를 자동으로 추출하고, 텍스트 형태로 사용자에게 제공함으로써 사용자가 진료 관련 정보를 편리하게 관리하고 분석할 수 있도록 돕는다. 

Input : 진료비 영수증

Output : 글자가 존재하는 영역을 표현하는 좌표

# 수행 내용

1. EDA 및 실험 계획
2. Scheduler, Optimizer 비교 실험
3. 잘못 라벨링된 데이터 삭제 후 훈련 진행
4. Augmentation 실험
5. AI Hub 데이터 추가 실험
    1. 1차 : 100장 추가
    2. 2차 : 670장 추가
    3. 3차 : 860장 추가

# 최종 모델 및 결과

Baseline의 100장의 이미지와 추가로 annotate한 200장 총 300장으로 훈련한 EAST 모델의 결과이다. 

- Augmentation
    - resize
    - adjust_height
    - rotate
    - crop
    - color_jitter
- Optimizer, Scheduler
    - AdamW
    - CosineAnnealingLR

**public(0.9686) → private(0.9227) 최종 19등**


# 자체 평가 의견

### 잘한 점

- 대회초반에 Optimizer, Scheduler 실험을 진행하여 결과를 확인하였다.
- Inference 결과를 확인하며 잘 못 라벨링된 데이터를 제거하는 시도를 하였다.
- 팀원들과 지속적인 상황 공유 및 정보 공유를 한 것

### 아쉬운 점

- 이번 대회에서는 제약 조건이 많아 성능 개선을 위한 다양한 시도를 못한 것
- KFold 후 ensemble을 시도하지 못했다.
- redis를 통해 GPU를 관리하지 않았다.
- 추가로 다시 라벨링하는 작업을 시도해보지 않았다.

### 배운 점 및 시도해 볼 것

- OCR task의 모델을 train한 경험
- labeling tool을 통해 데이터를 직접 구축한 경험

# 팀 구성 및 역할
- 도환 : Augmentation, AI Hub 데이터 추가 실험
- 아라 : AI Hub 데이터 추가 실험, Optimizer 실험
- 무열 : AI Hub 데이터셋 탐색, 추가 실험 피드백하기
- 성운 : Optimizer, Scheduler 실험, 잘못 라벨링된 데이터 확인 후 제거
- 현민 : Augmentation 및 Preprocessing 실험

|김도환 |                                                  서아라|성무열 |                                                  조성운|한현민|
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [<img src="https://avatars.githubusercontent.com/u/121927513?v=4" alt="" style="width:100px;100px;">](https://github.com/rlaehghks5) <br/> | [<img src="https://avatars.githubusercontent.com/u/68554446?v=4" alt="" style="width:100px;100px;">](https://github.com/araseo) <br/> | [<img src="https://avatars.githubusercontent.com/u/62093939?v=4" alt="" style="width:100px;100px;">](https://github.com/noheat61) <br/> |[<img src="https://avatars.githubusercontent.com/u/126544082?v=4" alt="" style="width:100px;100px;">](https://github.com/nebulajo) <br/> | [<img src="https://avatars.githubusercontent.com/u/33598545?s=400&u=d0aaa9e96fd2fa1d0c1aa034d8e9e2c8daf96473&v=4" alt="" style="width:100px;100px;">](https://github.com/Hyunmin-H) <br/> |

****