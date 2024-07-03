# 우당탕탕 주니어 탈출

📢 2024년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개

도심 내 자율주행에 필수적인 Sementic Segmentation을 위한 모델을 구현합니다.
![1003](https://github.com/andless2004/aiku-24-1-juniors_wild_escape/assets/129763673/acb7ae89-964f-4b57-adcb-35c0cbc5fd22)

Object Detection과 Segmentation에 관한 논문을 공부하고 이를 재구현하며 이해도를 높이고자 진행했습니다.
또한, 최종적으로 AIKU에서 진행한 AIKUTHON; Semantic Segmentation for Self-Driving Car를 위한 모델을 구현하는 프로젝트입니다.


## 방법론
### 문제 정의
![image](https://github.com/andless2004/aiku-24-1-juniors_wild_escape/assets/129763673/afbe7030-a6f4-4d1d-9dd8-9bef91095c1f)
차량 관점에서 사진이 주어졌을 때 이를 13개의 label(사람, 인도, 차도, 나무, 벽, 신호등 등)으로 구분하는 semantic segmetation model을 구현해야 합니다.

### U-Net
- U-Net 논문 리뷰 이후 프로젝트 팀원들과 피드백을 주고 받았습니다.
- AIKUTHON의 RLE encoding data를 사용하는 U-Net을 구현했습니다.

### Faster R-CNN
- Faster R-CNN 논문 리뷰 이후 프로젝트 팀원들과 피드백을 주고 받았습니다.
- R-CNN은 Object Detection task이기에 AIKUTHON data는 부적합합니다.
- 때문에, Kaggle의 'Cityscapes Image Pairs' 데이터셋을 활용하여 구현했습니다.

### SegFormer
- SegFormer 논문 리뷰 이후 프로젝트 팀원들과 피드백을 주고 받았습니다.
- 일반적인 상황에서 U-Net과 SegFormer의 성능을 비교하고, SegFormer의 성능을 높이기 위한 Hyperparameter tuning과 Data Augmentation 등의 기법을 활용했니다.

## 환경 설정
```
pip install segmentation_models_pytorch
pip install -U git+https://github.com/huggingface/transformers.git
pip install -U git+https://github.com/huggingface/accelerate.git
```
segformer를 위해 위 설치가 필요합니다.
추가로, oneformer를 사용하기 위해 ```natten```의 설치가 필요하며 아래와 같은 방식을 추천합니다.
```
pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
```

## 사용 방법

drive_dir을 지정한 후 학습을 진행할 수 있습니다.

## 예시 결과

<img width="256" alt="test" src="https://github.com/andless2004/aiku-24-1-juniors_wild_escape/assets/129763673/2607b8d1-c594-444f-9c44-320fa3b394d6">

Cityscapes Image Pairs Data를 통해 학습한 model의 결과.
대체로 잘 나타내나 일부 작은 사물; 얇은 표지판 기둥, 하늘의 신호등 등을 잘 잡아내지는 못하여 성능 개선이 필요합니다.

## 팀원
- [김승주]: 논문 리서치, 코드 작성
- [구영서]: 논문 리서치, 코드 작성
- [박경빈]: 논문 리서치, 코드 작성
- [윤혜원]: 논문 리서치, 코드 작성
