# 목차

- [**Instance Segmentation**](#1-instance-segmentation)
  - Mask R-CNN
  - YOLOACT
  - YolactEdge
- [**Panoptic Segmentation**](#2-panoptic-segmentation)
  - UPSNet
  - VPSNet
- [**Landmark localization**](#3-landmark-localization)
  - Hourglass
  - Densepose
  - RetinaFace

# 1. Instance Segmentation

`instance segmentation` : semantic segmentation + instance distinguish

#### 1-1. Mask R-CNN

![image-20220314200500058](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220314200500058.png)

- **Mask R-CNN은 대표적인 Instance Segmentation network 중 하나**
- **R-CNN 계열은 모두 2 stage detector**

- **Faster R-CNN와 차이점**

  - **ROI Align**

    기존 Faster R-CNN은 ROI Pooling 단계에서 정수로 밖에 픽셀 표현이 안됐지만, Mask R-CNN은 소수점 픽셀 표현이 가능 -> 정교해졌다.

  - **Mask branch**

    각 class에 대해서 binary mask를 생성하고, classification을 통해 구한 정답을 반영하여 mask를 선택한다. 

    위 그림에서 파란색 box가 mask branch를 의미한다. 

#### 1-2. YOLOACT

![논문 리뷰 - YOLACT: Real-time Instance Segmentation](https://raw.githubusercontent.com/byeongjokim/byeongjokim.github.io/master/assets/images/YOLACT/architecture.PNG)

- **single stage detector**

- **클래스 개수만큼 mask를 생성하는 것이 아닌, prototype 생성**

  prototype을 Mask Coefficient를 이용한 선형결합으로 조합하여 최종 mask 선택

  > 선형대수의 개념으로 보자면, prototype은 span 가능한 mask의 basis를 구한다고 생각할 수 있다. 

#### 1-3. YolactEdge

![image-20220314200950966](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220314200950966.png)

- **YOLOACT보다 소형화된 버전 (영상 이미지에서 활용 가능)**
- **이전 frame의 정보를 현재 frame에 적용하여 계산량 감소 (빨간 박스 부분)**



# 2. Panoptic segmentation

`panoptic segmentation` : instance segmentation + semantic segmentation

즉, **배경과 instance 모두를 각각 분리**해낸다. 

#### 2-1. UPSNet (Unified Panoptic Segmentation Network)

![UPSNet: A Unified Panoptic Segmentation Network – arXiv Vanity](https://media.arxiv-vanity.com/render-output/4973376/x1.png)

- **FPN** : 고해상도의 feature 추출

  **Semantic Head** : instance+stuff 구분

  **Instance Head** : Semantic Head로부터의 instance Box정보와 Instance Head로부터 Intance segmentation 정보를 합하여 mask logit 구성

  (이때, instance를 localize하기 위해, Yi 이미지를 resize/zero padding한다.)

  ![Panoptic Segmentation with UPSNet | by Vaishak V.Kumar | Towards Data  Science](https://miro.medium.com/max/1400/1*Z91vunswqJ-Yzc53wB9NyA.png) 

- **unknown class에 대한 정보도 포함**

#### 2-2. VPSNet (영상 이미지를 위한 network)

![image-20220314205428875](../../../../AppData/Roaming/Typora/typora-user-images/image-20220314205428875.png)

- `Motion map` : 이전 frame과 현재 frame에서의 같은 지점의 pixel의 움직임을 나타내는 map

- **motion map + 현재 feature map + 이전 feature map이 합쳐져서 target feature map 구성**

- **이전 feature map에서의 ROI와 현재 feature map에서의 ROI의 연관성을 구함**

  (tracking처럼 구현되어, 깜빡임이나 색깔 변화가 덜하다.)

- **head 부분은 UPSNet과 동일**



# 3. Landmark Localization

`Landmark Localization` : keypoint의 좌표를 예측하는 것을 의미

> 예를 들어 사람의 얼굴에서 눈썹, 코, 입 예측
>
> 또는 pose estimation 등이 있다. 

- **Coordinate regression v.s Heatmap classification**

  ![image-20220314210043517](../../../../AppData/Roaming/Typora/typora-user-images/image-20220314210043517.png)

  | 종류                        | 계산량           | 정확도           | 편향성 |
  | --------------------------- | ---------------- | ---------------- | ------ |
  | Coordinate<br />regression  | 상대적으로 적다. | 상대적으로 낮다. | biased |
  | Heatmap<br />classification | 상대적으로 많다. | 상대적으로 높다. | -      |

#### 3-1. Hourglass Network

여러개의 UNet과 비슷한 구조들을 여러개 쌓은 것과 같은 구조가 특징이다. 

![image-20220314210406604](../../../../AppData/Roaming/Typora/typora-user-images/image-20220314210406604.png)

- **bottleneck 구조로 receptive field를 넓힐 수 있다.** 
- **skip connection을 통해 low level feature까지 사용이 가능하여 정확도를 높일 수 있다.** 
- **UNet과의 차이점**
  - concatenate이 아닌 +
  - skip connection에 conv layer가 존재한다. 

#### 3-2. DensePose

UV map 표현법으로 3D를 표현한다. 

![image-20220314210725129](../../../../AppData/Roaming/Typora/typora-user-images/image-20220314210725129.png)

- **DensePose R-CNN = Faster R-CNN + 3D surface regression branch**

  각 body part에 대해서 segmentation map을 구한다. 



#### 3-3. RetinaFace

하나의 task가 아닌 multi task를 수행

![image-20220314211202521](../../../../AppData/Roaming/Typora/typora-user-images/image-20220314211202521.png)

- **RetinaFace = FPN + Multi-task branches**
  - classification
  - bounding box regression
  - 5 point regression (얼굴의 5 point)
  - mesh regression (3D)

- **Multi task를 통해 backbone을 강인하게 학습**

  - 여러개의 task로부터의 gradient 역전파를 통해 더 많은 정보를 전달
  - 더 많은 정보는 더 많은 이미지를 다루는 것과 같은 효과

  