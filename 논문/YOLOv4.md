# 목차

- [**Abstract**](#1-abstract)
- [**Introduction**](#2-introduction)
- [**Related work**](#3-related-work)
- [**Bag of Freebies**](#4-bag-of-freebies)
- [**Bag of Specials**](#5-bag-of-specials)
- [**Methodology**](#6-methodology)
- [**YOLOv4 최종 정리**](#7-yolov4-최종-정리)
- [**Experiment**](#8-experiment)
- [**Results**](#9-results)
- [**Appendix**](#10-appendix)



# 1. Abstract
<details>
<summary>접기/펼치기</summary>
<div markdown="1">  

#### 1-1. 한 줄 요약

YOLOv4는 기존의 BoS + modified BoF를 적용하여, 단일 GPU에서도 잘 돌아가는 빠르고 정확한 object detector를 만들었다.

그리고 [여기](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=dnjswns2280&logNo=222043364404)가 진짜 깔끔하게 정리했으니, 나중에 다시 볼 땐 이걸 보자.

#### 1-2. 개요 


CNN의 정확도를 향상시키기 위한 방법 (feature)들은 많고, 이 feature들은 이용하는데는 large dataset에서의 실험과 결과의 이론적인 증명이 필요하다고 한다. 

> 이 논문에서는 이론적인 증명과 실험이 매우 잘 되어 있다. 

몇몇 feature들은 적은 데이터셋, 특정한 모델, 특정한 문제에만 맞춰줘 있는 것인 반면, 

**batch-normalization**이나 **residual connection**과 같은 feature들은 **대부분의 model, task, dataset에 적용 가능**하다. 

> 추가적으로, 이런 feature들로는 
>
> weighted residual connections (WRC), cross stage partial connection (CSP), 
>
> cross mini batch normalization (CmBN), self adversarial training (SAT), 
>
> mish activation 등이 있다. 

**YOLOv4에서**는 새로운 feature인 **WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, dropblock regularization, CIoU loss**를 사용하고, 이들 중 몇개를 결합해서 SOTA 결과를 도출했다. 

> Tesla V100 환경에서 MS COCO dataset에 대해서 43.5% AP (65.7% AP_50)와 ~65 FPS의 성능을 이끌었다. 

[github link](https://github.com/AlexeyAB/darknet)

</div>
</details>

# 2. Introduction
<details>
<summary>접기/펼치기</summary>
<div markdown="1">  

  대부분의 **CNN기반 object detector**는 **주어진 특정 상황**에서만 사용이 된다. 

> 예를 들어, car collision은 빠르지만 덜 정확한 model을 사용하며, free parking space 찾는 것은 느리지만 정확한 model을 사용한다.

real time object detecter의 정확도를 높이는 것은 이러한 제약 사항 뿐 아니라, 인간의 개입을 줄이는 상황 (human input reduction인데, 잘못된 결과에 대한 처리라고 생각해도 되지 않을까?)에서도 잘 동작한다. 

**기존 realtime 동작**에는 **큰 mini-batch size**와 **많은 수의 GPU**가 필요했다. 

**YOLOv4는 하나의 real time conventional GPU에서 동작하는 CNN을 만들면서 위 문제들을 해결**했다. 

- **기여1.** 효율적이고 빠른 객체 탐지 모델 개발( 1080 Ti, 2080 Ti GPU에서 잘 돌아감)

- **기여2.** SOTA bag-of-freebies, bag of special 방법을 사용하며, 그들의 영향을 입증했다.

- **기여3.** SOTA 방법을 좀 수정하여, 싱글 GPU에 적합하게 만듬 (CBN, PAN, SAM 등)

![image](https://user-images.githubusercontent.com/71866756/154497635-5eae2c06-287d-48c0-8799-00aa69a6316e.png)

</div>
</details>

# 3. Related work
<details>
<summary>접기/펼치기</summary>
<div markdown="1">  
    
![image](https://user-images.githubusercontent.com/71866756/154618290-99ab2cbf-f807-46ad-a763-13b8cff85fae.png)

    
**기존 object detection의 구조**는 두 파트로 이루어져 있다. 

 **Imagenet pretrained backbone + head** (class, bbox예측) 이다. 

#### 3-1. backbone

| device | detector (backbone model)         |
| ------ | --------------------------------- |
| GPU    | VGG, ResNet, ResNext, DenseNet    |
| CPU    | SqueezeNet, MobileNet, ShuffleNet |

#### 3-2. head

| detector 종류      | option      | model                                                        |
| ------------------ | ----------- | ------------------------------------------------------------ |
| one-stage detector | None        | YOLO, SSD, RetinaNet                                         |
| two-stage detector | None        | R-CNN 시리즈, fast R-CNN, faster R-CNN, <br />R-FCN, Libra R-CNN |
|                    | anchor free | RedPoints                                                    |

> RedPoints : deformable convolution을 활용하여 객체의 둘레에 점을 찍어 얻은 reppoints를 기반으로 anchor 없이 객체 탐지를 수행하는 모델 

- 최근에는 backbone과 head사이에 layer를 추가해서 different stages에서 feature map을 모으기도 함

  > neck of an object detector라고 부름 - 여러개의 bottom-up path & top-down path로 구성
  >
  >  Feature Pyramid Network (FPN), Path Aggregation Network (PAN), BiFPN, and NAS-FPN 등이 있다. 

#### 3-3. 위의 모델들 제외한 나머지

- 새로운 backbone을 이용 : DetNet, DetNAS

- 아예 새로운 모델 : SpineNet, HitDetector

#### 요약

![image](https://user-images.githubusercontent.com/71866756/154497672-41378412-3276-4ccb-92a8-2194a8a1851d.png)



</div>
</details>

# 4. Bag of freebies
<details>
<summary>접기/펼치기</summary>
<div markdown="1">  

training기법을 바꾸는 등의 방법을 통해 accuracy를 증가시키는 방법을 의미한다. 

> training cost만 늘리고, inference cost는 그대로 유지

#### 4-1. Pixel-wise adjustments

- **Data Augmentation**

  input 이미지의 **variability 증가** -> 다른 **환경**에서 얻어진 이미지들에 대해 **강건함 증가**

  - **photometric distortion**

    brightness, contrast, hue, saturation, noise 조정

  - **geometric distortions**

    random scaling, cropping, flipping, rotating

  - **그 외**

    `random erase, cutout` :  랜덤하게 사각형 영역으로 골라내고, random또는 0으로 채우는 것

    `hide-and-seek, grid mask` : 램덤 또는 균등하게 다양한 사각형 영역을 골라내고 0으로 채움

    `DropOut, Dropconnect, DropBlock` : feature map 단위에서 진행

    `CutMix, MixUp` : 여러 이미지 사용하여 섞는 것

    `style transfer GAN` : CNN에 의해 학습된 texture bias를 효과적으로 줄일 수 있음

    > **texture bias**란  CNN을 이용한 이미지 물체 인식에 있어서 물체의 형상으로 물체를 인식하는 것이 아닌 **물체의 texture를 이용**하여 **물체를 인식**하게되는 경향을 의미를 하며, 물체의 **texture가 바뀌면 제대로 인식하지 못하는 점**에 대해서 설명하고 있습니다.



#### 4-2. label adjustment

- **semantic distribution bias**란 **클래스 간 데이터 불균형**을 의미한다. 

  - **two-stage object detector에서의 해결 방법**

    negative example mining 또는 hard example mining 

    > one-stage detector들은 dense prediction architecture에 속하기 때문 위 방법 사용 불가 

  - **one-stage object detector에서의 해결 방법**

    focal loss

    


- **one-hot으로 인한 카테고리간 상관도 확인 불가 문제**

  label smoothing을 통해 모델을 더 강건하게 만든다. 

  > knowledge distillation을 통해 더 좋은 soft label 생성 가능



#### 4-3. Objective Function of BBox Regression

**전통적인 object detector**는 regression을 직접적으로 수행하기 위해 **bbox의 중앙 좌표, 높이, 너비** (또는 **좌상단, 우하단**) 에 대해 **MSE를 사용**했으며, **anchor-based method**에 대해서는 **offset을 추정**하기 위해 사용했다. 

하지만, 직접적으로 좌표값을 추정하는 것은 이 좌표들을 **독립적인 변수로 보는 것과 마찬가지**이다. 

최근에는 **예측 bbox 영역, ground truth bbox영역**을 고려하는 **IoU loss**를 제안했다. 

IoU의 장점으로는 좌표와 달리 **scale invariant**하다는 것이다. 

> 최근에는 또 IoU 방법에 대해 발전시키고 있다. (GIoU, DIoU, CIoU등)
>
> GIoU : 예측 BBox, ground truth BBox를 포함하는 최소의 사각형을 이용하는 것
>
> DIoU : GIoU에 객체의 중앙의 거리를 추가적으로 고려
>
> CIoU : overlapping 영역, center point 거리, aspect ratio 동시에 고려하는 것 (DIoU에 aspect ratio를 추가한 것)
>
> (CIoU는 BBox regression (박스 위치를 교정해주는 것) 문제에 최고의 속도와 정확도를 달성시킴)

</div>
</details>

# 5. Bag of specials
<details>
<summary>접기/펼치기</summary>
<div markdown="1">  

약간의 inference cost 증가로 accuracy를 증가시키는 post-processing, plugin modules 방법들을 일컫는 말

일반적으로, 이런 **plugin modules**는 **receptive field를 늘리거나, attention mechanism을 도입**하거나, **feature integration capability를 강화**하는 등, **모델의 특정 속성을 강화**하는 것이다. 

> **feature integration theory**란 인간의 눈이 단순히 모든 픽셀의 색을 입력받아 물체를 구분하는 것이 아니라, 색, 방향, 곡률, 크기 등 **세부적인 특징**을 받는 세포가 있으며, 이를 **통합하여 물체를 인식**한다는 이론이다. 
>
> neural network도 인간의 뉴런을 본따서 만든 것이므로, 이런 표현을 쓰는 것 같다.



## 5-1. Plugin modules

#### 5-1-1. Increase Receptive Field

- **Spatial Pyramid Pooling (SPP)**

  Spatial Pyramid Matching (SPM)에서 유래된 기법으로 bag of words 대신 max pooling을 사용했다. 

  >SPP는 1차원을 출력하기에, fully convolutional network(FCN)을 적용못해서, yolov3에서는 kxk (k=1,5,9,13), stride 1의 max pooling output을 concat해서 사용했다. 
  >
  >![image-20220217211129203](https://user-images.githubusercontent.com/71866756/154501684-d9a901f1-d8fd-4da0-a286-af5818eb6e44.png)
  >
  >yolov3-608 (improved SPP)은  오직 0.5%의 계산량을 추가하여 MS COCO에서 AP_50dmf 2.7%의 이득을 보았다. 

- **Atrous Spatial Pyramid Pooling (ASPP)**

  imporved SPP와 다르게, **3x3 커널을 dilated ratio=k**로 만든 것

  ![image](https://user-images.githubusercontent.com/71866756/154497762-6337e7de-c46b-46ad-ac5f-d9c512a93596.png)

  > 이렇게 dilated convolution을 사용하면 더 넓은 receptive field를 가진다는 장점이 있다. 
  >
  > 하지만, rate가 커질 수록 유효한 weight가 줄어든다.
  >
  > (zero padding 영역으로 인해서) 

- **Receptive Field Block (RFB)**

  RFB는 **dilated ratio=k, stride=1 인 k x k인 커널**을 사용한다. 

  > ASPP보다 더 넓은 spatial coverage를 가진다. 
  >
  > RFB는 MS COCO에서 SSD (GPU얘기하는 거인듯?) AP_50을 5.7% 증가시켰다. 



#### 5-1-2. Attention Module

- **Channel Wise Attention**

  **SE module**은 ResNet50의 성능을 imagenet top-1을 1% 높였다. (computational effort는 단 2%증가)->GPU에서 추론 시간이 10% 증가해서 mobile devices에 적합하다.

- **point wise attention** 

  **Spatial Attention Module (SAM)**은 오직 0.1%의 추가 계산이 필요하고, ResNet50의 성능을 imagenet 에서 top-1을 0.5% 높였다. 그래서 추론 시간을 늘리지 않는다.



#### 5-1-3. Feature Integration

- **skip connection**
- **hyper-column** : low level physical feature를 high level semantic feature로 통합

FPN과 같은 multi-scale prediction method가 유명해져서 다른 feature pyramid를 합치는 lightweight module들이 제안되었음

> lightweight module들의 예시는 아래와 같다. 
>
> **SFAM** : channel wise level re-weighting을 multi scale concatenated feature map에 사용하기 위해서SE 모듈을 사용
>
> ![image](https://user-images.githubusercontent.com/71866756/154497947-023c6286-a159-4817-b9a5-6c8b2338ce27.png)
>
> **Adaptively Spatial Feature Fusion (ASFF)** : softmax를 point wise level re weighting으로 사용한 다음, 서로 다른 scale의 feature map을 더하는 것
>
> ![image-20220217215902681](https://user-images.githubusercontent.com/71866756/154501641-ff1215b5-bb35-4464-a05a-a46c6b6c0d21.png)
>
> **Bi-directional Feature Pyramid Networks(BiFPN)** : scale wise level re weighting을 위해 multi input weighted residual connections을 제안하였다. 그리고 서로 다른 scale의 feature map을 더한 것
>
> ![image-20220217220108695](https://user-images.githubusercontent.com/71866756/154501612-941171fc-8e5f-4f84-bb7f-c2df871fbf4b.png)



#### 5-1-4. Good Activation Function

좋은 activation function은 gradient를 효과적으로 propagate하고, 너무 많은 추가적인 연산량이 들지 않도록 한다.  

- **tanh, sigmoid** : gradient vanishing 문제 발생
- **ReLU** : gradient가 0이 될 수 있어, Dying ReLU 문제 발생

- **LReLU, PReLU, ReLU6, Scaled Exponential Linear Unit(SELU), Swish, hard-Swish, Mish** : gradient vanishing 해결하기 위해 등장

  > LReLU와 PReLU는 ReLU에서 기울기가 0이 되는 걸 막기 위한 목적임.
  >
  > ReLU6, hard-swish : quantization network를 위한 것
  >
  > SELU : neural network를 self-normalizing하는 목적으로 나옴
  >
  > Swish, Mish : continuously differentiable activation function (미분한 꼴이 input을 더한 꼴이 되어서 연속해서 미분 가능한듯)
  >
  > [위 활성함수 설명 링크](https://yeomko.tistory.com/39)



## 5-2. Post processing

일반적으로 Non-Maximum-Suppression (NMS) 방법을 사용한다. 

> NMS란 같은 객체에 대해 나쁘게 예측한 Bbox를 filtering하고, 높은 반응의 bbox를 후보로 유지하는 방법이다. 

- 원래 NMS는 context 정보를 고려하지 않았다. 그래서 Girshick이 R-CNN에서 classification confidence score를 넣었음 -> confidence score의 순서(값이 큰지 작은지)로 greedy NMS가 score의 내림차순으로 수행됨
- **soft NMS** : object의 폐색(occlusion) 때문에 greedy NMS에서confidence score가 IoU score와 함께 degradation되는 문제를 고려했음
- **DIoU NMS** : soft NMS의 기초를 두고, BBox screening process에 center point 거리 정보를 더하는 것

- 위에서 언급한 post processing방법이 capture된 이미지의 feature에 직접적으로 영향을 끼치지 않아서, anchor free method에서는 더이상 필요하지 않게 됨

  (anchor base method는 여러개의 anchor box를 미리 뽑아놓고, 거기에 정답을 맞추는 방식이라 filtering이 중요하다. 

  하지만, anchor free 방식은 객체의 중앙이나 keypoint들을 바로 예측하기 때문에 filtering이 필요없다. )


</div>
</details>

# 6. Methodology  
<details>
<summary>접기/펼치기</summary>
<div markdown="1">  

- GPU에서 convolutional layer에서 group의 수가 작은(1-8) CSPResNeXt50 / CSPDarknet53 사용

- VPU에서 grouped-convolution을 썻지만 SE block을 사용하는 것을 삼갔다. (EfficientNet-lite / MixNet [76] / GhostNet [21] / MobileNetV3 이런 모델들)

#### 6-1. Selection of architecture

- **목적1** : input network resolution, conv layer 수, parameter 수(filter size^2 * filters * channel/groups), filter의 수의 최적의 balance의 구조를 선택

  > 예를 들어, 
  >
  > **object classification** on the ILSVRC2012 (ImageNet) dataset
  > 
  > -> CSPResNext50 >  CSPDarknet53 (성능)
  > 
  > **object detection** on the MS COCO dataset
  > 
  > -> CSPResNext50 <  CSPDarknet53 (성능)
  
- **목적2** : receptive field를 넓히기 위한 additional block 선택, 서로 다른 detector level을 위한, 서로 다른 backbone level에서 최고의 parameter aggregation 방법 찾기

  > FPN, PAN, ASFF, BiFPN 등
  
  Classification task에서 좋은 성능이 나온 model이라도 detector에서는 좋지 못한 성능이 나올 수 있다.  Classifier와 다르게 detector는 아래 사항을 요구한다.
  
  - **높은 input size** : 작은 객체를 탐지하기 위해
  - **많은 layer** : input이 커진 것을 cover하기 위한 더 큰 receptive field를 위해
  - **많은 parameter** : 한 이미지에서 서로 다른 사이즈의 다양한 객체를 탐지하는 능력을 키우기 위해

이론적으로 큰 receptive field (더 많은 수의 3x3 conv), 많은 파라미터를 가지는 모델을 선택하는게 맞다. 

![image-20220216223552683](https://user-images.githubusercontent.com/71866756/154501536-dccc37d7-da05-4c2c-8ce5-dc79f7690433.png)

위의 결과에서 **CSPDarknet53**이 이론적으로 그리고 실험적으로도 **가장 detector에 적합**했다.

> 서로 다른 크기의 receptive field의 영향은 아래와 같음
>
> 1. object size만큼의 크기 : 전체 객체를 볼 수 있다.
> 2. network size만큼의 크기 : 객체 주변의 문맥을 파악할 수 있다. 
> 3. network size보다 큰 크기 : image point와 final activation 사이 connection의 수를 늘린다.

논문에서는 backbone으로 **SPP block을 CSPDarknet53**에 붙여서 사용하였다. 

> SPP block은 receptive field를 늘리고, 중요한 문맥적 특성을 파악해내고, network operation speed를 거의 줄이지 않기 때문에 

PANet을 path-aggregation neck으로 사용하였다. 

> 서로 다른 detector level에 대해 (서로 다른 backbone level에서의 parameter aggregation)방법으로 사용함. 
>
> yolov3에서는 FPN을 사용하였다. 

**[결론]**

정리하자면  YOLOv4는 **SPP가 추가된 CSPDarknet53 backbone**, **PANet path-aggregation neck**, and **YOLOv3 (anchor based) head** 로 구성하였다.  

> 그리고 미래에는 bag of freebies를 각종 문제를 해결하고 detector accuracy를 늘리기 위해서 더 사용할 예정이다. 
>
> 추가로, 논문에서는 Cross-GPU batch normaliztion(CGBN 또는 SyncBN) 또는 비싼 장비를 사용하지 않았음
>
> (이 SOTA 결과를 GTX 1080Ti나 RTX 2080Ti와 같은 전통적인 GPU에서 사용할 수 있게 하기 위해)



#### 6-2. Selection of BoF and BoS

객체 탐지 training의 성능을 올리기 위해 CNN은 보통 아래것들을 사용한다. 

- **Activations:** ReLU, leaky-ReLU, parametric-ReLU, ReLU6, SELU, Swish, or Mish
- **Bounding box regression loss**: MSE, IoU, GIoU, CIoU, DIoU
- **Data augmentation**: CutOut, MixUp, CutMix 
- **Regularization method:** DropOut, DropPath, Spatial DropOut, or DropBlock 
- **Normalization of the network activations by their mean and variance**: Batch Normalization (BN), Cross-GPU Batch Normalization (CGBN or SyncBN), Filter Response Normalization (FRN), or Cross-Iteration Batch Normalization (CBN) 
- **Skip-connections:** Residual connections, Weighted residual connections, Multi-input weighted residual connections, or Cross stage partial connections (CSP)

이 아래부터는 어떤 것을 제외하고, 어떤 것을 사용했는지에 대한 설명이다. 

- **Activations :** PReLU, SELU (이 둘은 train하기 어려워서 제외), ReLU6 (quantization network을 위해 만들어졌으니 제외)
-  **Regularization** : Drop Block paper에서 다른 방법들과 비교를 친절히 해놨고 제일 좋다고 했으니 사용
- **normalization** : GPU만 쓸거니깐 syncBN은 제외

이 아래부터는 single GPU만 쓰니깐, 이거에 맞춘 추가적인 design과 향상 부분이다.  

- **data augmentation :** Mosaic, Self-Adversarial training (SAT) 이 두개의 새로운 방식 소개

- **genetic algorithm**을 통해 최적의 hyper parameter tuning

- 효율적인 트레이닝과 탐색을 위해 기존 방법을 좀 수정하였다.

  (modified SAM, modified PAN, Cross mini batch normalization(CmBN) )



#### 6-3. Mosaic

![image-20220216230320368](https://user-images.githubusercontent.com/71866756/154501482-d70a4da9-9dbb-4c41-8da6-ef532fbe8c9c.png)

4개의 training image를 섞어 하나의 이미지를 만든다. 

이렇게 4개의 context가 한 이미지에 들어가게 되고, normal context 밖의 객체까지 찾아낼 수 있었다고 한다. 

각 레이어마다 4개의 이미지에 대해서 batch normalization을 진행하기 때문에 batch size도 클 필요가 없다고 한다. 

#### 6-3. Self-Adversarial Training (SAT)

2 forward, backward stage에서 사용

1st stage : network weight를 original image로 바꾸게 되면, 자기 자신에 대해 적대적으로 만들어줘서 원하는 객체가 없다고 오인하게 만들 수 있다. 

> 무슨 의미인지 몰라서 찾아보니,
>
> trained된 모델에 대해서 weight는 freeze시키고, input image를 optimize하게 한다. 
>
> 이렇게 하면 input image에 noise가 낀다고 한다. 

2nd stage : 이 변환된 이미지에서 객체를 찾게 한다. 

>노이즈가 낀 이미지에 대해서 학습을 진행하면 좀 더 디테일한 영역을 학습 시킬 것이다. 



#### 6-4. CmBN (cross mini batch normalization)

CmBN은 CBN의 수정된 버전으로 mini batch 사이의 통계정보만 모은다.



#### 6-5. modified SAM & modified PAN

SAM을 spatial-wise attention -> point-wise attention 변경 + PAN의 shortcut connection을 concatenate으로 변경

</div>
</details>

# 7. YOLOv4 최종 정리

<details>
<summary>접기/펼치기</summary>
<div markdown="1">  


| 구성                    | bag of freebies (BoF)              | Bag of Specials                                        |
| ----------------------- | ---------------------------------- | ------------------------------------------------------ |
| backbone (CSPDarknet53) | cutmix, Mosaic (data augmentation) | Mish (활성함수)                                        |
|                         | DropBlock (정규화)                 | Cross-stage partial connections (CSP)                  |
|                         | label smoothing                    | multi input weighted residual<br />connections (MiWRC) |
| Neck (SPP+PAN)          |                                    |                                                        |
| Head (YOLOv3)           |                                    |                                                        |
| detector                | CIoU-loss                          | mish                                                   |
|                         | CmBN                               | spp block                                              |
|                         | DropBlock                          | sam block                                              |
|                         | Mosaic                             | pan path aggregation block                             |
|                         | self-adversarial training          | DIoU-NMS                                               |
|                         | eliminate grid sensitivity         |                                                        |
|                         | multiple anchor for single gt      |                                                        |
|                         | cosine annealing scheduler         |                                                        |
|                         | optimal hyper parameter            |                                                        |
|                         | random training shapes             |                                                        |

![image](https://user-images.githubusercontent.com/71866756/154618414-d61a8c12-d5eb-444c-8693-3ac1d459807c.png)



</div>
</details>

# 8. Experiment
<details>
<summary>접기/펼치기</summary>
<div markdown="1">  

  - MS COCO (test-dev 2017), ImageNet(ILSVRC 2012 val)에서 실험

#### 8-1. 분류 문제

![image-20220217224304462](https://user-images.githubusercontent.com/71866756/154501420-ed54be61-f1f3-4031-8d15-ce1a1a6c4ac3.png)

위 방법들 중 아래 방법들에서 정확도 향상을 확인할 수 있었다. 

- **cutmix**

- **mosaic**

- **class label smoothing**

- **mish**

  따라서 backbone에서 **cutmix, mosaic, class label smoothing 사용**했다 (mish는 옵션) 

![image-20220216232103322](https://user-images.githubusercontent.com/71866756/154501388-b86fbf96-ba83-4eb7-8b25-1df5923c68df.png)

#### 8-2. Detection 문제

- **S** : Eliminate grid sensitivity 

  the equation bx = σ(tx)+ cx, by = σ(ty)+cy, where cx and cy are always whole numbers, is used in YOLOv3 for evaluating the object coordinates, therefore, extremely high tx absolute values are required for the bx value approaching the cx or cx + 1 values. 

  We solve this problem through multiplying the sigmoid by a factor exceeding 1.0, so eliminating the effect of grid on which the object is undetectable. 

- **M**: Mosaic data augmentation

- **IT**: 하나의 이미지에 대해서 IoU threshold보다 크게 나온 anchor들을 사용

- **GA**: 전체 주기의 처음 10%동안, genetic algorithms을 이용하여 hyperparameter tuning

- **LS**: label smoothing 

- **CBN**: Cross mini-Batch Normalization (CBN)을 사용

- **CA**: Cosine annealing scheduler사용 

- **DM**: Dynamic mini-batch size로 자동으로 mini batch를 늘려가는 것

- **OA**: 512x512 resolution에서 최적화된 Anchors 사용

- **GIoU, CIoU, DIoU, MSE** : BBox regression에 서로 다른 loss 사용

![image-20220216232547346](https://user-images.githubusercontent.com/71866756/154501341-0bb37abd-8adb-435d-a1f4-413cc6ac2105.png)

- **BoS 테스트 (PAN, RFB, SAM, Gaussian YOLO(G), ASFF)**

  -> 가장 좋았던 결과는 **SPP,PAN, SAM**

![image-20220216232535847](https://user-images.githubusercontent.com/71866756/154501309-79007262-3f04-4be0-91bf-f76fb9d6eb88.png)



#### 8-2. Influence of different backbones and pretrained weightings on detector training

- classification에서 좋았다고 detector에서도 좋진 않다.

  > CSPResNeXt50 > CSPResNeXt53 (classification)
  >
  > CSPResNeXt50 < CSPResNeXt53 (object detection)

- BoF와 mish를 CSPResNeXt50 classifier에 적용했을 때, classification accuracy가 증가했지만, 이 pre trained 된 weight를 detector에 적용했을 때 좋지 않았다. 
- 근데 BoF와 mish를 CSPResNeXt53에 적용했을 때는classification 그리고 이 pretrained된 weight를 detector에 적용했을 때는 둘 다 좋았다.

-> 결론은 **CSPResNeXt53 backbone으로 좋다.** 

![image-20220216233254920](https://user-images.githubusercontent.com/71866756/154501257-3e92b881-9cfb-4cb9-87ed-c61b4e8a2607.png)

#### 8-3. Influence of different mini-batch size on Detector training

BoF와 BoS를 적용하니 mini batch는 detector performance에 영향을 주지 않아서, 비싼 GPU에서 돌릴 필요가 없어졌다.

![image-20220216233230746](https://user-images.githubusercontent.com/71866756/154501216-383e09f9-80f4-4b58-bcb1-af7c66a33dc3.png)

</div>
</details>

# 9. Results  
<details>
<summary>접기/펼치기</summary>
<div markdown="1">

![image](https://user-images.githubusercontent.com/71866756/154505163-abc1ed82-605b-47bc-8f83-a38641731f31.png)

그 어떤 detector보다 빠르고 정확했다!

![image](https://user-images.githubusercontent.com/71866756/154500907-a0a6aa0d-6c1e-4361-be0f-2ab4096c8489.png)
![image](https://user-images.githubusercontent.com/71866756/154500779-5cbece22-6800-4a8b-8a88-ac18a069b89e.png) 
![image](https://user-images.githubusercontent.com/71866756/154500668-8054fa0e-57bd-4baf-9cff-decb454fb75e.png) 
</div>
</details>

# 10. Appendix

<details>
<summary>접기/펼치기</summary>
<div markdown="1">  

#### 1. Cross mini Batch Normalization (CmBN)

- **Batch Normalization 이란?**

  각 batch로 계산한 통계값 (평균, 분산)이 전체 training set과 일치한다고 가정하여, **mini-batch** 안에 존재하는 **sample**들로 **평균과 분산을 계산**한다. 

  ![image](https://user-images.githubusercontent.com/71866756/154500620-d469c8a1-b96a-417b-935e-cb41c770c3f8.png)

  이 값을 토대로 **whitening**을 진행한다. (값의 분포가 평균 0, 분산 1을 갖도록 하는 방법이후,

  ![image](https://user-images.githubusercontent.com/71866756/154500587-80a5f8bd-f1e0-4bdf-b749-ac253aa10e39.png)

   whitening된 값에 학습 가능한 파라미터 감마와 베타를 갖도록 선형 변환을 수향한다. 

  ![image](https://user-images.githubusercontent.com/71866756/154500537-ff21b31e-af91-4b61-ad0a-3808589aeb58.png)

  - **Batch Normalization의 문제점**

    많은 연산량과 메모리 점유율이 필요한 object detection, segmentation task에서는 GPU 한계 때문에 작은 batch-size를 사용할 수 밖에 없다. 

    **작은 batch-size의 통계값은 training set의 통계값과 동일하지 않게 된다.** 

- **Cross-iteration Batch Normalization (CBN) 이란?**

  small batch에서 발생하는 Batch Normalization 문제를 해결하기 위한 방법으로, **이전 iteration에서 사용한 sample 데이터의 평균과 분산을 계산**한다.

  ![img](https://blog.kakaocdn.net/dn/b06q76/btq42zwx7FA/FKo5JAa5ckpCd3a6tZpDlK/img.png)

  현재 가중치와 이전 가중치가 다르기 때문에, 단순히 이전 iteration의 통계값을 이용하면 부정확하기 때문에, 테일러 시리즈를 사용하여 **이전 가중치와 현재 가중치의 차이만큼 보상**하여 근사한다.

  가중치 값이 매우 작다고 가정하기 때문에 테일러 시리즈를 사용할 수 있다.  

  - **테일러 시리즈**

    ![image](https://user-images.githubusercontent.com/71866756/154500465-7bf16604-fa46-4a6c-a953-60bfb023afe1.png)

    ![image](https://user-images.githubusercontent.com/71866756/154500414-c57dd32c-4b80-45b4-be5a-c74e013a35ae.png)

    위 식에 따라서 가중치의 차이도 아래 식으로 나타낼 수 있다. 

    ![image](https://user-images.githubusercontent.com/71866756/154500310-9c0b2deb-4686-47ba-9a0c-906167cb75ca.png)

    > f(t)를 현재 가중치, f(a)를 이전 가중치로 보고 f(t) - f(a)를 전개한 것!

    어차피, 값의 차이는 매우 적으니, 3차 이상부터는 날린다. 

    ![image](https://user-images.githubusercontent.com/71866756/154500228-fa2152fb-dc10-465f-ad0d-9370c8661cf5.png)

    이렇게 새로 구한 평균과 분산을 이용해 batch normalization을 진행한다. 
  
    > 추가적으로, 바로 이전 iteration 뿐만 아니라 몇 개 이전의 iteration까지 같이 계산할 수 있고, 이를 hyper parameter k로 정의한다. 
    >
    > ![image](https://user-images.githubusercontent.com/71866756/154500171-ab22ef25-0e2e-46ae-a3c3-ed7ffe135ce1.png)
    >
    > k 값이 있음에 따라 평균과 E(X^2)을 구하는 식이 달라진다.
    >
    > (현재 - 1번째 iter / 현재 - 2번째 iter / ... / 현재 - k번째 iter 이런식으로 계산)
    >
    > ![image](https://user-images.githubusercontent.com/71866756/154500111-82d19cdb-3370-471c-b325-e71494de8c27.png)
    >
    > 원래 E(X^2)을 구하는 식과 달라진 점이 있는데, 바로 max연산이다. 
    >
    > 원래 분산은 E(X^2) - E(X)^2으로 계산하고, E(X^2)은 항상 E(X)^2보다 크지만, 테일러 시리즈로 근사하였기 때문에, 작아질 수 있어서 음수가 나오지 않게 하기 위해 max 연산을 취한다. 

- **Cross-mini batch normalization (CmBN)**

  CBN이 iteration 단위로 되어있다면, CmBN은 mini batch 단위로 계산하는 것

[**Ref**]

[블로글 링크](https://deep-learning-study.tistory.com/635)

[논문 링크](https://arxiv.org/abs/2002.05712)



------

#### 2. Path Aggregation Network for Instance Segmentation (PAN, PANet)

- **PAN이란?**

  **PANet**이라고도 불리는 **PAN**은 Mask R-CNN을 기반으로 Instance Segmentation을 위한 모델이다. 

- **PAN의 주요 방법론**

  - **Bottom-up Path Augmentation**

    ![image](https://user-images.githubusercontent.com/71866756/154500045-7797532c-cabd-4d8b-b4b7-3f93d113804a.png)

    기존의 방법이 빨간 선이였다면 Bottom-up Path Augmentation은 초록 선이다. 

    low-lovel feature와 high-level feature 사이의 **경로를 단축**시켜, **low-level feature를 최대한 살리는 방법**이다. 

  - **Adaptive Feature Pooling**

    ![image](https://user-images.githubusercontent.com/71866756/154499980-fe34dc51-3b2a-46da-b8ec-e52f73a852e1.png)

    N2~N5 각각의 feature map에 RPN이 적용되어 ROI를 생성한다.

    이 ROI는 ROI Align을 거쳐 일정한 크기의 벡터가 생성된다. 

    각 피쳐맵에서 생성한 일정한 크기의 벡터를 **max 연산**으로 하나로 결합하여 **class와 box 예측**

    N2~N5의 모든 정보를 활용하여 **low-level, high-level 정보 모두 활용 가능**

  - **Fully-connected Fusion**

    ![image](https://user-images.githubusercontent.com/71866756/154499922-ddd95a56-5521-44b8-97bb-030483da6aaf.png)

    **FCN**은 각 **class에 해당하는 pixel**을 나타내는 이진 마스크 예측, 

    **FC**는 **배경과 객체를 구분**하는 마스크를 예측하여 이 둘을 더하여 예측값 생성

[Ref]

[블로그 링크](https://deep-learning-study.tistory.com/637)

[논문 링크](https://arxiv.org/abs/1803.01534)



------

#### 3. IoU, GIoU, DIoU

- **IoU (intersection over union)**

  예측한 BBox와 ground truth BBox의 교집합 / 합집합으로 표현되며, 

  ![image](https://user-images.githubusercontent.com/71866756/154498026-24235af9-88e7-4089-b64e-fc547a20e848.png)

  IoU가 1에 가까울수록, 즉 **두 BBox가 겹칠수록 loss는 작아진다**.  

  단순히 BBox에 좌표에 대한 l2 norm으로 구하는 loss보다 훨씬 정확하다. 

  ![image](https://user-images.githubusercontent.com/71866756/154498012-c8fc4516-a63f-4841-9b4a-4d8eee6ed6cc.png)

  하지만, 단점으로는 **교집합이 존재하지 않을 때 문제가 발생**한다. 

  ![image](https://user-images.githubusercontent.com/71866756/154499853-a93b653e-aaa8-4732-a198-71d26f7e8e09.png)

  제일 아래쪽 사진이 loss가 제일 커야 정상이지만, 그렇지 못한다. 

- **GIoU (Generalized Intersection over Union)**

  GIoU는 IoU의 문제점을 해결하기 위해 등장했다. 

  예측 BBox와 ground truth BBox를 모두 포함하는 제일 작은 사각형을 구하는 것!

  ![image](https://user-images.githubusercontent.com/71866756/154499775-bb9f9f3a-11e0-4ba8-bf9d-34f3c1118e41.png)

  GIoU식을 보면, IoU에서 전체 넓이 C에 대해서 C - (A U B) 로 중간 그림에서 보면, 전체 넓이에 대해 회색부분의 넓이 비를 구하여 뺀 것이다. 

  쉽게 말해서, 같은 IoU에 대해서 회색 공간의 크기에 비례하여 loss도 키울 수 있게 되는 것이다.  

  이렇게 하면 IoU의 문제점을 해결할 수 있다. 

  Loss = 1 - GIoU

  ![image](https://user-images.githubusercontent.com/71866756/154498113-060c5502-eb9f-41d0-b9f5-ca04723bbfd3.png)

  하지만, 단점으로는 **예측 BBox가 ground truth BBox를 포함**하고 있으면, **IoU와 마찬가지**로 동작하기 때문에 수렴속도가 느리고 성능이 좋지 않다. 

  ![image](https://user-images.githubusercontent.com/71866756/154499574-d1f6a53a-51b8-4ada-bc37-102c7c802240.png)

- **DIoU (Distance Intersection over Union)**

  GIoU가 면적 기반의 페널티를 부여했다면, DIoU는 여기에 거리 기반의 페널티를 부여한다. 

  ![image](https://user-images.githubusercontent.com/71866756/154499502-75ce2652-1e4f-45e1-95b6-d76f79ada241.png)

  ![image](https://user-images.githubusercontent.com/71866756/154499381-2315171d-4a29-465b-a5b7-74ea1000ac2e.png)

  d는 두 BBox의 중심 거리를 의미한다.

  c는 두 BBox를 포함하고 있는 가장 작은 직사각형을 의미한다. 

  장점으로는 **두 BBox 중심의 거리를 직접적으로 줄이기** 때문에 **GIoU에 비해 수렴이 빠르다.** 

- **CIoU (Complete Intersection over Union)**

  성공적인 BBox regression은

  1. 겹치는 부분
  2. 중심점 사이 거리
  3. 높이, 너비 비율

  이 세가지를 모두 고려해야 한다. 

  1,2까지 고려한 것이 DIoU라면, 이 세가지를 모두 고려하는 것이 CIoU이다. 

  ![image](https://user-images.githubusercontent.com/71866756/154499252-d2ed6824-8598-4618-be88-490fd4e5deba.png)

  v는 두 BBox의 aspect ratio의 일치성을 측정하는 역할이며, alpha는 positive-trade-off parameter로 non-overlapping case와 overlapping case의 균형을 조절한다. 

  ( 이 부분에 대해서는 추후에 좀 더 수식을 공부해야겠다. )

[**Ref**]

[IoU 개념 정리 (IoU, GIoU, DIoU, CIoU)](https://silhyeonha-git.tistory.com/3)

[GIoU(Generalized Intersection over Union)](https://silhyeonha-git.tistory.com/3)

[논문 링크](https://arxiv.org/abs/1911.08287)



------

#### 4. Spatial Pyramid Pooling (SPP)

SPP는 Spatial Pyramid Matching에서 기인하였으므로, Spatial Pyramid Matching에 대해 알아보자. 

- **Spatial Pyramid Matching**

  Spatial Pyramid Matching은 fully connected layer의 input 사이즈가 정해져 있다는 문제점을 해결하기 위해 등장하였다. 

  - **Bag of words**

    원래 Bag of words는 문서를 자동으로 분류하기 위한 방법으로, 글에 포함된 **단어들의 분포**를 보고 **문서의 종류를 판단**한다. 

    **영상처리, CV**에서는 이 기법을 주로 **이미지 분류나 검색**에서 사용되었지만, 최근에는 물체나 scene을 인식하는 용도로도 사용되고 있다. 

    > 영상 분류에서 BoW 방법이 어떻게 사용되는지 알아보자. 
    >
    > **Step1**. **Feature Extraction**
    >
    > ( 영상에서 feature (주로 SIFT 등의 local feature)들을 추출 )
    >
    > **Step2**. **Clustering**
    >
    > feature clustering (k-means clustering) 수행
    >
    > (codeword는 codebook을 구성하는 feature로 hyperparameter를 통해 영상 feature를 몇개의 clustering으로 나눌지 결정하고, 이에 따라 codeword의 개수가 정해진다. )
    >
    > **Step3**. **Codebook Generataion**
    >
    > 대표 feature들로 codebook 생성
    >
    > (즉, 이미지 분류에 중요한 feature들을 가지고 있다고 생각하면 된다.
    >
    > codeword는 codebook을 구성하는 feature로 hyperparameter를 통해 영상 feature를 몇개의 clustering으로 나눌지 결정하고, 이에 따라 codeword의 개수가 정해진다. )
    >
    > **Step4.** **Image Representation**
    >
    > image를 codeword와 매칭하여 히스토그램으로 표현
    >
    > **Step5. Learning and Recognition**
    >
    > 학습 및 인식은 크게 두가지 방법으로 진행한다. 
    >
    > 1. Baysian 확률을 이용한 generative 방법 : 히스토그램 값을 확률로서 해석
    > 2. SVM을 이용한 discriminative 방법 : 히스토그램 값을 feature vector로서 해석

  - **Spatial Pyramid Matching**

    Bag of words의 단점은 이미지가 기하학적인 정보를 잃어버린다는 점이다.

    이러한 단점을 해결하기 위해 등장하였다. 

    이미지를 **여러 단계의 resolution으로 분할**한 후, 각 단계의 **분할 영역마다 히스토그램**을 구하여 전체적으로 비교하는 방법이다. 

    ![image](https://user-images.githubusercontent.com/71866756/154499205-03b92669-ac1f-4ae0-9a2f-a9fc99425619.png)

    **이거 논문 잇는데 이해가 안되네;;;**

- **Spatial Pyramid Pooling**

  ![image](https://user-images.githubusercontent.com/71866756/154499025-c105d7de-c39c-455c-875c-049c26fdb033.png)

  위와 같이 미리 정해진 영역으로 나눠진 **피라미드**를 이용한다. 

  (4x4, 2x2, 1x1을 각각 하나의 피라미드라고 한다. **각 피라미드 안에서 한 칸을 bin**이라고 한다.)

  입력받은 **feature map을 각 bin에 대해서 max pooling** 연산을 한다. 

  maxpooling 결과를 stack하여 출력으로 내보낸다. 

  이렇게 되면, **출력값은 사전에 설정한 bin의 개수와 입력 채널 개수로 항상 동일**하게 된다.

  (fully connected layer의 input size가 동일해야 한다는 문제 해결!)

  [**Ref**]

  [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://velog.io/@lolo5329/%EB%85%BC%EB%AC%B8%EC%9A%94%EC%95%BD-Spatial-Pyramid-Pooling-in-Deep-Convolutional-Networks-for-Visual-Recognition)

  [Bag of Words 기법](https://darkpgmr.tistory.com/125)



------

#### 5. Spatial Attention Module (SAM)

![image](https://user-images.githubusercontent.com/71866756/154498929-237cc713-c915-4c5c-a714-ea62238a8c35.png)

**Average Pooling과 MaxPooling을 channel 축으로 적용**한 것이다. 

예를 들어, C x H x W -> (1 x H x W) x 2가 되는 것이다. 

특징으로는 feature간의 inter-spatial relationship을 통해 spatial attention을 생성하고, 

**channel attention이 어떤 정보가 있냐**에 집중했다면, **spatial attention은 정보가 어디에 있냐**를 중점으로 둔다고 한다. 

[**Ref**]

https://deep-learning-study.tistory.com/666

https://arxiv.org/abs/1807.06521



------

#### 6. Feature Pyramid Network

![image](https://user-images.githubusercontent.com/71866756/154498822-2e3d16ee-d8e3-49df-95d1-dabbdab69e9e.png)

**Top-down 방식**으로 특징을 추출하며, 

각 추출된 결과들인 **low-resolution 및 high-resolution 들을 묶는 방식**이다. 

각 레벨에서 독립적으로 특징을 추출하여 객체를 탐지하게 되는데 

상위 레벨의 이미 계산 된 **특징을 재사용** 하므로 **멀티 스케일 특징들을 효율적으로 사용**할 수 있다. 

CNN 자체가 레이어를 거치면서 피라미드 구조를 만들고 forward 를 거치면서 더 많은 의미(Semantic)를 가지게 된다. 

각 레이어마다 예측 과정을 넣어서 Scale 변화에 더 강한 모델이 되는 것이다. 

이는 skip connection, top-down, cnn forward 에서 생성되는 피라미드 구조를 합친 형태이다. 

forward 에서 추출된 의미 정보들을 top-down 과정에서 업샘플링하여 해상도를 올리고

forward에서 손실된 지역적인 정보들을 skip connection 으로 보충해서 스케일 변화에 강인하게 되는 것이다.

출처: https://eehoeskrap.tistory.com/300 [Enough is not enough]



------

#### 7. Mish 활성화 함수  

![image](https://user-images.githubusercontent.com/71866756/154618610-7c33b696-6b79-4e07-804e-fbbc3936e946.png)  
  

![img](https://blog.kakaocdn.net/dn/bNMfJN/btqEGDFuxqe/aEPskQf9rGAOikQRykXxnk/img.png)

Mish 활성화 함수는 무한대로 뻗어나가기 (unbounded above) 때문에 포화를 피할 수 있으며, 

> 여기서 포화의 의미를 gradient exploding으로 생각했는데 그게 아니였다. 
>
> 포화(saturation)란 입력값이 변해도 출력값이 변하지 않는 상태를 의미한다. 
>
> 즉, gradient 포화 = gradient vanishing로, 이 둘은 같은 의미이다. 

bounded below이기 때문에 strong regularization이 나타나며 overfitting을 감소시킬 수 있다고 한다. 

또한, ReLU와는 다르게 음수인 부분에서도 gradient 존재하며, 작은 음수의 input은 작은 음수 값으로 매칭이 된다. 따라서 expressivity와 gradient flow를 향상시킨다고 한다. 

[**Ref**]

https://hongl.tistory.com/213

https://stats.stackexchange.com/questions/544739/why-does-being-bounded-below-in-swish-reduces-overfitting



------

#### 8. Cross-stage partial connections (CSP)

CSPNet의 구조인 CSP는 컴퓨팅 파워가 낮은 환경에서도 용이하게 하기 위한 구조로서, network의 연산량이 optimization 과정 중, gradients의 정보 중복으로 인해 증가한다는 점을 고려한 결과이다. 

![image](https://user-images.githubusercontent.com/71866756/154618519-a6da010d-85d1-4b41-a56b-05dae888c073.png)

위 그림에서 dense layer의 입력과 출력이 concatenation이 된다. 

![image](https://user-images.githubusercontent.com/71866756/154618536-bf6138ff-c490-40d7-b701-c0e3d15f881c.png)

위 식을 보면 중복된 input으로 인해, 가중치를 업데이트할 때, gradient 정보도 중복이 된다.

 ![image](https://user-images.githubusercontent.com/71866756/154618552-d7a10328-aa45-4de9-90e9-5241f6d3fedd.png)

![image](https://user-images.githubusercontent.com/71866756/154618562-d6113e88-f717-4ee9-ae95-48cdfc949137.png)

![image](https://user-images.githubusercontent.com/71866756/154618576-6f424ad7-6550-4109-a979-b9acfd6c1850.png)

[**Ref**]

https://ichi.pro/ko/cspnet-cross-stage-partial-network-64805303419044





</div>
</details>
