# 목차

- [**CutMix 출현 배경**](#1-cutmix-출현-배경)
- [**CutMix vs other data augmentation method**](#2-cutmix-vs-other-data-augmentation)
- [**CutMix algorithm&expression**](#3-cutmix-algorithm-&-expression)
- [**Results**](#4-results)
- [**성능평가지표**](#성능평가지표)



# 1. CutMix 출현 배경

### 기존 Drop Out 기법

- **특징**

  black pixel 또는 random noise로 구성된 patch를 덧붙이는 방식

- **장점**
  - generalize performace 향상
- **단점**
  - 정보의 손실 (information loss)
  - 학습의 비효율성 증가 (leads to training inefficiency)

이러한 단점을 보완하기 위해서 **crop된 training data patch**를 이용하는 CutMix 기법이 출현하였다. 



# 2. CutMix vs Other data augmentation 

![image-20220124221431600](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124221431600.png)

![image-20220124223923619](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124223923619.png)

### CutMix

- **방식**

  training image들을 자르고 합치는 방식으로, 각각이 합쳐진 영역에 따라 label 또한 변경

- **장점**

  - generalize performance 향상
  - [input noise와 out of distribution에 robust](###noise와-ood에-강건한-cutmix)
  - 정보의 손실 X
  - object 전체의 view + partial view를 통한 분류로 localization 능력 향상
  - 이미지의 texture보다 shape에 focus를 두어 classification & detection에 탁월

- **단점**

  - 약간의 추가적인 연산이 필요 (큰 차이 X)

### MixUp

- **방식**

  일정 비율로 이미지 덧셈 연산, CutMix와 마찬가지로 label 또한 변경

- **장점**

  - classification 능력 향상

- **단점**

  - detection & localization에는 좋은 성능을 보이지 X

### CutOut

- **방식**

  이미지에서 일정 부분 crop (fill zero)

- **장점**

  - classification 능력 향상

- **단점**

  - detection & localization에는 좋은 성능을 보이지 X

### 

# 3. CutMix algorithm & Expression

### Pseudo code

![image-20220124222339749](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124222339749.png)

### Expression

![image-20220124222403970](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124222403970.png)
$$
x\in R^{W*H*C},\; y:label
\\M\; :\; binary\; mask\;(crop\; 영역 \;pixel\; value\; :\; 0)\\
\lambda\sim\beta(\alpha,\alpha)\\
$$
**현재 논문에서는 alpha값을 1로 두어 Unif(0,1) 분포로 샘플링한다.**
$$
\alpha=1일 \;때\; Unif(0,1)이\; 되는 \;증명)\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \\
X\sim Beta(\alpha, \beta)\\
f_X(x)=\frac {1}{B(\alpha, \beta)}x^{\alpha-1}(1-x)^{\beta-1}\\
(0<x<1, \alpha,\beta >0)
\\확률분포에\; 대한\; 확률분포이니깐\; x의\; 범위는\; 0부터\; 1\\
B(\alpha, \beta)=\int_0^1x^{\alpha-1}(1-x)^{\beta-1}dx이므로\\
Beta(1,1)=f_X(x)=\frac {1}{B(1,1)}=\frac {1}{1}=\frac {1}{1-0}=Unif(0,1)
$$
**alpha=1로 정한 실험적 배경)**

왼쪽이 input image에 대한 CutMix, 오른쪽이 feature map 수준의 CutMix (0=image level, 1=after first conv-bn, 2=after layer1, 3=after layer2, 4=after layer3) 

after layer3를 제외하고 나머지에서는 성능 향상이 있었다. 하지만, input image 적용이 제일 효과적

![image-20220124231622747](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124231622747.png)

- **CutMix Steps**

  - sample bbox B

    ![image-20220124223301451](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124223301451.png)

    ![image-20220124223340489](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124223340489.png)
    $$
    위\; 수식에서\; r_w,\; r_h가 \;각각\; W\sqrt{1-\lambda}, \;H\sqrt{1-\lambda}인 \;이유는\\
    crop되는 \;영역의\; 비율을\; 1-\lambda로 \;맞추기\; 위함이다.\;(\frac{r_wr_h}{WH}=1-\lambda)
    \\r_x,r_y : 자를\; 이미지\; 중앙좌표\\
    r_w,r_h : 자를 \;이미지\; 너비,\; 높이
    $$
    

#  4. Results

### ImageNet Classification

- **Test할 기법**

  - CutOut

    Mask size 112x112, uniformly sample

  - Stochastic depth

    ResNet에서 residual block을 확률적으로 drop (rate=0.25)

  - MixUp

  - Manifold MixUp

    MixUp을 feature map 수준에서 random하게 선택하여 적용(alpha=0.5와 1.0 중 1.0이 성능이 더 좋아 1.0으로 실험)

  - DropBlock

    270 epoch 

  - CutMix

  - Feature CutMix

    feature map수준에서 CutMix

- **Base setting**

  - ImageNet-1k (1.2M training images, 50K validation images, 1K categories)
  - data augmentation
    - Resizing
    - cropping
    - flipping

  - Training
    - 300epoch
    - 초기 lr = 0.1 (75, 150, 225마다 0.1배)

- **Result**

  ![image-20220124225826683](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124225826683.png)

  ![image-20220124225830524](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124225830524.png)

  - ResNet-50 -> ResNet-101 (depth increase) : 1.99% 개선
  - ResNet-50 + CutMix : 2.28% 개선
  - depth를 늘리는 것보다 CutMix를 적용하였을 때, 성능개선이 더 우수
  - 깊은 신경망에서의 CutMix 성능 역시 우수



### CIFAR Classification

- **Test할 기법**

  - Stochastic depth

    rate=0.25

  - Label smoothing (rate=0.1)

    one-hot이 아닌 정답이 아닌 label에도 일정 값을 할당해주는 방법

  - CutOut

    Mask size 16x16

  - CutOUt + Label smoothing (rate=0.1)

  - DropBlock

    keep-prob(0.9), block_size (4)

    ![image-20220124230654460](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124230654460.png)

    ![image-20220124230805634](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124230805634.png)

    (dropout과 다르게 block단위로 drop, feat_size=feature map size, lambda=activation unit drop rate)

  - DropBlock + Label smoothing (rate=0.1)

  - MixUp (alpha=0.5, 1.0)

  - Manifold MixUp (alpha=1.0)

  - CutOut + MixUp (alpha=1.0)

  - CutOut + Manifold Mixup(alpha=1.0)

  - ShakeDrop

    - shakeshake와 비슷한 방식으로 shakeshake은 ResNeXt에만 적용이 가능하기 때문에, ResNet에도 적용하기 위해 shackDrop 사용
    - x + aF(x) + (1-a)F(x) 구조라서 a가 0~1값으로 들어가기 때문에 x만 되는 경우는 없음
    - shake drop은 b_l=1이면 x+F(x), b_1=0이면 x+aF(x)라서 그냥 ResNet에도 적용가능

    ![image-20220124231209076](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124231209076.png)

    

  - CutMix

  - CutMix + ShakeDrop

- **Base setting**

  - Minibatch (64)
  - 

  - Training
    - 300epoch
    - 초기 lr=0.25 (150epoch : 0.025, 225epoch : 0.0025)

- **Result**

  ![image-20220124231027176](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124231027176.png)

  

### CutMix 방식에 따른 성능

![image-20220124231848636](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124231848636.png)

`Center Gaussian CutMix` : rx,ry를 평균이 image 중앙인 gaussian 분포

`Fixed-size CutMix` : rw, rh를 16x16으로 고정

`Scheduled CutMix` : training 진행에 따라 선형적으로 확률을 0~1로 증가

`One-hot CutMix` : lable을 비율로 정하는 것이 아닌, 합친 이미지 중 넓은 이미지의 label 사용

`Complete-label CutMix` : 두 정답 label에 0.5씩 가중치 적용

- **결과**

  가장 기본적인 CutMix가 제일 성능이 우수



### Weakly Supervised Object Localization (WSOL)

- **정의**

  object의 정확한 위치가 아닌 label이 정답 label로 주어지는 경우

- **등장 배경**

  bbox, segmentation annotation을 만드는 데 드는 비용과 시간을 아끼고자 등장

- **동작 방식**

  feature map에서 어떤 부분이 activate 되는지에 따라 localization이 진행됨

  ![image-20220124232335485](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124232335485.png)

- **CutMix, MixUp, CutOut이 localization에 미치는 영향**

  - **CutMix이 WSOL에 효과적인 이유**

    localization에서 중요한 것은 전체 object 영역에서 cue를 뽑아내는 것으로, CutMix를 통해 객체의 특징을 나타내는 각각의 crop된 이미지들을 잘 분류해내면, 그것은 곧객체의 특징을 잘 잡아낸다는 의미이다. 

    따라서 객체가 crop되었을 때, 각각의 부분에 대한 cue들이 많아지는 것과 같은 의미이며 crop된 이미지가 아닌 전체 이미지를 합쳐서 보았을 때, 그만큼 객체를 탐지하는 cue들이 많아지는 것과 같다고 해석할 수 있을 것 같다. (개인적인 해석)

  - **MixUp이 WSOL에 효과가 좋지 않은 이유**

    MixUp의 경우, 문제점이 자연스러운 이미지가 아니라는 점이다. 

    즉, 어떤 객체를 구별해낼 cue가 그만큼 적다는 얘기이고, cue가 적은 만큼 확실히 구별가능한 cue들로 객체를 인식하고 localization을 진행할 것이기 때문에 성능이 좋을 수가 없다. 

  - **CutOut 역시 마찬가지로 WSOL에 효과적이다.**

- **결과**

  ![image-20220124232949379](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124232949379.png)



### Transfer learning of Pretrained Model

`Downstream task` : pre-trained된 model을 이용하여 supervised-learning task를 해결하는 것

`image captioning` : 이미지를 모델에 돌려 이미지에 대한 설명을 출력

- **Pascal VOC object detection**

  - 원래 SSD와 Faster RCNN은 VGG-16기반 모델이지만, backbone을 ResNet-50으로 변경
  - ResNet-50 backbone은 ImageNet 기반 pretrained 모델이며 여기에 Pascal VOC 2007, 2012에 맞춰 fine-tuning 진행 

- **MS-COCO image captioning**

  - Neural Image Caption (NIC)이 base model
  - backbone의 encoder단을 GoogLeNet에서부터 ResNet-50으로 변경

- **결과**

  ![image-20220124233715960](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124233715960.png)





### noise와 OOD에 강건한 CutMix

보통 deep한 모델의 경우 adversarial attack (일부로 노이즈를 섞어서 틀리게 하는 기법)에 매우 취약하며, input이미지의 약간의 변화에도 매우 민감하다. 

하지만, CutMix의 경우 adversarial samples, occluded (가려진) samples, out of distribution samples(학습 데이터와 다른 분포를 갖는 데이터 혹은 학습 데이터에 포함되지 않은 class의 데이터)에 매우 robust하다.

-  Adversarial perturbation

  fast gradient sign method(FGSM)이 사용되었다. 

  ( 이미 학습된(가중치가 고정된) 모델에 대해서 손실함수의 gradient를 계산하여 손실을 최대화하는 이미지를 생성하는 방법 )

  - 결과

    ![image-20220124234457042](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124234457042.png)

- Occlusion

  center occlusion (가운데 hole을 0으로 채우는 방법), boundary occlusion (hole을 제외한 부분을 0으로 채우는 방법)이 사용되었다. 

  - 결과

    ![image-20220124234540798](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124234540798.png)

- In-between

  하나의 정답 레이블이 아닌 다른 label들이 섞여 있는 것

  - 결과

    ![image-20220124234639353](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124234639353.png)

- Out Of Distribution (OOD)

  CIFAR-100 datasets으로 학습된 PyramidNet-200모델을 사용

  - OOD datasets

    TinyImageNet, LSUN, uniform noise, gaussian noise 등 총 8개의 datasets을 이용

  - 결과

    ![image-20220124234845576](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220124234845576.png)



# 성능평가지표

- **Intersection over union (IOU)**

  예측된 bbox와 ground truth bbox의 중첩되는 면적을 합집합의 면적으로 나눈 것 (0.5 이상일 경우 TP, 미만일 경우 FP)

- **Precision (정확도)**

  TP/(TP + FP) 즉, TP/(All detections)

- **Recall (검출율)**

  TP /(TP+FN) 즉, TP/(All Ground Truths)

- **Precision-recall (PR곡선)**

  recall값(x)에 따른 precision값(y)

- **Confidence**

  검출한 것에 대해 알고리즘이 얼마나 정확하다고 생각하는지 알려주는 값 (학습한 대로 값을 출력)

- **Threshold**

  confidence의 제약을 줘서, 얼마 이상일 시 제대로 검출되었다고 판단하는 값

- **AP (average precision) **

  보통 계산 전에 PR 곡선을 단조적으로 감소하는 그래프로 변경 후, 각 사각형의 면적을 계산한다.

- **mAP (mean average precision)**

  각 클래스의 AP를 구한 다음, 그것을 모두 합한 다음에 클래스의 개수로 나눠준다.

