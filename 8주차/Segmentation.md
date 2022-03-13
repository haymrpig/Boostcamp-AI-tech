# 목차



# 1. Semantic Segmentation

object들을 구분하지만, 클래스별로만 구분한다.

Semantic Segmentation의 주요 모델들을 알아보자.

#### 1-1.Fully Convolutional Networks (FCN)

- **특징**

  - 첫 end to end 구조	
  - 입력과 출력의 해상도가 같음
  - fully connected layer가 아닌 fully convolutional layer (1x1 conv)

- **구조적 특징**

  - **Convolution을 통해 작은 feature map까지 줄인다.**

    > 더 넓은 receptive field를 확보할 수 있다. 
    >
    > 하지만, 해상도가 낮아진다. 

  - **Upsampling 적용**

    > 낮은 해상도를 높은 해상도로 복원!

    - **Transposed convolution**

      ![딥러닝에서 사용되는 여러 유형의 Convolution 소개 · 어쩐지 오늘은](https://cdn-images-1.medium.com/max/1200/1*Lpn4nag_KRMfGkx1k6bV-g.gif)

      위와 같은 방식은 현재 stride가 1인 상태이다. 

      transposed convolution은 중첩 부분이 생긴다 (overlap) 는 단점이 있다. 

      (stride와 convolution 사이즈를 잘 조절해야 한다. )

    - **Interpolation + convolution**

      `NN-resize convolution`, `Bilinear-resize convolution`

      먼저 학습 파라미터가 아닌 Nearest-neighbor 또는 Bilinear를 통해 interpolation을 진행하고, 학습 파라미터 convolution을 취하는 방법
      
      (overlap되는 문제를 해결할 수 있다. )

  - **Skip connection**

    낮은 해상도의 feature map을 upsampling한다고 해서 정보를 복원하는 것은 매우 어렵다. 
  
    또한 segmentation은 low level feature와 high level feature 둘 모두를 필요로 하기 때문에 skip connection을 추가한다. 

#### 1-2. U-Net

https://github.com/haymrpig/Boostcamp-AI-tech.git

논문에 잘 정리되어 있다!

#### 1-3. DeepLab

- **Keyword**

  `Conditional Random Fields(CRFs)` : 후처리에 사용되는 방법으로, 픽셀과 픽셀 사이 관계를 이어주고 픽셀 맵을 그래프로 보인 것

  (따로 공부해보는 것 추천!)

  - 보통 처음 layer를 통과해서 나오는 segmentation 결과는 실제 이미지와는 경계부분도 흐릿하고 다르다. 

    > 이는 나온 결과를 input과 비교하는 feed back 구조가 없어서 그렇다. 
    >
    > 이를 해결하기 위해서 이미지의 경계선들을 활용해서 출력이 경계에 맞도록 처리한다. 

  `Dilated Convolution` : 실제 convolution보다 더 넓은 receptive field를 가질 수 있다. 

  `Depthwise seperable convolution` : 채널별로 convolution 진행 -> pointwise convolution 진행

  > 계산량을 줄이면서 똑같은 결과를 낼 수 있다. 