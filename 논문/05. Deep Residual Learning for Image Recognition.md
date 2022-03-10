# Deep Residual Learning for Image Recognition

* https://arxiv.org/abs/1512.03385
* 2015.12
* Microsoft Research (Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun)



## Abstract & 1. Introduction

* (이미지로부터 필요한 feature map을 뽑아내기 위한) unreferenced function을 학습시키지 않고 residual function을 학습시키는 것으로 레이어 구성을 변경하였다
  * 최적화 시키기 더 쉽다
  * 레이어 깊이 증가에 따른 정확도 증진 효과를 더 잘 경험할 수 있다
* 이전 VGGNet보다 8배 깊은 레이어로 구성되어 있음에도 불구하고 파라미터 수는 더 적고 계산 복잡도는 더 낮다
* ILSVRC & COCO 2015 대회 성적(1위) 자랑



* 선행 연구들을 통해 알 수 있는 점
  * LeNet, AlexNet 등의 연구를 통해 CNN의 Image Classification에서의 효과성이 입증됨
  * 깊은 신경망은 low/mid/high-level feature들을 종단간 학습 방식으로 자연스럽게 통합
  * feature의 "level"은 쌓인 레이어의 수(깊이)에 의해 풍부해질 수 있으며, VGGNet/GoogLeNet 등의 선행연구에서 이가 밝혀졌다
    * R-CNN, SPPNet, Fast R-CNN, Faster R-CNN, FCN 등 Classification 외 컴퓨터 비젼 분야에서도 "deep"한 모델들의 효과성은 검증됨

* 깊이가 중요하다면 그냥 레이어를 깊게 쌓아버리면 되는 것 안될까?
  → **Vanishing/Exploding Gradients 문제** 때문에 그럴 수 없다.
  * 선행 연구들은 여러 initialization, normalization 기술들로 이를 해결하고자 함 - 10개 정도의 layer까지는 SGD를 통해 학습 가능하게 됨
  * 더 깊은 신경망에 대해서는 **degradation problem** 발생
  * 모델의 깊이가 깊어지면 정확도의 포화도가 다다르다가(=깊이 증가에 따른 정확도 증진 효과가 더뎌지다가?) 이후 급격히 저하된다. (Figure 1)
  * ![Fig1](https://user-images.githubusercontent.com/38153357/152932898-5ed90b35-01fc-4488-bed0-5028cb0b555a.png)
  * ※ 이 문제는 **overfitting에 의한 문제가 아니다!**
    * overfitting: training error은 낮은데 validation/test error은 높음
    * degradation 현상에서는 training error도 높아지는 것을 알 수 있음 (→ 학습 및 최적화의 문제)
* degradation 문제를 통해서 모든 시스템이 최적화 하기 쉽지 않다는 것은 알 수 있다
  * 그런데 깊이는 왜 최적화를 어렵게 만드는가?
    * 일반 shallow model VS shallow model에 입력값을 그대로 출력하는 멍텅구리 identity mapping을 쌓은 deeper model 비교 
      : deeper model이 shallow model에 비해 높은 에러율을 보여서는 안된다. → 근데 실험해보니 에러가 높네?
* 새로운 구조의 제안: 
  **deep residual learning** via **shortcut connection** (=skip connection, residual connection)
  * 레이어들이 우리가 원하는 underlying mapping <img src="https://latex.codecogs.com/svg.image?\mathcal{H}(x)" title="\mathcal{H}(x)" />을 바로 학습하도록 하지 X
  * <img src="https://latex.codecogs.com/svg.image?\mathcal{H}(x)" title="\mathcal{H}(x)" />를 residual(잔차) mapping <img src="https://latex.codecogs.com/svg.image?\mathcal{F}(x)" title="\mathcal{F}(x)" />와 identity mapping <img src="https://latex.codecogs.com/svg.image?x" title="x" />의 합으로 분해하여 residual mapping <img src="https://latex.codecogs.com/svg.image?\mathcal{F}(x)" title="\mathcal{F}(x)" />을 학습 하도록 하는 것이 더 쉬울 것으로 가정
    * 만약 이미 중간에 optimal한 결과가 나왔을 때, 이후 레이어들이 identity mapping으로 되게 만드는 것보다 residual mapping을 0으로 만드는 것이 더 쉬울 것이다.
    * ("reference"로써 이미 identity mapping <img src="https://latex.codecogs.com/svg.image?x" title="x" />이 주어진다)
  * identity mapping의 역할을 수행하는 shortcut connection (Figure 2의 우측 곡선 화살표)
  * ![Fig2](https://user-images.githubusercontent.com/38153357/152932938-6c1f70cc-c318-4a49-bf15-67a3c46d9588.png)
  * ※ 추가 파라미터나 추가 계산 복잡도 요구량이 없다!
    * 기존 SGD와 backpropagation으로 학습 가능하며 구현도 쉽다
* 실제로 실험해 본 결과,
  * ImageNet
    * 1) 깊이가 깊어도 쉽게 최적화 되며, residual learning을 사용하지 않은 동일 깊이의 plain 네트워크는 그러지 못하고 degradation 문제가 발생한다
    * 2) 깊이가 증가할 수록 정확도 증진 효과가 뛰어나다
    * 여러 ResNet을 앙상블 한 결과 3.57% top-5 error (1위)
  * CIFAR-10
    * 동일 현상
  * 100 ~ 1000개의 레이어
  * ImageNet detection, ImageNet localization, COCO detection, COCO segmentation 과제에서 1위
  * 결과를 보니 비젼 분야 뿐만 아니라 다른 도메인에서도 효과적일 수 있을 것이라 생각



## 2. Related Work

* Residual Representation
  * VLAD, Fisher Vector: vector quantization(≒ 원하는 값으로의 embedding?)에서는 잔차 벡터를 학습하는 것이 더 쉽다 (*"encoding residual vectors is shown to be more effective than encoding original vectors"*)
  * Multigrid method, hierarchical basis pre-conditioning: 저수준 (low-level) 비젼 및 컴퓨터 그래픽 분야에서는 편미분을 해결하기 위해(=최적화를 위해?) 사용하는 방식으로 residual한 특성을 사용
    * 위 solver들이 타 solver에 비해 훨씬 빨리 수렴
  * 좋은 알고리즘 재구성(reformulation) 또는 전제 조건 설정(preconditioning)은 최적화를 더 간단히 해줄 수 있다는 것을 알 수 있다.
* Shortcut Connections
  * 1990년대 때부터 shortcut connection에 대한 아이디어는 제시되어 왔었음
  * GoogLeNet의 auxiliary classifier, inception layer 內 shortcut branch 등
  * ![auxiliary](https://user-images.githubusercontent.com/38153357/152932986-8fe7d686-0b09-46db-82d9-82e6275e4f17.png)
  * 가장 유사한 concurrent 연구: highway networks (2015)
    * shortcut connections with gating functions
      * data-dependent하며 parameter 존재
      * 0으로 수렴하면 non-residual 함수가 됨
      * 레이어가 극심히 깊어지면 정확도 증진 X
  * ResNet의 identity shortcut
    * 별도의 학습 가능 parameter 추가가 필요 없음
    * 절대 닫히지 않기 때문에 (=0으로 수렴하지 않기 때문에) 항상 모든 정보가 통과되며 추가적인 residual function 학습만 필요하다



## 3. Deep Residual Learning

* Residual Learning

  * ![Fig2](https://user-images.githubusercontent.com/38153357/152932938-6c1f70cc-c318-4a49-bf15-67a3c46d9588.png)
  * weight layer는 우리가 원하는 underlying mapping <img src="https://latex.codecogs.com/svg.image?\mathcal{H}(x)" title="\mathcal{H}(x)" />와 identity mapping <img src="https://latex.codecogs.com/svg.image?x" title="x" />의 잔차 <img src="https://latex.codecogs.com/svg.image?\mathcal{H}(x)-x" title="\mathcal{H}(x)-x" />에 점근적으로 수렴하며(=학습하며), identity mapping은 바로 전달된다.
    * degradation 문제는 identity mapping을 multiple nonlinear layer로 학습 하는 것은 어렵다는 것을 암시한다
  * 전체 학습 목표는 <img src="https://latex.codecogs.com/svg.image?\mathcal{F}(x)&plus;x" title="\mathcal{F}(x)+x" />로 변경된다
    * identity mapping이 이미 optimal 하다면 solver는 residual mapping <img src="https://latex.codecogs.com/svg.image?\mathcal{F}(x)" title="\mathcal{F}(x)" />을 0으로 수렴시킬 것이다.
    * 실제로 identity mapping이 optimal할 경우는 많이 없겠지만 훌륭한 전제 조건으로써 역할을 수행한다.
    * ![Fig7](https://user-images.githubusercontent.com/38153357/152932962-c52997c7-dac8-4b4b-acb4-558633fa0ee2.png)
    * layer response(=레이어를 거친 후의 변화량??)의 표준편차를 살펴본 결과, plain 모델 대비 ResNet 계열의 layer response가 더 낮아 identity mapping 전제 조건 효과가 좋다는 것을 알 수 있다.
      * 우리가 원하는 optimal function <img src="https://latex.codecogs.com/svg.image?\mathcal{H}(x)" title="\mathcal{H}(x)" />이 zero mapping보단 identity mapping에 더 가깝다면, identity mapping을 기반으로 작은 변화량(perturbations)을 학습하는 것이 더 쉬울 것이다.
      * (= 많이 변형시키지 않아도 원하는 feature map에 도달할 수 있다?)

* Identity Mapping by shortcuts

  * 구현 방식: (레이어가 2개인 경우) <img src="https://latex.codecogs.com/svg.image?y&space;=&space;\sigma(W_2&space;\sigma(W_1&space;x)&plus;x)" title="y = \sigma(W_2 \sigma(W_1 x)+x)" />
    * ※ 레이어가 1개인 경우 (<img src="https://latex.codecogs.com/svg.image?W_1&space;x&space;&plus;&space;x" title="W_1 x + x" />)는 shortcut connection의 효과가 없었음
  * shortcut connection은 추가 파라미터 / 추가 계산 복잡도가 필요 없어서 plain counterpart과의 공평하고 합리적인 비교가 가능
  * dimension이 맞지 않는 경우에서의 shortcut connection: <img src="https://latex.codecogs.com/svg.image?\mathcal{F}(x,&space;\{W_i\})&space;&plus;&space;W_s&space;x" title="\mathcal{F}(x, \{W_i\}) + W_s x" />
    * 논문 뒷편에서 3가지 방식으로 실험

* Network Architectures

  * VGGnet의 구현 철학을 많이 계승하였다고 밝힘
    * 3×3 kernel filter
    * 1) *"for the same output feature map size, the layers have the same number of filters"*
    * 2) *"if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer"*
    * downsampling by convolutional layers with stride of 2 (pooling 사용 X) (Figure 3의 점선)
    * global average pooling, fully connected layer with softmax
    * ![Fig3](https://user-images.githubusercontent.com/38153357/152933163-23dedd2b-bc68-4b97-aad3-f9f9d3eeff09.png)
    * 참고: VGG보다 레이어가 깊지만 filter 수가 더 적고, complexity도 더 낮으며, 학습 시간 또한 현저히 낮았다

  * dimension이 증가하는 경우: zero-padding 또는 1x1 convolution을 통한 linear projection 사용
    * (!) 해결 안된 부분: 크기가 줄어드는데 dimension이 증가? channel 축 zero-padding? 

* Implementation

  * ※ AlexNet, VGGnet의 기법들 많이 차용
  * scale augmentation + 224×224 crop
  * random horizontal flip
  * per-pixel mean subtraction (=정규화의 의미?)
  * color augmentation from AlexNet
  * batch normalization after each convolution and before activation
  * PReLUnet의 weight initialization
  * SGD with mini-batch size of 256
  * learning rate 0.1 → divided by 10 when error plateaus (에러가 일정 수준에서 머무를 때)
  * 60 × 10^4 iter
  * weight decay 0.0001 / momentum 0.9
  * dropout X
  * 10-crop testing(=10-fold cross-validation) (≒ AlexNet)
  * multiple scale: 짧은 쪽이 {224, 256, 384, 480, 640}이 되도록 resize + 평균 



## 4. Experiments & Appendix

### ImageNet Classification

* 모델 configuration
  * ![Table1](https://user-images.githubusercontent.com/38153357/152933213-f00a5899-a08e-4a3c-b749-5361e778d5ed.png)
    * 18-34 layer가 같은 구성 // 50-101-152 layer가 같은 구성
* plain 모델 vs ResNet 결과
  * ![Fig4](https://user-images.githubusercontent.com/38153357/152933255-9d6724f8-dda1-4296-b9fc-9d0b2f52bb84.png)
  * ![Table2](https://user-images.githubusercontent.com/38153357/152933320-83c65922-1f61-4233-a05a-19f75f594b82.png)
  * ![Table3](https://user-images.githubusercontent.com/38153357/152933340-afbfd6a4-bc4a-4893-a897-00630308890b.png)
  * plain 모델: 18 layer vs 34 layer
    * 더 깊은 34 layer 모델에서 validation error가 높게 나타남 = degradation 문제
    * vanishing gradient으로 인해 나타나는 문제는 아닌 것으로 주장
      * batch normalization으로 인해 신호들이 0이 아닌 분산값을 갖도록 지속적으로 조절 → forward  propagation 문제 없음
      * *"we also verify that the backward propagated gradients exhibit healthy norms with batch normalization"* → backward  propagation 문제 없음
    * Table 3의 결과 상 plain 모델도 괜찮은 성능 보이는 것을 알 수 있음
    * 깊은 plain 신경망의 경우 지수함수적으로 낮은 수렴률을 가질 수 있다고 추측하며, 이는 학습 오류를 낮추는 것에 타격을 준다
  * ResNet 모델 (zero-padding 사용, 추가 파라미터 X): 18 layer vs 34 layer
    * 34 layer의 성능이 18 layer의 성능보다 더 좋음 (2.8% 차이)
      * 34 layer의 training error가 18 layer에 비해 확연히 낮으며, validation data에 generalizable 가능
      * degradation 문제가 잘 해결되었다
    * 34 layer의 경우 (plain 모델 대비) top-1 error 3.5% 낮아짐
      * residual learning의 효과
    * 18 layer의 경우 (plain 모델 대비) 비슷한 성능을 보이지만 수렴 속도가 훨씬 빨랐음
* Identity vs Projection Shortcuts
  * ![Table3](https://user-images.githubusercontent.com/38153357/152933340-afbfd6a4-bc4a-4893-a897-00630308890b.png)
  * option A vs B vs C
    * option A: "zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameter-free"
    * option B: "projection shortcuts are used for increasing dimensions and other shortcuts are identity"
    * option C: "all shortcuts are projections"
    * A < B : A의 zero-padded dimension으로부터는 아무 residual learning이 없었기 때문인 것으로 주장
    * B < C
    * 하지만 A, B, C 간 적은 차이로 인해 projection shortcut은 필수가 아닌 것으로 결론 (이후 C 사용 X)
* Deeper Bottleneck Architectures
  * 학습 시간을 줄이기 위해 bottleneck 구조 사용
  * ![Fig5](https://user-images.githubusercontent.com/38153357/152933399-ba08a194-e5ba-4d59-9202-d636edfe77d5.png)
  * Figure 5 우측의 identity shortcut이 projection으로 대체된다면 time complexity와 모델 사이즈가 두배가 된다
  * 50 및 101-152 layer ResNet에서 bottleneck 구조 사용
    * 152 layer ResNet도 VGG16/19보다 낮은 복잡도를 가짐
* 18-34 vs 50-101-152 layer ResNet
  * ![Table4](https://user-images.githubusercontent.com/38153357/152933428-a595d118-d8e5-456d-b301-c67bd8a4ed3e.png)
  * degradation 문제 없음

### CIFAR-10 and Analysis

* (ImageNet 때와는 다른 설정 사용, 세부 내용 생략)
* Exploring Over 1000 layers
  * training error를 살펴보았을 때 degradation 또는 최적화 관련 문제 없었음
  * 단, 110 layer ResNet보다 성능이 낮아서 overfitting으로 판단 (파라미터 수 대비 적은 데이터셋을 활용 했기 때문에)

### Object Detection on PASCAL and MS COCO

* Faster R-CNN 방식의 VGG-16, ResNet-50/101 사용



## 주관적 정리 및 참고사항

* oversimplified 3줄 요약
  1. Shortcut Connection은 신경망을 깊게 쌓아도 효율적으로 모델을 학습 시킬 수 있는 Residual Learning 장치
  2. Deep Bottleneck Architecture에 특히 identity shortcut이 효과적
  3. Classification 뿐만 아니라 Localization과 Detection 분야에서 모두 당시 SoTA를 기록하며, 최근 연구에서도 비교 Baseline 모델로 자주 쓰일 정도로 영향력 있는 연구
* 참고자료
  * 라온피플(네이버블로그) blog.naver.com/laonple [(1)](https://blog.naver.com/laonple/220761052425), [(2)](https://blog.naver.com/laonple/220764986252), [(3)](https://blog.naver.com/laonple/220770760226)
  * 만렙개발자(티스토리) [lv99.tistory.com](https://lv99.tistory.com/25)
  * philBaek 개발 일기장(티스토리) [phil-baek.tistory.com](https://phil-baek.tistory.com/entry/ResNet-Deep-Residual-Learning-for-Image-Recognition-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)
