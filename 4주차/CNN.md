# 목차

1. [**Convolution 연산**](#1-convolution-연산)
   - Stride
   - Padding
   - Max Pooling
   - Dilated convolution
2. [**Back Propagation**](#2-back-propagation)
3. [**Cost Function**](#3-cost-function)
4. [**Optimization 용어 정리**](#4-optimization)
5. [**Optimizer의 종류**](#5-optimizer의-종류)
   - Gradient Descent
     - mini batch gradient descent
     - batch gradient descent
   - Momentum
   - Nesterov accelerated gradient
   - Adagrad
   - Adadelta
   - RMSprop
   - Adam
6. [**Activation function**](#6-activation-function)
   - Sigmoid
   - Softmax
   - ReLU
   - Leaky ReLU
   - Maxout
   - tanh
7. [**Regularization**](#7-regularization)
8. [**History**](#8-history)
9. [**CV Applications**](9-cv-applications)

# 1. Convolution 연산

- **정의**

  고정된 크기의 kernel을 입력 데이터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조로 신호를 kernel을 이용해 국소적으로 증폭 또는 감소시켜 정보를 추출/필터링하는 작업이다. 

  - 장점

    데이터 입력에 따라 kernel의 크기가 변하지 않는다. 

![image](https://user-images.githubusercontent.com/71866756/144757713-86bded0b-92fc-4224-b5bd-12822504ed28.png)

- **출력 크기 계산**

  - 입력 크기 : (H,W) , 커널 크기 : (Kh, Kw), 출력 크기 : (Oh, Ow)

    ![image](https://user-images.githubusercontent.com/71866756/144757738-3cb0c1b2-71eb-40dd-8778-526cc9148cbc.png)

  - 채널이 여러개인 경우 커널의 채널 수 = 입력의 채널 수!!

    ![image](https://user-images.githubusercontent.com/71866756/144757755-e516f7a7-0f6e-439d-96e9-22d55ae4106a.png)

    - 출력은 커널개수만큼 나온다. 

- **용어 정리**

  - **Stride**

    커널을 입력 데이터에 convolution연산할 시 몇 칸씩 뛰어넘을지를 의미

    ![image](https://user-images.githubusercontent.com/71866756/144757767-9ba785e7-1ee1-47ac-a616-d5390fb68b2d.png)

  - **Padding**

    출력 크기 계산 수식을 이용하면 출력의 크기는 입력보다 작아지게 된다. 따라서 이 경우는 Padding이 없는 경우이며, 만약 padding이 있는 경우에는 입력의 최외곽 일정 크기만큼을 일정한 수로 채우는 것을 의미한다. 

    ![image](https://user-images.githubusercontent.com/71866756/144823796-76bd6338-2be4-46ad-9d8d-74a0dac640bd.png)


  - Example

    ![image](https://user-images.githubusercontent.com/71866756/144757798-477e4037-fb61-4bca-8953-0546454a4abf.png)
    
  - **Max Pooling**
  
    input의 크기가 클 경우 max pooling을 이용하여 parameter들을 효과적으로 줄여줄 수 있다. 
    
    ![image](https://user-images.githubusercontent.com/71866756/144823862-02085478-618f-47a0-bc57-3316e8c77754.png)
    
  - **Dilated convolution**
  
    해당 값이 1일 경우 일반적인 convolution과 같지만,  2 이상일 경우 kernel 사이 간격을 조정하여 convolution 연산을 실행한다. 예를 들어 저해상도에서 고해상도로 만들 때 사용하기도 한다. 
    
    ![image](https://user-images.githubusercontent.com/71866756/144823914-e36ae6e1-80f8-402c-8196-d6c43a0c97dd.png)
    
    
    
    



# 2. Back Propagation

chain rule을 이용하여 backward로 각각의 parameter들이 cost에 미치는 영향을 계산하는 방식

- **Chain rule**

  `df/dx = dg/dx * df/dg`

  

  

# 3. Cost Function

- **Regression Task**

  - MSE( Mean Square Error )

  ![image](https://user-images.githubusercontent.com/71866756/144757811-f1036d9a-8ecf-4a12-8cd0-56c4fe9d7f4c.png)

- **Classification Task**

  - CE( Cross-Entropy )

    loss를 최대한 정답 레이블에 맞춰서 계산하기 위한 방법이다. one-hot coding에서 정답 레이블이 아닌 다른 레이블의 값은 0이므로 아래 식에서 0이 곱해져서 나머지 정답이 아닌 레이블에 대한 error는 계산되지 않게 된다. 

    또한 log이기 때문에 차이가 클수록 error가 더욱 커지고, 차이가 작을수록 error는 더욱 작아진다. 

  ![image](https://user-images.githubusercontent.com/71866756/144757823-b31b23b9-be35-4809-bd71-d7ea1e763768.png)

- **Probabilistic Task**

  - MLE( Maximum Likelihood Estimation)

  

  ![image](https://user-images.githubusercontent.com/71866756/144757835-eb608cb0-053a-4803-bcaa-2b6579126082.png)

  

# 4. Optimization

- **Parameter, Hyper parameter**

  - Parameter

    최적해에서 찾고 싶은 값으로 weight, bias 등이 있다. 

  - Hyper parameter

    사용자가 지정하는 값으로 learning rate 등이 있다. 

- **Generalization**

  `generalization gap` : trainig error와 test error사이의 차이

  > Generalization Performance가 좋다라는 것은 학습한 결과가 test data에서도 잘 동작하는 것
  
  

![image](https://user-images.githubusercontent.com/71866756/144757915-aa963733-a769-4b35-9fe5-7a62a174534e.png)

- **Underfitting & Overfitting**

  - Underfitting

    학습결과가 학습 데이터의 경향을 잘 표현하지 못하는 것 

  - Overfitting

    학습결과가 학습 데이터에 너무 치중되어 나타난 것으로 test data로 모델을 돌렸을 시 결과가 좋지 못할 가능성이 높다. 

- **Cross-validation**

  K fold validation과 같은 의미

  해당 모델의 Generalization Performance를 높이기 위한 방법으로 하나의 데이터셋을 train data와 validation data, test data로 구분하는 것을 의미한다. 

- **Bias & Variance**

  ![image](https://user-images.githubusercontent.com/71866756/144757934-01852996-8f87-4edb-a66b-d0c1680bc3cf.png)

  `Variance` : variance가 클 경우, 입력마다 출력이 매우 다르게 분포하므로 overfitting의 문제점이 있다. 

  `Bias` : bias가 낮다는 것은 비슷한 입력들에 대해서 평균이 원하는 결과와 비슷한 경우를 의미한다. 

  - **Bias와 Variance는 Tradeoff 관계이다.** 

    cost function의 경우 bias, variance, noise를 포함하고 있기 때문에 bias가 낮은 경우 그에 따라 자연스럽게 variance는 높을 수 밖에 없다. 그 반대 역시 마찬가지이다. 

    ![image](https://user-images.githubusercontent.com/71866756/144757964-03569ce3-a75b-4335-8f19-028212382fb1.png)

- **Bootstrapping**

  학습 데이터가 주어졌을 때, 그 학습데이터를 여러개의 sub data로 나누어 여러 모델을 만드는 것을 의미한다. 

  - **Bagging( bootstrapping aggregating )**

    bootstrapping을 통해 학습한 여러 모델들의 output의 평균을 내는 것으로 보통 한개의 모델을 이용하는 것보다 이런 모델들의 output의 평균을 이용하는 것이 더 좋은 경우가 많다. 

  - **boosting**

    한 모델을 학습할 때, 결과가 제대로 나오지 않는 데이터셋에 대해서 또 다른 모델을 만들고, 여러개의 모델을 sequence로 연결하여 강한 모델을 만드는 방법

  ![image](https://user-images.githubusercontent.com/71866756/144757995-308a5a0c-0237-4bb0-8599-e0dae2912515.png)

  

# 5. Optimizer의 종류

- **Gradient Descent**

  하나의 샘플 데이터로 gradient를 업데이트하는 것

  ![image](https://user-images.githubusercontent.com/71866756/144758090-898246d5-30a7-4a8a-9397-a0c8059eba0a.png)

  - **mini-batch gradient descent**

    전체 데이터가 아닌, 그보다 작은 batch 사이즈로 gradient를 업데이트 하는 것

  - **batch gradient descent**

    데이터 전체를 이용하여 한번에 업데이트 하는 것

  - **batch 사이즈에 따른 특징**

    batch 사이즈가 매우 큰 경우, 오른쪽 그림처럼 sharp minimum이 된다. 즉, 같은 위치에서 training한 결과는 minimum이지만, test한 결과가 매우 높은 값을 나타낼 수 있다. 

    batch 사이즈가 작은 경우, 왼쪽 그림처럼 gradient차이가 크지 않기 때문에 training에서의 좋은 결과는 test에서도 좋은 결과로 나올 가능성이 높다. 
    
    -> training과 test의 분포의 차이 ( Internal Covariate Shift )

  ![image](https://user-images.githubusercontent.com/71866756/144758555-7c761519-85e1-4849-94f1-c6ff398f8e4e.png)

- **Momentum**

  베타값이 들어가서 현재 weight를 이전 batch size만큼 학습했을 때의 weight에서 특정값을 곱하여 뺀다. 

  즉, 이전 batch의 gradient를 어느정도 유지하여 gradient의 변화 폭이 커도 어느정도 학습이 잘 된다는 장점이 있다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758124-980e7e0f-9408-4316-91bf-c7f6e2b1cb2c.png)

- **Nesterov Accelerated Gradient**

  Momentum과 비슷하지만, lookahead로 한번 더 이동하여 그 위치에서 계산한 값을 현재 계산에 넣어주어 좀 더 빠르게 minimum을 찾을 수 있다.

  수렴하는 속도가 빠르다. (이론적으로 증명이 가능하다.)

  ![image](https://user-images.githubusercontent.com/71866756/144758145-20c7dfff-1933-4541-80ab-b080f64496d6.png)

- **Adagrad**

  값이 많이 변한 parameter들에 대해서는 적게 변화시키고, 값이 적게 변한 parameter들에 대해서는 많이 변화시키는 방법이다. 

  이전 parameter들의 변화를 기록하고 있어야 하며, 이 값은 학습이 진행될 수록 축적이 되어 G_t가 커지기 때문에 오랜 시간 학습을 진행할 경우 분모가 무한히 커져 전체적인 값이 0에 가까워진다. 

  즉, 학습이 오랫동안 지속할 경우, 변화가 점점 작아지고 학습이 안될 가능성이 있다.  

  ![image](https://user-images.githubusercontent.com/71866756/144758151-bf70d5ab-fc1f-4a48-a92e-ef83dcf8469b.png)

- **Adadelta**

  Adagrad의 문제를 해결하기 위해서 window값을 지정해준다. 

  즉, Adagrad처럼 값이 축적되어 커지진 않고, 만약 parameter의 개수가 클 경우에 메모리 문제가 발생할 수 있으니 감마를 이용하여 어느정도 완화시켜준다. 

  Adadelta의 경우 learning rate가 없다는 단점이 있다. 

  - **EMA ( Exponential Moving Average )**

    `EMA`는 이동평균법으로 평균과의 가장 큰 차이점은 시간이라는 개념이다. 
    
    평균은 같은 시간대에서 산출되는 것이 흔한 반면, 이동평균은 동일대상을 서로 다른 시점에서 구한다는 점이 차이점이다. 
    
    moving average filter를 생각하면 이해가 빠르다. 
    
    H_t는 weight의 변화량을 의미하는데, 이 값을 넣어주기 때문에, learning rate가 없는 것이다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758225-35daeabe-4e5c-4920-b7aa-540e8567b1bc.png)

- **RMSprop**

  Adadelta에 step size, 즉 learning rate을 추가한 방법이다. (실험적으로 증명되었다.)

  ![image](https://user-images.githubusercontent.com/71866756/144758204-610fe652-0d4b-41b7-aa22-1877989adb27.png)

- **Adam**

  현재 가장 무난하게 사용되고 있는 방법으로 Momentum과 EMA( 이동 평균법 )을 합친 방법이다. 

  입실론은 0으로 나눠지는 것을 막기 위한 요소로 Adam의 가장 중요한 요소이다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758181-b5cfac99-18e4-4da7-83bb-1fc6916fcc86.png)

  

# 6. Activation function

- **Sigmoid**

  sigmoid의 경우 결과를 0~1사이의 값으로 변경해준다. 나온 결과들의 합이 1이 되지 않아도 되기 때문에 multi-label classification task에서 종종  사용한다. 

  - sigmoid + cross-entropy를 BinaryCrossEntropy라고 부른다. 

    ![image](https://user-images.githubusercontent.com/71866756/145266182-42b5f15d-a362-4bf4-8ee7-01233ef2e50b.png)

- **Softmax**

  softmax의 경우 나온 결과를 0~1로 변환하고, 나온 결과들의 합이 1이 되도록 하기 때문에 전체 합을 해당결과에서 나누어 확률을 구한다. multi-class classification task에서 종종 사용한다. ( 정답 label이 하나이기 때문에 전체 합이 1이 될 수 있다. )

  - softmax + cross-entropy를 CategoricalCrossEntropy 라고 부른다. 

- **ReLU**

  backpropagation을 통해 계산을 하면 훨씬 이전 layer에서의 gradient가 너무 작아지는 문제가 발생하여 학습이 제대로 진행되지 않는다. ( softmax를 적용하였을 경우, 각각의 parameter들이 0~1로 바뀌기 때문에 )

  ![image](https://user-images.githubusercontent.com/71866756/145266139-6f260a58-8a59-4121-8bce-129aa7088a54.png)

  위 문제를 해결하기 위해서 ReLU가 사용되었다. ReLU의 경우 값이 x에 따라 결정되기 때문에 vanishing gradient문제가 발생하지 않는다. 

  단점으로는 0이 많아지기 때문에 일부 뉴런이 죽을 수가 있다. ( Dying ReLU ) 

  ![image](https://user-images.githubusercontent.com/71866756/145266266-bf364bd1-57c9-4e29-89ad-20d5c2e986fb.png)

- **Leaky ReLU**

  ReLU의 Dying ReLU현상을 막기위해 만들어졌다. 

  ![image](https://user-images.githubusercontent.com/71866756/145266308-101cdf76-d080-419c-9e75-70770565001d.png)

- **Maxout**

  ReLU의 장점을 모두 다 가지며, Dying ReLU문제 또한 해결할 수 있지만, 계산이 복잡하고 양이 많다는 단점이 있다. 

  ![image](https://user-images.githubusercontent.com/71866756/145267695-69e3bb94-87b8-41ed-8203-d8e923c2a79f.png)

- **tanh**

  sigmoid와 유사한 형태이지만 그래프의 위치와 기울기가 다르다. 입력신호를 -1~1로 normalization한다. 하지만 이 역시 미분값이 0이 나오는 구간이 존재하므로 vanishing gradient를 해결할 순 없다. 

  ![image](https://user-images.githubusercontent.com/71866756/145267757-12b13bac-f125-405a-ae88-98accd6dd605.png)
  
  
  
  
  
  

# 7. Regularization

overfitting을 방지하기 위한 여러가지 기법들을 의미한다. 

- **Early Stopping**

  validation error가 커지기 시작하는 시점에서의 학습을 중단하는 것을 의미한다. 

- **Parameter Norm Penalty**

  전체적인 Parameter의 값을 낮춘다는 의미이다. 
  $$
  total\; cost = loss(D;W) +\frac{\alpha}{2}||W||_2^2
  $$
  전체적인 parameter의 값 (절대값)이 작을수록, model의 function space가 더 부드러질 것이며, 부드러운 function일수록 generalize가 잘 될 것이다라는 가정을 내포하고 있다. 

- **Data Augmentation**
  - Noise Robustness (add random noise inputs or weights)
  - label smoothing
- **Batch Normalization**
  - batch norm
  - layer norm
  - Instance norm
  - group norm


- **dropout**

  layer에서 몇몇 parameter들이 다음 layer로 가는 것을 막는다는 아이디어이다.

  train에서는 dropout을 사용하지만, eval에서는 사용하면 안된다. 

- **Ensemble**

  bagging이 앙상블에 해당한다. 여러 모델을 만들고 최종 결과에 모아 성능을 향상시키는 기법이다. 

- **Weight Initialization**

  weight를 0으로 초기화할 경우, backpropagation 진행 시 gradient가 0으로 학습이 되지 않는다.

  - **RBM( Restricted Boltzmann Machine )**

    - Restricted : 같은 layer에서의 연결은 없다는 의미 ( 다른 layer와는 fully connected )

      ![image](https://user-images.githubusercontent.com/71866756/146974137-41c3f95e-71c5-4152-9cbe-0ff83e454dee.png)

> - RBM 동작 방식
>
>   1. 첫 layer input x가 들어왔을 때 학습을 통해 출력 y가 결정되고, 출력 y로 다시 x를 복원할 수 있도록 학습을 시켜 weight를 setting한다. ( pre-training 단계에서 RBM을 몇번을 학습해야 하는지는 연구가 되었지만, 정확하게 나온 결과는 없다. )
>   2. setting된 weight는 fix한다.
>   3. 다음 layer도 똑같은 방식으로 진행한다.
>   4. RBM을 통해 tuning된 weight로 back propagation을 통해 기존 방식처럼 학습을 진행하는 것을 Fine-tuning이라고 한다. 
>
>   ![image](https://user-images.githubusercontent.com/71866756/146974246-fbec33c8-fe90-4964-b987-ca08c0342d97.png)
>
>   -> 요즘에는 잘 사용하지 않는 방식이다. ( 계산 방식이 복잡하기 때문에 )

  - **Xavier**

    layer의 특성에 따라 weight를 초기화하는 방식

    - Xavier Normal Initialization

      평균과 표준편차를 아래 그림처럼 사용하는 방식( nin : layer input 수, nout : layer output 수 )

      ![image](https://user-images.githubusercontent.com/71866756/146974290-868b7cf8-b94f-43e0-8c09-2fe00d14917f.png)

    - Xavier Uniform Initialization

      ![image](https://user-images.githubusercontent.com/71866756/146974381-ed4d6a58-cf85-4b9e-a3b0-9ad473e9ff29.png)

  - **He Initialization**

    마찬가지로 layer의 특성에 따라 weight를 초기화하는 방식

    - He Normal Initialization

      ![image](https://user-images.githubusercontent.com/71866756/146974429-7a17a0f1-966f-4dc8-8a03-c7937dee6a6a.png)

    - He Uniform Initialization

      ![image](https://user-images.githubusercontent.com/71866756/146974476-7f6157ea-a66f-4578-9c5c-3d1babea475e.png)

- **Batch Normalization**

  activation function 변경, careful initialization, small learning rate 등은 gardient vanishing / exploding을 해결할 수 있는 간접적인 방식이다. 

  batch normalization 또한 이 문제를 해결하기 위한 방식으로 좀 더 직접적인 방식이라고 할 수 있다. 

  각 layer마다 normalization을 진행하는데, 학습 시 사용하는 mini batch마다 normalization을 하기 때문에 batch normalization이란 불린다. 

  ![image](https://user-images.githubusercontent.com/71866756/146974512-8a85f5d6-5f44-4a0b-95c9-95f539ee5f3e.png)
  $$
  \epsilon : 분모를 0이 되지 않게 하기 위한 아주 작은 값\\
  \beta, \gamma : batch normalization을 계속 진행할 경우,\\ activation function의 nonlinearty를 잃게 될 수 있기 때문에 추가해주는 값( 학습 parameter)
  $$
  !!! 주의 사항 !!!

  - training 과정에서 구한 mini-batch mean( sample mean )과 mini-batch variance( sample variance )는 데이터에 따라 달라지기 때문에, test시 그대로 이용하는 경우 알맞은 값이 아닐 가능성이 높아지게 된다. 
  - 따라서 학습시의 모든 sample mean & sample variance를 따로 저장하여 test시에는 이 값들의 평균으로 구한 learning mean & learning variance를 사용한다. ( 고정된 값을 사용한다. )



# 8. History

2012 - AlexNet

2013 - DQN

2014 - Encoder/Decoder, Adam

2015 - GAN, ResNet

2017 - Transformer

2018 - Bert

2019 - Big Language Models(GPT-X)

2020 - Self-Supervised Learning(SimCLR)

+SPPNet+Fast R-CNN

# 9. CV Applications

### 9-1. Semantic Segmentation

- **Convolutionalization**

  input -> flatten -> dense -> output 을

  input -> Conv -> output으로 변경하는 것을 의미한다. 

  >input : 16x4x4
  >
  >output : 10x1x1
  >
  >1. flatten 후, dense후 output으로 내보내기 위한 파라미터는 16x4x4x10 = 2560
  >
  >2. 4x4 conv를 이용하는 경우 파라미터는 16x4x4x10 = 2560
  >
  >위 두 방법의 parameter의 개수는 똑같다. 

  그렇다면 Fully Convolutional Network를 왜 쓰냐?

  바로, **input 이미지의 크기에 상관없이 output을 뽑아낼 수 있으며, semantic segmentation의 경우, heatmap으로 결과를 내보낼 수 있기 때문**이다. 

  fully connected layer의 경우 reshape이 들어가기 때문에, input 크기가 달라지면 모델을 수정하거나, 데이터를 resize를 해야 한다. 

  하지만, fully convolutional network는 input 사이즈가 달라져도 output사이즈의 변화가 있을 뿐, 모델을 수정할 필요가 없다. 

- **Deconvolution (conv transpose)**

  convolution의 엄밀히 말하면 역연산은 아니지만, 비슷한 동작을 한다.

  (parameter 계산 방법은 똑같다.) 



### 9-2. Detection

- **R-CNN**

  이미지에서 bbox을 여러개 임의로 뽑아낸다. -> region에 대한 feature를 CNN을 통해 얻은 후 SVM으로 분류 진행

  - 문제점

    뽑아낸 bbox에 대해서 CNN을 다 통과시켜야 하므로, 오래 걸린다. 

    즉, 2000개의 bbox를 뽑아냈으면, 모델을 2000번 돌리는 것과 마찬가지이다. 

- **SPPNet**

  이미지에서 여러개의 bbox를 뽑되, conv연산은 전체 이미지에 대해서 적용하고, feature map에서 원래 bbox의 위치의 tensor값들을 뽑아온다는 idea

  따라서, Conv을 한번 돌리기 때문에, R-CNN보다 훨씬 빠르다. 

- **Fast R-CNN**

- **Faster R-CNN**

  - **Region proposal**

    `anchor box` : 미리 정해놓은 bbox (고정된 크기의 bbox들)

    `anchor box`에 object가 있을 확률을 계산한다. 

  - **9-4-2**

    9 : (128, 256, 512) 사이즈의 bbox와, 가로:세로 = (1:1, 1:2, 2:1)로 총 3x3=9가지의 anchor box를 이용한다. 

    4 : x길이, y길이, x offset, y offset의 총 네가지의 변화를 줄 숭수 있는 parameter를 정의

    2 : 해당 bbox가 쓸모있는지/없는지 판단하는 2 parameter

    따라서, fully convolutional layer가 9*(4+2)의 output을 가지게 된다. 

- **YOLO**

  Region Proposal 없이 바로 bbox를 예측한다. 

  그렇기에 Faster R-CNN보다 빠르다. 

  











