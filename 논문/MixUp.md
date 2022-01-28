# 목차

- [**출현 배경 & 효과**](#1-출현-배경) 
- [**algorithm & expression**](#2-algorithm-expression)
- [**Results**](#3-results)
- [**그 외**](#4-그-외)
  - 다양한 방법의 MixUp

# 1. 출현 배경

크고 깊은 모델의 경우 powerful하지만 memorization과 같이 의도하지 않은 부작용을 가지고 있으며, 적대적 이미지들에 대해서 강건하지 못하다는 단점을 가지고 있다. 

이러한 문제들을 해결하기 위해서 이미지와 label을 convex combination 형태로 적용한  MixUp 기법이 등장하였다. 

기존의 ERM 방식은 모델이 복잡하고 데이터의 양도 비슷하게 많은 곳에서 사용했었지만, 아래의 단점 때문에 모순이 발생한다. 

이러한 ERM기반의 방식을 VRM기반으로 변경하여 만든 것이 MixUp이다. 

- **ERM (empirical risk minimization)**

  - 수식

    실제 분포를 정확히 알 수 없기 때문에, 경험적인 즉, sampling을 통해서 loss를 최소화하는 방식으로 찾아야 한다. 

    ![image-20220128141359944](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128141359944.png)
    $$
    l : loss\; function,\; P(X,Y) : joint\; distribution
    $$
    여기서 우리는, risk를 최소화할 수 있는 f를 찾아야 한다. 

    ![image-20220128141412968](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128141412968.png)
    $$
    x=x_i,\; y=y_i,\; 즉\; input과 \;정답이 \;일치했을 \;경우\; 1인\; P를\; 정의할\; 수 \;있다.\\
    (P : empirical \;distribution)
    $$
    그러면 위 두 수식을 합치면 아래 식이 나오게 된다. 

    ![image-20220128141422360](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128141422360.png)

    하지만, 만약 parameter의 수가 데이터의 수랑 비슷하거나 더 많게 된다면, loss를 최소화하는 

    할 수 있는 값은 training data를 memorize하는 trivial solution을 구할 수 밖에 없다. 

    ( 이 부분에 대해서는 개인적인 의견으로는 feature의 수가 데이터보다 많을 경우 non-trivial solution이 나오는 선형종속으로 생각할 수 있을 것 같다. 이와 같은 맥락으로 parameter가 많을 경우, non-trivial solution이 나오기 때문에 loss를 최소화하는 parameter들을 trivial solution으로 압축하기 위해서는 training dataset에 overfitting하는 방법밖에 없다는 의미라고 생각한다)

  - 단점

    - 모델 parameter가 데이터 수에 따라 증가하면 수렴을 보장받지 못한다. 
    - memorization이 심하다

- **VRM (vicinal risk minimization)**

  - 서로 다른 데이터들과 라벨들을 섞으면서 가능해졌다. 

- **효과**
  - regularize를 통한 generalization performance의 향상
  - 손상된 label들에 대한 memorization을 줄이며, 적대적 이미지에 강건하다. 
  - GAN (generative adversarial network)의 학습을 안정화시킨다. 



# 2. Algorithm & Expression

- **ERM -> VRM**

  위의 설명에서의 ERM 수식은 overfitting의 문제점이 발생하기에 여기에 VRM의 개념을 추가한다. 

  VRM이란 vicinal risk minimization으로 아래 수식을 보면 

  ![image-20220128142412311](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128142412311.png)

  여기에서 v란

  ![image-20220128142433769](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128142433769.png)

  이고, 이 수식이 의미하는 바는 training data에 gaussian noise를 더하는 것과 같다. 

  이렇게 P를 단순히 정확한 정답에만 의존하는 것이 아닌, 가능한 확률로서 생각하는 것이다. 즉, overfitting을 방지할 수 있다는 의미가 된다. 

  이후, risk에 관한 수식을 수정하게 되면

  ![image-20220128142609681](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128142609681.png)

  이와 같이 나오며, 아래와 같은 mixup, 즉, generic vicinal distribution 수식이 나오게 된다. 

  ![image-20220128142617804](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128142617804.png)

  위 식을 이용해서 실제 데이터와 라벨을 아래 수식으로 표현하게 된다. 

  ![image-20220128142916013](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128142916013.png)

  람다 값은 베타 분포를 따른다. 

- **Algorithm**

  ![image-20220128143027902](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128143027902.png)

# 3. Results

#### ResNet-50, 101에서 ERM방식과의 비교 (ImageNet-2012)

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128143242718.png" alt="image-20220128143242718" style="zoom:67%;" />



#### PreAct ResNet-18 & WideResNet-28-10&DenseNet-BC-190 (CIFAR-10, 100)

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128143401999.png" alt="image-20220128143401999" style="zoom:67%;" />



#### LeNet, VGG-11 (Google commands dataset)

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128143458863.png" alt="image-20220128143458863" style="zoom:67%;" />



#### Corrupted label & adversarial 

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128143549816.png" alt="image-20220128143549816" style="zoom:67%;" />



#### UCI datasets

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128143655357.png" alt="image-20220128143655357" style="zoom:67%;" />



#### GAN

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128143735817.png" alt="image-20220128143735817" style="zoom:67%;" />

- MixUp이 판별자의 gradient 정규화 역할을 수행하여 training 안정화 진행

- 판별자의 smoothness를 통해 생성자에 안정적인 gradient를 보장

  (vanishing gradient 해결)



# 4. 그 외

#### 다양한 방법의 MixUp

- feature map 수준의 MixUp

- 같은 class의 input MixUp

- 다른 class의 input을 MixUp하지만, label은 covex combination의 더 큰 weight의 이미지 label 선택

- label smooting

  <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220128144722366.png" alt="image-20220128144722366" style="zoom:67%;" />

  - weight decay (weight의 값이 증가하는 것을 제한하는 방법) 10^-4일 때는, 초기에 기획했던 방법의 성능이 가장 우수하다.
  - weight decay 5*10^-4일 때는 layer1에서 MixUp을 한 것이 더 우수했다. 
  - 두 결과 모두 input에 가까운 data augmentation이 우수하게 나오는 것을 의미한다. 

