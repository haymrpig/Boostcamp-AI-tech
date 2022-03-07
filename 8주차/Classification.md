# 목차

- [**Computer Vision이란**](#1-computer-vision이란)
  - rendering
- [**데이터와 샘플링된 데이터 간극 줄이기**](#2-데이터와-샘플링된-데이터-간극-줄이기)
  - CNN
  - data augmentation
- [**전이학습**](#3-전이학습)
  - transfer learning
  - fine tuning
  - knowledge distillation
  - semi supervised learning

# 1. Computer vision이란

`representation` : inverse rendering을 통해 만들어진 장면의 대한 정보를 나타낸 자료구조 (high level description) 

`rendering` : 정보를 통해 2D 이미지를 만들어내는 것

![image-20220308000248626](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220308000248626.png)



# 2. 데이터와 샘플링된 데이터 간극 줄이기

- **Data에 대한 오해**

  여러 사람이 수집한 데이터는 분포가 넓게 퍼져있을 것 같지만 사실은 아니다!

  > 특정 여행지에 대한 사진을 모은다고 해보자. 
  >
  > 비록 서로 다른 사람들이 찍은 사진을 모았다고 해도, 그 여행지에서 잘 찍히는 구도가 있을 것이고, 이쁜 사진을 찍기 위해서는 대부분의 사람들이 같은 구도에서 촬영을 할 것이다. 
  >
  > 즉, 이렇게 모은 data들은 bias된 data라고 할 수 있다. 

  사실 세상에 존재하는 모든 데이터를 다 저장하고 있다면, 분류 문제가 아닌 k nearest neighbor을 이용한 검증문제로 바뀌게 된다. 

  하지만 이는 메모리 용량, 속도 문제 그리고, 영상 간의 유사도를 정의해야 하기 때문에 쉽지 않다. 

- **Data augmentation**

  직관적으로 생각하면 fully connected layer에서 weight는 해당 클래스의 이미지와 비슷한 느낌을 갖게 된다. 

  > input으로 들어온 이미지와 weight의 내적 연산이기 때문에!
  >
  > 내적 연산은 두 벡터의 유사도를 의미하기에 input과 weight는 비슷한 느낌을 가진다고 할 수 있을 것 같다. 

  만약 layer가 단순하고, input 이미지에 약간의 변형이 생긴다면 제대로 분류해내지 못할 가능성이 크다.  

  그래서 등장한 것이 CNN이다!!

  fc는 이미지 전체에 대해서 weight 내적 연산을 진행한다면, CNN은 local에서 진행되기 때문에 이미지의 변형이 일어나도 잘 구분해낸다. (overfitting 방지 효과!)

  또한 같은 맥락으로 data augmentation을 통해 data의 분포를 퍼트려주는 것 또한 overfitting을 방지할 수 있다.  

  > 최근에는, autoAugmentation이라고 Google에서 발표한 논문이 있다. 
  >
  > 이 방법은 여러개의 augmentation에 대해 효과가 좋은 augmentation을 찾아준다고 한다.
  >
  > 단점으로는 많은 수의 GPU가 필요하다는 것! 



# 3. 전이 학습

`transfer learning` : 앞단을 freeze, fc layer만 학습시키는 것

`fine tuning` : 이식한 layer들과 fc layer 모두 학습시키는 것

> fine tuning의 경우 데이터가 충분할 경우 유용하다. 

`knowledge distillation` : 큰 모델에서 작은 모델로 지식을 전이하는 것

- **Knowledge distillation**

  - **Label 이 존재하지 않는 경우 (unsupervised learning)**

    teacher model과 비슷하도록 학습

    ![image-20220308003016690](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220308003016690.png)

  - **Label이 존재하는 경우 (supervised learning)**

    teacher model과의 loss와 ground truth와의 loss 모두를 이용하여 학습

    ![image-20220308003040742](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220308003040742.png) 

    - **Distillation loss**

      KL div (soft label, soft prediction에서 사용)

    - **Student loss**

      CrossEntropy (hard label, soft prediction에서 사용)

    - **normal softmax (T=1)**
      $$
      \frac {exp(z_i)}{\sum_j exp(z_j)}
      $$
      기존 softmax의 경우에는 확률의 차이를 극대화한다. 

      ex) softmax(5, 10) = (0.0067, 0.9933)

    - **softmax with temperature (T=t)**
      $$
      \frac {exp(z_i/t)}{\sum_j exp(z_j/t)}
      $$
      차이를 좀 더 부드럽게 만들어준다. 

      ex) softmax(5, 10) = (0.4875, 0.5125)

- **Semi-Supervised Learning**

  Step1. labeling된 데이터로 학습 진행

  Step2. labeling이 안 된 데이터를 모델에 돌려 labeling 진행

  Step3. 위 두 데이터를 모두 학습에 사용

  - **최신 연구 (Self-Training)**

    teacher모델로 pseudo labeling한 데이터와 실제 labeling된 데이터를 RandAugment를 거쳐 student 모델에 넣는다. 

    student모델을 teacher모델로서 위 과정을 반복한다. 

    (student 모델은 계속 커져야 한다.)

    ![image-20220308003913399](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220308003913399.png)

​	