# 목차

- [**Conditional Generative Model**](#1-conditional-generative-model)
- [**Super Resolution**](#2-super-resolution)
- [**Image Translation GAN**](#3-image-translation-gan)
- [**다양한 GAN의 사용**](#4-다양한-gan의-사용)

# 1. Conditional Generative Model

`Conditional Generative Model` : 일반적인 GM과 다르게 특정 condition을 주고, condition 기반으로 확률을 구한다. 

> 예제1. 가방의 스케치가 주어진 상태에서 가방 이미지 생성
>
> 예제2. 저퀄리티 오디오가 주어진 상태에서 고퀄리티로 super resolution
>
> 예제3. 외국어가 주어진 상태에서 다른 언어로 변환
>
> 예제4. 기사 제목이 주어진 상태에서 기사 내용 생성

- **Generative Model의 구조**

  - Discriminator

    input이 실제인지 아닌지 판별

  - Generator

    실제와 같은 output을 생성

# 2. Super resolution

`super resolution` : 저해상도 이미지 -> 고해상도 이미지를 생성해내는 것

![image-20220315110435952](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220315110435952.png)

#### 2-1. Naive Regression Model

MAE, MSE loss를 이용한 모델의 경우, output이 만족스러울 만큼 sharp하지 못한다. 

그 이유는 아래 그림으로 설명 가능하다. 

![image-20220315120835954](../../../../AppData/Roaming/Typora/typora-user-images/image-20220315120835954.png)

빨간 box들로 이루어진 plane을 대상 이미지의 manifold라고 생각해보자. 

그렇다면 MAE와 MSE로는 평균 loss가 작은 것을 찾을 것이다. 

그렇다면 Manifold상의 특정 이미지와 매칭되는 것이 아닌, 모든 이미지들과의 loss중 작은 곳을 찾게 될 것이다. 그렇게 되면, 실제 이미지처럼 뚜렷한 결과를 얻을 수 없이 애매한 결과만 나올 것이다. 

> 예를 들어, MSE 기반의 결과가 1번 위치라고 한다면, 2번과의 loss는 매우 클 것이기 때문에
>
> 1번 위치는 선택될 수 없다. 
>
> 즉, 모든 manifold상의 이미지에서 조금씩 가까운 가운데가 결과로 나올 것!

#### 2-2. Super Resolution GAN

naive regression model과 달리, discriminator는 실제 이미지와 가짜를 판별하기 때문에, manifold상의 특정 이미지와 비슷하기만 해도 loss는 매우 낮게 측정될 것이다. 

그렇기 때문에, 실제 이미지에 가깝게 뚜렷한 이미지를 생성해 낼 수 있는 것이다. 



# 3. Image Translation GAN

#### 3-1. Pix2Pix

`pix2pix` : input 이미지를 다른 domain상의 이미지로 변환하는 모델을 의미한다. 

![image-20220315121343262](../../../../AppData/Roaming/Typora/typora-user-images/image-20220315121343262.png)

- **Loss Function**

  ![image-20220315121806681](../../../../AppData/Roaming/Typora/typora-user-images/image-20220315121806681.png)

  - **GAN loss**

    실제와 비슷한 sharp한 이미지를 생성할 수 있도록 함

  - **L1 loss**

    되고자 하는 y와 비슷해지도록 + GAN의 학습을 안정화시키는 역할

#### 3-2. CycleGAN

`CycleGAN` : 기존의 이미지에서 이미지로의 변환은 일치하는 pair가 필요했지만, CycleGAN에서는 일치하는 pair없이 training이 가능하다. (분포를 닮도록)

- **Loss Fucntion**

  ![image-20220315135045353](../../../../AppData/Roaming/Typora/typora-user-images/image-20220315135045353.png)

  - **GAN Loss** : 이미지를 A->B, B->A domain으로 바꾸도록 만드는 loss

  - **Cycle-consistency loss** : 이미지와 변형된 이미지가 서로 바뀌어도 같게 만드는 loss

    > Cycle-consistency loss가 없는 경우, CycleGAN은 pair로 학습하는 것이 아니기 때문에,
    >
    > discriminator는 해당 결과 분포만 닮도록 학습하면 된다. 
    >
    > 따라서, 다른 이미지를 넣더라도 같은 결과 이미지가 나올 수가 있기 때문에, 
    >
    > Cycle-consistency loss로 변형한 결과를 다시 원본으로 복원하는 loss도 필요하다. 

#### 3-3. Perceptual loss

GAN의 경우 `alternating training (discriminator와 generator를 번갈아 학습시키는 것)`이 필요하며 학습이 어렵다. 

GAN없이 고해상도 이미지를 얻을 수 있는 방법은 `perceptual loss`를 사용하는 것이다. 

- **Perceptual loss는 train이 쉽고, 코드가 간단하다.**
- **Perceptual loss는 pre-trained network를 필요로 한다.** 

- **구조**

  ![image-20220315145140087](../../../../AppData/Roaming/Typora/typora-user-images/image-20220315145140087.png)

  - Image Transform Net은 training동안 고정된다. 

- **Loss function**

  - **Feature reconstruction loss**

    target과 transformed image를 feature level에서 L2 loss를 구한다. 

  - **Sytle reconstruction loss**

    target과 transformed image를 통해 feature map과`Gram matrices` 구하고, L2 loss를 구한다. 

    > Gram matrices란 간단하게 설명하면, 
    >
    > 두 image feature map을 내적하여 channel * channel matrix를 만들고, 공간의 통계적 특성을 나타내도록 만드는 것이다. 

- **보통 Style이 필요없는 Super resolution의 경우 Feature reconstruction loss만 사용하기도 한다.** 



# 4. 다양한 GAN의 활용

- deepfake
- Face de-identification
- Face anonymization with passcode
- Video translation