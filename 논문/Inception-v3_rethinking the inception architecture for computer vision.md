# 목차

- **등장 배경**
- **사용된 기법들**
  - Factorizing convolutions with large filter size
  - efficient grid size reduction



# 등장 배경

모델이 깊고 복잡해질수록 성능이 좋아지는 것은 당연하지만, 제한된 resource 환경에서 효율은 매우 중요하다. 따라서 이 논문에서는 모델을 효율적으로 구성하는 기법들에 대해 소개를 하였으며, 그러한 기법들을 이용한 Inception-v3 모델을 제시하였다. 



# 사용된 기법들

### 1. Factorizing Convolutions with Large Filter size

**가장 처음에 소개되는 기법은 conv filter의 분해이다.** 

**conv filter의 분해로 취할 수 있는 이득은 크게 두가지 이다. **

- **parameter 개수 줄이기**
- **비선형성 증가**



#### 1-1. parameter 개수 줄이기

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220204153149496.png" alt="image-20220204153149496" style="zoom:67%;" />

위의 이미지에서 처럼 5x5 filter를 3x3 filter 두개로 대체할 수 있다. 

>결과적으로는 하나의 픽셀이 나오기 위해서, 5x5의 경우 25개의 parameter가 필요하지만, 3x3의 경우 3 * 3 * 2개의 parameter만이 필요하기 때문에 25/9 배 만큼의 computational 이득을 취할 수 있다. 

만약, 3x3보다 더 작은 filter로 대체한다고 생각해보자. 

3x3을 마찬가지로 2x2로 줄일 수가 있다. 하지만, 실험 결과 2x2보다는 1x3, 3x1로 줄이는 것이 더 효과가 좋았다.

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220204154054450.png" alt="image-20220204154054450" style="zoom:67%;" />

위 그림처럼 nxn filter를 nx1로 쪼갤 수 있지만, early layer에서의 적용은 바람직하지 않았고, n이 12~20 사이일 때 효과가 가장 좋았다. 

![image-20220204154701531](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220204154701531.png)

위의 그림처럼 분해가 가능하다. 



#### 1-2. 비선형성 증가

하나의 conv filter를 여러개의 작은 filter들로 대체하면서 filter사이사이 비선형 activation 함수를 적용할 수 있다. 

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220204153838958.png" alt="image-20220204153838958" style="zoom:67%;" />

위 사진처럼 사이사이 ReLU 함수를 넣은 실험결과가 선형적으로 배치했을 때보다 더 좋은 것을 알 수 있다. 이는 비선형성의 증가로 인한 결과로 보인다. 



### 2. Efficient Grid Size Reduction

conv과 pooling을 병렬적으로 진행하고, 둘의 channel wise로 concatenate하는 새로운 방식을 제안한다. 

dxd, k channel에서 d/2xd/2, 2k channel로 가고 싶을 때는

>  방법1. 1x1 conv -> pooling

>  방법2. pooling -> 1x1 conv

이 있다. 

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220204155915806.png" alt="image-20220204155915806" style="zoom:67%;" />



하지만, 방법1은 상대적으로 더 cost가 많이 들고, 방법2는 bottleneck 문제를 일으키기 때문에 새로운 방법을 제시한다. 

> 방법3. conv + pooling 

![image-20220204155902124](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220204155902124.png)