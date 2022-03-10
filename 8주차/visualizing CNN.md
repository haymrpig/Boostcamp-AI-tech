# 목차

- Visualizing CNN
- Analysis of model behaviors
- Model decision explanations

# 1. Visualizing CNN

- **Filter weight visualization**

  보통 첫번째 layer를 visualize하는데, 그 이유는 high level로 갈수록 차원이 늘어나고 알아보기 힘들기 때문이다. (사람이 해석할 수 없다!)



# 2. Analysis of model behaviors

- **High level feature에서 쓸만한 방법들**

  - **NN (Nearest neighbors)을 이용한 모델 검증**

    모델에서 마지막 fc layer를 뜯어내고, output으로 feature map을 출력하도록 한다. 

    database상의 이미지들의 feature map을 저장하고, 내가 확인하고 싶은 이미지들의 feature map과 NN 알고리즘을 통해 clustering한 그룹들을 찾아내어 확인하면 특징이 잘 뽑혔는지 알 수가 있다. 

  - **Dimensionality reduction**

    사람이 이해하기 힘든 고차원 feature들을 저차원으로 바꾼다!

    (t-SNE 알고리즘 사용하여 분포를 확인할 수 있다. )

- **Mid, High level에서 쓸만한 방법들**

  - **Layer Activation 분석** 

    hidden layer의 특정 channel을 threshold를 주어 나타내면 해당 unit이 어디를 attention하고 있는지 알 수 있다. 

  - **Maximally activating patches**

    hidden layer의 특정 channel에서 가장 높은 값을 가지는 픽셀 근방의 patch를 뜯어와 확인하는 것!

- **Output을 이용할 때 쓸만한 방법들**

  - **class visualization (Gradient ascent)**

    Step1. (black or random initial)된 이미지를 model에 넣는다.

    Step2. 확인하고픈 class의 값이 최대가 되도록, backpropagation을 통해서 입력이미지를 update한다. (loss를 최소화하는 게 아닌, output을 최대화하는 것을 목적으로 함)

    Step3. update된 이미지를 다시 model에 넣는다. 

    Step4. 위 과정 반복

    -> 초기값 설정에 따라 나오는 결과가 다를 수 있다. 



# 3. Model decision explanation

모델이 특정 input에 대해 어떤 시야를 가지고 있는지 확인 가능

- **Saliency test**

  - **Occlusion map**

    패치를 임의의 위치에 붙이고, 원래 이미지가 출력됐어야할 class의 확률 변화를 살펴본다. 

    모든 위치에 대해서 패치를 붙이고, 각각의 확률을 계산하여 어느 부분이 민감한지 아닌지 확인해 볼 수 있다. 

  - **via Backpropagation**

    Step1. 확인하고 싶은 입력 영상을 넣어준다.

    Step2. class score에 따른 gradient를 back propagate한다. 

    Step3. 절대값을 취한 gradient를 이미지화 한다. 

    > 절대값을 취하는 이유는
    >
    > 입력에서 많이 바뀌어야 하는 부분의 gradient가 크기 때문이다.(부호 상관없이 절대적인 크기만 고려)

  - **ReLU 이용**

    - **Saliency Map using normal ReLU**

      forward pass에서 ReLU의 특성상 음수인 부분은 0으로 masking이 된다. 

      backward pass에서는 이 0인 부분을 기억하고, gradient에서 0인 부분의 값을 0으로 만들어준다. 

    - **Deconvolution**

      backward pass에서 gradient가 음수인 부분을 0으로 masking한다. 

      (즉, backward에서 ReLU를 적용하는 것)

    - **Guided backpropagation**

      위 두 gradient를 and 연산을 취한것

      (수학적인 의미를 해석하기보단 직관적인 유추를 하는 게 맞다. 수학적으로는 좀 이해가 되지 않는 방식이다.)

  - **CAM**

    기존의 구조를 조금 수정해야 한다. 

    Global average pooling을 추가하고 마지막 layer는 fc가 하나여야 한다.

    즉, 마지막 score가 나오기 위해서 마지막 GAP을 거친 feature들에 weight가 곱해진다. 

    ![image](https://user-images.githubusercontent.com/71866756/157659593-5f626e56-f00e-476f-9336-6f14a7cc4275.png)

    위 수식과 같이 sum_(x,y)가 GAP를 의미하므로, GAP를 거쳐 공간 정보를 잃기 전의 값들에 weight를 곱한 것이 heatmap 형태로 표현하면 CAM이 된다. 

    CAM 결과를 보면 localization도 되는 것을 확인할 수 있다.

    (weakly supervised learning)

    - 단점

      구조가 바뀌는 모델의 경우 성능이 떨어질 수 있다. 

  - **Grad-CAM**

    Guided backpropagation + Grad-CAM (Guided-Grad-CAM)을 쓸 경우 좋은 결과를 얻을 수 있다. 

    > Guided backpropagation 같은 경우 해당 클래스를 정확하게 집어내지 못한다. 
    >
    > Grad-CAM의 경우 경계가 뚜렷하지 않고, 흐리멍텅하다. 
    >
    > 이 둘을 합치면 경계가 뚜렷하며, 해당 클래스를 정확하게 집어내는 결과를  볼 수 있다. 

    
