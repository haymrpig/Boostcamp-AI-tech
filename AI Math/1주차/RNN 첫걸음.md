# 1. RNN(recurrent neural network)

보통 sequence/serial 데이터로 독립적이지 않은 경우에 대한 network이다. 

`시퀀스 데이터` : 독립된 정보들이 아닌 서로 종속적인 데이터로 event의 발생 순서가 중요하다. ex) 소리, 문자열, 주가 등

- **조건부 확률을 이용한 시퀀스 데이터 다루기**  
  ![image](https://user-images.githubusercontent.com/71866756/150506715-354a0d58-4bc7-48ee-b840-9cb02f1d2276.png)  
  여기서 수식으로는 이전 모든 정보를 이용하는 것처럼 보이지만, 실제 RNN에서는 모든 정보를 이용하진 않고, 몇 개의 과거 정보는 truncate하는 방법을 쓰기도 한다.   
  ![image](https://user-images.githubusercontent.com/71866756/150506744-aa9badd1-3947-4ba4-9710-7c72362489bd.png)  
  위 식에서 볼 수 있듯이 시퀀스 **데이터의 길이**는 **가변적**이기 때문에, 가변적인 데이터를 다룰 수 있는 모델이 필요하다. 

  - **자기회귀모델 (AR, Autoregressive Model)**  
    ![image](https://user-images.githubusercontent.com/71866756/150506798-59727686-c857-4c46-a2a3-41248a9a4ea5.png)  
    이러한 타우는 hyper parameter이기 때문에, 적절한 값의 선택이 중요하다. 

     

  - **잠재 AR 모델 (RNN)** 

    바로 이전 정보 외의 나머지 더 과거의 정보들을 하나의 잠재변수로 인코딩하는 모델이다.   
    ![image](https://user-images.githubusercontent.com/71866756/150506856-59660461-6450-4b74-ae6a-f5ac4dc11b94.png)  



- **RNN 구조**

  가장 기본적인 RNN 모형은 MLP와 유사하다.  

    
  ![image](https://user-images.githubusercontent.com/71866756/150506919-5f8aa301-f02a-454f-a7d6-d0407b0d6217.png)  
  중요한 것은 W 가중치는 t에 따라서 변하지 않는다!!

  - **BPTT (Backpropagation Through Time)**

    X1부터 Xt까지의 가중치들이 Xt부터 X1까지 차례로 업데이트가 되는 방식을 의미한다. 

  - **truncated BPTT**

    시퀀스의 길이가 길어지는 경우 BPTT를 통한 역전파 알고리즘의 계산이 불안정해지므로 길이를 끊는 것이 필요하고 이것을 truncated BPTT라고 부른다. 

    (Gradient vanishing을 해결)

**최근에는 이런 기본적인 모델을 사용하지 않고 LSTM과 GRU를 쓴다.** 
