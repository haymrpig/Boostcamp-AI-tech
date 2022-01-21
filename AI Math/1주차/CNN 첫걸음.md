# CNN

커널을 입력벡터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조

**신호**를 커널을 이용하여 국소적으로 **증폭 또는 감소**시켜 **정보를 추출/필터링**하는 것

(국소적 : 커널 사이즈만큼씩 적용이 되기 때문에)

- **convolution 연산 결과**

  입력 크기 : (H,W)

  커널 크기 : (Kh, Kw)

  출력 크기 : (Oh, Ow)

  Oh = H-Kh+1

  Ow = W-kw+1

- **convolution 수식**  
  ![image](https://user-images.githubusercontent.com/71866756/150506353-392763de-6f75-435d-b79a-7e439ec766dc.png)  

- **역전파에서의 convolution 연산**  
  ![image](https://user-images.githubusercontent.com/71866756/150506500-0c57ffd2-0e76-4dd0-a7ee-fd6d4133a21e.png)  
  역전파에서도 convolution 연산은 유지된다. 즉, 미분값에 convolution연산을 취해서 gradient를 구할 수 있는 것이다. (f는 커널, g는 input)


  EX)  
  ![image](https://user-images.githubusercontent.com/71866756/150506558-bf16b1d4-4f3c-4de0-a763-def97fd53fcb.png)  
  
