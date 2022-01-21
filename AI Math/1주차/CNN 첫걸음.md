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
  $$
  [f*g](x)=\int_{R^d}f(z)g(x+z)dz=\int_{R^d}f(x+z)g(z)dz=[g*f](x)\\
  이산의\; 경우는\; \int를\; \sum으로 \;바꾸면\; 된다.
  $$

- **역전파에서의 convolution 연산**
  $$
  \begin{aligned}\frac{\partial}{\partial x}[f*g](x)&=\frac{\partial}{\partial x}\int_{R^d}f(z)g(x+z)dz\\
  &=\int_{R^d}f(z)\frac{\partial g}{\partial x}(x+z)dz\\
  &=[f*g'](x)
  \end{aligned}
  $$
  역전파에서도 convolution 연산은 유지된다. 즉, 미분값에 convolution연산을 취해서 gradient를 구할 수 있는 것이다. (f는 커널, g는 input)

  EX)
  $$
  \frac {\partial L}{\partial w_i}=\sum_j\delta_jx_{i+j-1}
  $$