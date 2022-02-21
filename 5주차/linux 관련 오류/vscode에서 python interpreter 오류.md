# vscode에서 python interpreter 오류

wsl을 사용하는 경우에도 terminal에서 code .를 실행해도 vscode는 사용환경을 windows로 인식한다. 

따라서, remote WSL로 사용환경을 Linux로 변경해주어야 한다. 

그렇지 않으면, 패키지에 대한 경로 문제로 import가 안될 수가 있다. 

- **vscode extension에서 Remote-WSL 설치**

  ![image-20220221015416615](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220221015416615.png)

- **ctrl+shift+p로 Remote-WSL검색, Remote-WSL:New WSL Window 선택**

- **ctrl+shift+p로 Python:Select Interpreter 검색, Linux에 설치한 python을 지정해준다.**

  > 만약 검색이 안될 경우, 오른쪽 하단에 뜨는 python관련 확장자 설치 클릭

