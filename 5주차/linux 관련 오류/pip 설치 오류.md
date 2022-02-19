# pip 설치 오류

## E : Unable to locate package python3-pip 오류 해결

```
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install python3-pip
```

- **sudo add-apt-repository universe**
  - Ubuntu, Debian은 패키지와 패키지에 대한 정보를 저장하고 있는 서버인 apt 패키지 저장소 개념을 가지고 있다. 
  - universe는 원격 저장소로 우분투에서 지원하진 않지만, 무료인 SW이다. 
  - apt 패키지 저장소 파일은 /etc/apt/sources.list에 적힌대로 관리한다. 
  - 위 명령어는 repository를 sources.list에 넣는 명령어이다. 

- **sudo apt-get update**
  - 위 명령어는 운영체제에서 사용 가능한 패키지들과 그 버전에 대한 정보를 업데이트하는 명령어이다.
  - 설치되어 있는 패키지를 최신으로 없데이트 하는 것이 아닌 설치 가능한 리스트를 업데이트한다. 

- **sudo apt-get install python3-pip**
  - 따라서 설치 가능한 패키지를 업데이트한 후 설치할 수 있다. 