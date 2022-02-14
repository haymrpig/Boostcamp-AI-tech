# 목차

- **Linux**
- **Shell Command**



# 1. Linux

#### 1-1. linux를 왜 사용하는가?

서버 관련 작업을 할 때 linux를 많이 사용한다. 

그 이유는 

- Mac과 Windows같은 경우는 유료이지만, **linux는 무료**이기 때문이다. 

- 오픈소스이기 때문에 **확장성**이 좋다.
- 유닉스의 특징으로 **안정성과 신뢰성**이 좋다
- **쉘 커맨드**와 **쉘 스크립트**를 사용할 수 있다. 



#### 1-2. 대표적인 linux 배포판

- **Debian** : 온라인 커뮤니티에서 제작해 배포
- **Ubuntu** : 영국 캐노니컬 회사에서 만듬, 초보자가 이용하기 쉬움
- **Redhat** : 레드햇이라는 회사에서 배포
- **CentOS** : 레드햇 공개 버전에서 브랜드와 로고 제거후 배포한 버전



#### 1-3. Linux 사용 방법?

- **Virtual box, Docker**

- **Window의 경우 WSL**

- **Notebook에서 터미널 실행**

- **Cloud에서 띄우는 인스턴스에서 연습**

  : 일단은 여기서 하자!



# 2. Shell Command

#### 2-1. 쉘의 종류

`쉘`은 사용자가 문자를 입력해 컴퓨터에 명령할 수 있는 프로그램이다. 

`터미널/콘솔` : 쉘을 실행하기 위해 문자를 입력 받아 컴퓨터에 전달 + 프로그램의 출력을 화면에 작성

- **sh** : 최초의 쉘
- **bash** : linux 표준 쉘
- **zsh** : Mac 카탈리나 OS 기본 쉘



#### 2-2. 쉘은 언제 사용할까?

- **서버에서 접속**해서 사용하는 경우
- **crontab** 등 **linux의 내장 기능**을 활용하는 경우
- **데이터 전처리**
- **Docker**를 사용하는 경우
- **수백대의 서버**를 관리할 경우
- jupyter notebook에서 cell앞에 !를 붙이면 쉘 커맨드가 사용됨
- 터미널에서 **python3, jupyter notebook도 쉘 커맨드**
- **test code 실행**
- **배포 파이프라인 실행** (Github Action 등에서 실행) 



#### 2-3. 기본 명령어

- **man** : 쉘 커맨드의 매뉴얼 문서를 보고 싶은 경우

  ex) man python (종료는 esc + :q)

- **mkdir** : 폴더 생성하기

- **ls** : 현재 폴더, 파일 확인

  - ls -a : 전체 파일 출력

  - ls -l : 퍼미션, 소유자, 만든 날짜 등 세부 내용까지 출력

  - ls -h : 용량을 사람이 읽기 쉽도록 GB, MB 등 표현

  ex) ls -al, ls -lh, ls, ls ~

- **pwd** : 현재 폴더 경로를 절대 경로로 출력

- **cd** : 폴더 변경하기, 이동하기

- **ehco** : python의 print문

  - 쉘 커맨드 입력시 쉘 커맨드 결과 출력

  ex) echo 'pwd' (이거 따옴표가 아니라 backtik임)

- [**vi**](####vi-추가-설명) : vim 편집기

- **bash** : bash로 쉘 스크립트 실행

- **sudo** : 관리자 권한으로 실행하고 싶은 경우

- **cp** : 파일 또는 폴더 복사하기

  - -r : 디렉토리 안에 파일 있으면 재귀적으로 모두 복사

  - -f : 복사할 때 강제로 실행

- **mv** : 파일, 폴더 이동 (이름 변경)

- **cat** : 특정 파일 내용 출력

  ex) cat test1.sh test2.sh : 파일 두개 합쳐서 출력

  cat test1.sh test2.sh >> new.sh : 파일 두개 합쳐서 하나의 파일을 생성

- **history** : 최근 입력한 쉘 커맨드 출력

  history로 확인한 커맨드의 번호 앞에 !를 붙여서 명령어를 작성하면 그 커맨드가 실행됨

  ex) history에 31이 cat test1.sh였다면, command line에 !31을 하면 똑같이 실행됨 

- **find** : 파일 및 디렉토리 검색

  ex) find . -name "File" : 현재 폴더에서 File이란 이름을 가지는 파일/디렉토리 검색

- **export** : 환경변수 설정

  - 매번 쉘을 실행할 때마다 환경변수를 저장하고 싶으면 .bashrc, .zshrc에 저장하면 된다. 
  - source ~/.bashrc 하면 즉시 적용가능

  ex) export water="물", echo $water하면 "물"이 나온다.

- **alias** : 별칭 지정

  ex) alias ll2='ls -l'

- **head/tail** : 파일 앞/뒤 n행 출력

  ex) head -n 3 test.sh : 앞 3줄 출력

- **sort** : 행단위 정렬

  - -r : 내림차순 정렬

  - -n : Numeric sort

  ex) cat fruits.txt | sort -r : fruits.txt의 각 행을 sort해서 보여줌

- **uniq** : 중복된 행이 연속으로 있는 경우 중복 제거

  - -c : 중복 행의 개수 출력

  ex) cat fruits.txt | sort | uniq | wc -l : wc는 word count, 라인 수를 세줌

- **grep** : 파일에 주어진 패턴 목록과 매칭되는 라인 검색

  - -i : 대소문자 구분 없이 찾기
  - -w : 정확히 그 단어만 찾기
  - -v : 특정 패턴 제외한 결과 출력
  - -E : 정규 표현식 사용
    - ^단어 : 단어로 시작하는 것 찾기
    - 단어$ : 단어로 끝나는 것 찾기
    - . : 하나의 문자 매칭

- **cut** : 파일에서 특정 필드 추출

  - -f : 잘라낼 필드 지정
  - -d : 필드를 구분하는 구분자. Default는 \t

  ex) cat file | cut -d : -f 1,7 : \t으로 구분되는 값 중, 1, 7번째 값을 가져온다. 

- **>, >>** : Redirection이라고 부르며, 프로그램의 출력 (stdout)을 다른 파일이나 스트림으로 전달한다

  - '>' : 덮어쓰기 (Overwrite) 파일이 없으면 생성하고 저장
  - '>>' : 맨 아래에 추가하기

  ex) echo "hi" > test.sh

  echo "hi" >> test.sh

- **|** : Pipe라고 부르면 프로그램의 출력을 다른 프로그램의 입력으로 사용하고 싶은 경우이다. 

  ex) ls | grep "vi" : ls의 결과를 grep 명령어의 입력으로 보내어 "vi"를 찾는다. 

  ls | grep "vi" > output.txt : 결과를 output.txt에 저장

- **ps** : 현재 실행되고 있는 프로세스 출력하기

  - -e : 모든 프로세스
  - -f : full format으로 자세히 보여줌

- **curl** : command line 기반의 data transfer 커맨드, request 테스트 할 수 있다. 웹 서버를 작성한 후 요청이 제대로 실행되는지 확인할 수 있음

  ex) curl -X localhost:5000/ {data}

  - httpie 등도 있음

- **df** : 현재 사용 중인 디스크 용량 확인

  - -h : 사람이 읽기 쉬운 형태로 출력

- **scp** : SSH를 이용해 네트워크로 연결된 호스트 간 파일을 주고 받는 명령어

  - -r : 재귀적으로 복사
  - -P : ssh 포트 지정
  - -i : SSH 설정을 활용해 실행

  ex) scp local_path user@ip:remote_directory

  ex) scp user@ip:remote_directory user2@ip2:target_remote directory

- **nohup** : 터미널 종료 후에도 계속 작업이 유지하도록 실행 (백그라운드 실행)

  nohup의 permission은 755여야 한다. 

  ex) nohup python3 app.py &

  - 종료 : ps ef | grep app.py한 후, pid(process id)를 찾은 후 kill -9 pid로 프로세스 kill
  - screen이란 도구도 있다. 

- **chmod** : 파일의 권한을 변경

  - r : Read, 4
  - w : write, 2
  - x : execute, 1
  - '-' : denied

  위에 각각이 적힌 값으로 권한을 변경 -> 7은 rwx모두 가능, 5는 rx만 가능

  ex) chmod 755 test.txt : 총 세자리로 각각의 자리는 대상을 의미한다. 



#### 2-4. 표준 스트림 (stream)

Unix에서 동작하는 프로그램은 커맨드 실행시 3개의 stream을 생성한다. 

- **stdin** : 0으로 표현하며, 입력을 의미 (비밀번호, 커맨드 등)
- **stdout** : 1로 표현, 출력 값 (터미널에 나오는 값)
- **stderr** : 2로 표현, 디버깅 정보나 에러 출력



#### 2-5. 쉘 스크립트

`쉘 스크립트`는 쉘 커맨드를 모아놓은 것이다.

EX)

- #!/bin/bash : Shebang으로 이 스크립트를 Bash 쉘로 해석하겠다는 의미이다. 
- $(date+%s) : date를 %s(unix timestamp)로 변형한다는 의미이다. 
- START=$(date+%s) : 변수 저장

 [Ref]

 [예시1](https://github.com/zzsza/shell-scripts)

[예시2](https://github.com/denysdovhan/bash-handbook)

[예시3](https://github.com/epety/100-shell-script-examples)

#### vi 추가 설명

- **Command Mode**

  | 명령어 | 기능                                                  |
  | ------ | ----------------------------------------------------- |
  | dd     | 현재 위치한 줄 삭제                                   |
  | i      | insert모드로 변경                                     |
  | x      | 커서가 위치한 곳의 글자 1개 삭제 (5x : 문자 5개 삭제) |
  | yy     | 현재 줄을 복사                                        |
  | p      | 현재 커서가 있는 줄 바로 아래에 붙여넣기              |
  | k      | 커서 위로                                             |
  | j      | 커서 아래로                                           |
  | l      | 커서 오른쪽으로                                       |
  | h      | 커서 왼쪽으로                                         |

- **Insert Mode**

  파일 수정할 수 있는 모드

- **Last Line Mode**

  ESC + : 누르면 나오는 모드

  | 명령어 | 기능                                          |
  | ------ | --------------------------------------------- |
  | w      | 현재 파일명으로 저장                          |
  | q      | vi 종료 (q!는 강제종료)                       |
  | wq     | 저장 후 종료 (wq!는 강제저장 후 종료)         |
  | /문자  | 문자 탐색 (탐색 후 n을 누르면 계속 탐색 실행) |
  | set nu | vi 라인 번호 출력                             |

  



# 명령어 연습문제

1. test.txt 파일에 "Hi!!!!"을 입력해주세요
   - echo "Hi!!!!" > test.txt
2. test.txt 파일 맨 아래에 "kkkk"를 입력해주세요
   - echo "kkkk" >> test.txt
3. test.txt의 라인 수를 구해주세요
   - cat test.txt | wc -l
4. 카카오톡 그룹 채팅방에서 옵션 -대화 내보내기로 csv로 저장 후, 쉘 커맨드 1줄로 카카오톡 대화방에서 2021년에 제일 메세지를 많이 보낸 TOP 3명 추출하기