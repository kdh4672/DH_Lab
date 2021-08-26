# 도커 연습 (Day 3)

- Container 상에서 git 명령어를 이용하여 repository를 git clone 하기
```
docker run --name repo alpine/git clone https://github.com/docker/getting-started.git
```
- repo에 있는 /git/getting-started 폴더를 현재 위치로 옮기기
```
docker cp repo:/git/getting-started/ .
```

- docker 이미지 현재 위치로 불러오기
```
docker build -t docker101tutorial .
```

- docker container 실행시키기. (local machine과 별개로 작동함)
```
docker run -d -p 80:80 --name docker-tutorial docker101tutorial
```

- docker image를 docker hub에 올려서, 다른 machine들에서도 언제든지 다운받아 사용가능하게 하기
```
docker tag docker101tutorial kdh4672/docker101tutorial
docker push kdh4672/docker101tutorial
```

- docker hub에 올려진 image를 pull 하기
```
docker pull kdh4672/docker101tutorial
```

# Day 4

- 참고할 자료 : [유튜브-생활코딩](https://www.youtube.com/watch?v=Ps8HDIAyPD0)

- docker 개념 그림
![docker_개념](https://media.oss.navercorp.com/user/26454/files/a95d5e80-0658-11ec-98fc-38bf39e6caaa)

- docker hub에 있는 'httpd' 라는 이미지 pull해서 run하기까지

```
docker pull httpd
```
![docker_pull_httpd](https://media.oss.navercorp.com/user/26454/files/9f882b00-0659-11ec-8413-84ebefcb04fa)

```
docker run --name ws2 httpd
# docker run httpd 할경우 container name이 random으로 생성됨
```
#### gui 결과
![docker_run_ws2_gui](https://media.oss.navercorp.com/user/26454/files/6c926700-065a-11ec-9fd9-1c8edcb8a027)
#### 터미널 결과
```
docker ps
```
![terminal_ps](https://media.oss.navercorp.com/user/26454/files/107c1280-065b-11ec-804b-7f21196c8baa)
```
docker ps -a
```
(stop된 container들도 보여줌)
![docker_ps_a](https://media.oss.navercorp.com/user/26454/files/3b666680-065b-11ec-9d53-ea959760ffd4)

- docker stop, start, log

```
docker stop ws2
```

```
docker start ws2
```

```
docker log ws2
```
```
docker log -f ws2
```
(연속해서 log 보기)

- docker container 삭제

```
docker rm ws2
```
(ws2가 실행중이라면 docker stop 한다음 rm 해야함)

- docker image 삭제
```
docker rmi httpd
```

- docker port forwarding

```
docker run -p 80:80 httpd
```
(host port: 80 . container port: 80)
![prot_80_80](https://media.oss.navercorp.com/user/26454/files/fb54b300-065d-11ec-94e8-4426c51c6d68)
```
docker run -p 8000:80 httpd
```
(host port: 8000 . container port: 80)
![8000_80](https://media.oss.navercorp.com/user/26454/files/217a5300-065e-11ec-9276-5b295b5fc54c)

- 실제로 실행시켜보기
```
docker run --name ws3 -p 8000:80 httpd
```
![실행](https://media.oss.navercorp.com/user/26454/files/2c81b300-065f-11ec-93c9-60b62ab9eeca)


- container 속에 들어가서 코드 수정 (ex : index.html)

```
docker exec [OPTIONS] CONTAINER COMMAND [ARGUMENT]
```

```
docker exec ws3 ls
```

#### ws3 대상으로 command 계속하기
```
docker exec -it ws3 /bin/bash
or
docker exec -it ws3 /bin/sh
```
나가기
```
exit
```

#### 실제로 수정하기

```
docker exec -it ws3 /bin/bash
cd ./htdocs
apt update
apt install nano
nano index.html
It works --> Hello, Docker
```
![Hello,Docker](https://media.oss.navercorp.com/user/26454/files/9484c900-0660-11ec-8f68-4cdc3100ab2d)

- Host와 Container를 연결하여 source code를 container 밖에서 수정하고 저장하기 (컨테이너에도 동시에 저장됨)
![docker_attatch](https://media.oss.navercorp.com/user/26454/files/fba27d80-0660-11ec-93b7-9b0d2112beb1)
```
docker run -p 8888:80 -v ~/htdocs:/usr/local/apache2/htdocs/ httpd
```
![hello_never_index](https://media.oss.navercorp.com/user/26454/files/b41cf100-0662-11ec-8b3d-aa83fbf7d48b)
![hello_naver_page](https://media.oss.navercorp.com/user/26454/files/c008b300-0662-11ec-94ba-a799a53fa05a)


