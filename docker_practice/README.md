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
![docker_개념](https://user-images.githubusercontent.com/54311546/130893793-fba64d63-9c13-4d16-8946-fb85480a8b6b.png)

- docker hub에 있는 'httpd' 라는 이미지 pull해서 run하기까지

```
docker pull httpd
```
![docker_pull_httpd](https://user-images.githubusercontent.com/54311546/130893804-78c1730f-6b22-48a0-88a1-0daf1e3828c4.png)


```
docker run --name ws2 httpd
# docker run httpd 할경우 container name이 random으로 생성됨
```
#### gui 결과
![docker_ws2_terminal](https://user-images.githubusercontent.com/54311546/130893808-a216fff7-4417-4ef3-80ba-dd2d809212f9.png)

#### 터미널 결과
```
docker ps
```
![terminal_ps](https://user-images.githubusercontent.com/54311546/130893815-cc050cf5-e315-46b1-9826-f64e1f3e336f.png)

```
docker ps -a
```
(stop된 container들도 보여줌)
![docker_ps_a](https://user-images.githubusercontent.com/54311546/130893803-c8bcdcb1-910f-4f75-ac73-6e47c534b75b.png)


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
![prot_80_80](https://user-images.githubusercontent.com/54311546/130893814-f6d56cae-0dff-4aeb-adfe-17533229fbd4.png)
```
docker run -p 8000:80 httpd
```
(host port: 8000 . container port: 80)
![8000_80](https://user-images.githubusercontent.com/54311546/130893786-2ebe93c6-a91f-480f-a1f6-fa4d2a2a385b.png)

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
![Hello,Docker](https://user-images.githubusercontent.com/54311546/130893812-cccf1524-0b9e-4260-a938-ce3dba6ed0ad.png)

- Host와 Container를 연결하여 source code를 container 밖에서 수정하고 저장하기 (컨테이너에도 동시에 저장됨)
![docker_attatch](https://user-images.githubusercontent.com/54311546/130893795-60244692-bf44-4169-b515-324f6b19e56b.png)
```
docker run -p 8888:80 -v ~/htdocs:/usr/local/apache2/htdocs/ httpd
```
![hello_never_index](https://user-images.githubusercontent.com/54311546/130893811-f2fcb204-75db-4d85-98d5-3aefaa3ba691.png)
![hello_naver_page](https://user-images.githubusercontent.com/54311546/130893810-0391d4b8-e036-4d97-83cb-152523978622.png)


