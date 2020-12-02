- build.sh 로 빌드하려면 ...
1) source build.sh
   -->build 폴더 생성 및 자동으로 build 폴더로 이동
2) cmake --build .
   -->build 안에 있는 CMakeFiles 대로 설치

끝

만약 이 중에 오류가 발생한다면, 
If one of the dependencies can't be found by CMake, use the "-DCMAKE_PREFIX_PATH=" option to tell CMake where to find them. For example, if CMake can't find GLEW :
```
cmake -DCMAKE_PREFIX_PATH="/path/to/glew/glew-2.1.0/build/" ..
```

path
