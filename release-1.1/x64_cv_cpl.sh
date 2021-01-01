#!/bin/bash
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/ts/opencv-3.4.12/x64_out/include/opencv2:/home/ts/opencv_3.4.12/x64_out/include/opencv:/home/ts/opencv-3.4.12/x64_out/include:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ts/opencv-3.4.12/x64_out/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home/ts/opencv-3.4.12/x64_out/lib


if [ $1 = "cpl" ];then
g++ $2 -o demo -lopencv_world -std=c++11
elif [ $1 = "clr" ];then
rm -rf ./demo ./*.avi
elif [ $1 = "run" ];then
./demo $2 $3 $4 $5
fi
