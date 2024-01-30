cmake_minimum_required(VERSION 3.26)
project(pico_dt)

set(CMAKE_CXX_STANDARD 17)

add_executable(pico_dt src/main.cpp
        src/DecisionTreeNode.cpp
        src/DecisionTreeNode.h)
