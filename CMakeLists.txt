cmake_minimum_required(VERSION 3.26)
project(pico_dt)

set(CMAKE_CXX_STANDARD 17)

add_executable(pico_dt_test src/main.cpp
        src/DecisionTreeNode.cpp
        src/DecisionTreeNode.h)

add_library(pico_dt INTERFACE
        src/DecisionTreeNode.cpp)
