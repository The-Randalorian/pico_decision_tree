cmake_minimum_required(VERSION 3.26)
project(pico_dt)

set(CMAKE_CXX_STANDARD 17)

add_executable(pico_dt_test src/main.cpp
        src/DecisionTreeNode.cpp
        src/DecisionTreeNode.h)

add_library(pico_dt INTERFACE)
target_sources(pico_dt INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/src/DecisionTreeNode.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/DecisionTreeNode.cpp
)
target_include_directories(pico_dt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
