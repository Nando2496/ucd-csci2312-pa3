cmake_minimum_required(VERSION 3.3)
project(ucd-csci2312-pa3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Point.cpp Point.h Cluster.cpp Cluster.h
ErrorContext.cpp ErrorContext.h ClusteringTests.cpp ClusteringTests.h KMeans.cpp KMeans.h Exceptions.h Exceptions.cpp Cluster.cpp Point.cpp KMeans.cpp Exceptions.cpp)
add_executable(ucd-csci2312-pa3 ${SOURCE_FILES})
