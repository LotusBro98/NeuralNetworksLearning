cmake_minimum_required(VERSION 3.9)
project(01)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} .)

set(CMAKE_CXX_STANDARD 11)

add_executable(01
        sources/Layer.cpp sources/Layer.h sources/Dataset.cpp sources/Dataset.h sources/Network.cpp sources/Network.h sources/main.cpp)

target_link_libraries( 01 ${OpenCV_LIBS} )
