cmake_minimum_required(VERSION 3.10)
project(yolo_obb_track)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
#set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /O2") 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} /O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_C_STANDARD 11)

# Linker external library
SET(Eigen3_DIR "/usr/local")
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR "/usr/local")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/7.0.RC1/runtime/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

set(THIRDPART $ENV{THIRDPART_PATH})
if (NOT DEFINED ENV{THIRDPART_PATH})
    set(THIRDPART "/usr/local/Ascend/thirdpart")
    message(STATUS "set default THIRDPART: ${THIRDPART}")
else()
    message(STATUS "set THIRDPART: ${THIRDPART}")
endif()

include_directories(
   ${INC_PATH}/runtime/include/
   ${THIRDPART}/include/acllite/
)

link_directories(
    ${THIRDPART}/lib/
    ${LIB_PATH}
    /usr/lib
    /usr/local/lib
)

# add_subdirectory(src)
file(GLOB SRC_LIST src/*.cpp)

# add_executable(${PROJECT_NAME} main_obb.cpp ${SRC_LIST})
add_executable(${PROJECT_NAME} main_all.cpp ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ascendcl acl_dvpp stdc++ acllite dl rt)



