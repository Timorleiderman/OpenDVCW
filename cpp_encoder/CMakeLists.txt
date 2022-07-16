cmake_minimum_required(VERSION 3.0.0)
project(tave VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")

find_package( OpenCV REQUIRED )

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVSWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_library(AVSWSCALE_LIBRARY swscale)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
find_library(AVDEVICE_LIBRARY avdevice)

add_executable(tave main.cpp)
target_include_directories(tave PRIVATE ${OpenCV_INCLUDE_DIRS} ${AVCODEC_INCLUDE_DIR} ${AVSWSCALE_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
target_link_libraries(tave PRIVATE ${OpenCV_LIBS} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVSWSCALE_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
