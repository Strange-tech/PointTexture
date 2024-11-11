# 1,设置工程名称，在Linux下可以随便设置
project( Altas )

# 2,设置 CMake 最低版本号
cmake_minimum_required( VERSION 3.5 )

# 3,设定编译参数
set(CMAKE_CXX_STANDARD    11)  # 指定 C++ 版本
set(CMAKE_BUILD_TYPE "Release")  # 调试使用Debug，可以查看中间变量；发布使用Release，运行速度快
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# 4,设定源码列表,查找指定目录(都放在./src/中)中的所有源文件,并将名称保存到 DIR_SRCS 变量中
aux_source_directory(./source/thirdparty/ THIRDPARTY)
aux_source_directory(./source/xatlas/ XATLAS)

# 5,设定头文件路径（还可以增加其他第三方库的头文件路径）
include_directories(./source/thirdparty/ ./source/xatlas/)

# 6,查找并添加OpenCV的头文件目录
# find_package(OpenCV REQUIRED)  
# message( STATUS "    version: ${OpenCV_VERSION}" )  # 我电脑上装的是opencv3.3.1
# message( STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" )
# include_directories(${OpenCV_INCLUDE_DIRS})

# 7，把源码编译成一个可执行文件，文件名为test03（可以随便取名），会保存在当前目录下
add_executable( Example ./source/examples/example.cpp ${THIRDPARTY} ${XATLAS})
# target_link_libraries( Example ${OpenCV_LIBS} )  # 可执行文件名 链接 OpenCV库
