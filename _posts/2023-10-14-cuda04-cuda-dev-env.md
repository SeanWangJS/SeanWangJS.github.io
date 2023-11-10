---
title: CUDA 编程入门（4）：搭建 CUDA 开发环境
tags: CUDA 并行计算
---

## 项目结构

上一篇我们写了一个简单的小程序，使用一条命令就可以编译通过，但实际的项目结构要复杂得多，因此在本篇文章中，我们先介绍如何使用 VSCode 搭建一个 CMake 工程来进行 CUDA 开发，并集成 googletest 框架来做测试。

首先，我们的项目结构如下

```
|.vscode/
|--c_cpp_properties.json
|include/
|--vector_add.h
|src/
|--vector_add_kernel.cu
|test/
|--CMakeLists.txt
|--test.cpp
|CMakeLists.txt
```

这里的 `c_cpp_properties.json` 主要是用来指引 VSCode，使其能够识别在源码中 include 的头文件，即下面的 `includePath` 配置的路径

```json
{
    "configurations": [
        {
            "name": "Win32",
            "intelliSenseMode": "windows-msvc-x64",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "includePath": [
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include",
                "${workspaceRoot}/include",
                "C:/ProgramData/googletest/include"
            ]
        }
    ],
    "version": 4
}
```

## 编写 CMakeLists.txt

主文件夹下的 CMakeLists.txt 内容如下

```CMake
cmake_minimum_required(VERSION 3.10)

project(vector_add LANGUAGES CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)

file(GLOB_RECURSE CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(test)
enable_testing()
```

这里我们使用了 `file(GLOB_RECURSE CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/src/*.cpp)` 来自动扫描 src 目录下的所有 .cu 和 .cpp 文件，然后将其作为源文件添加到工程中。

通过 `add_library`，我们将 src 目录下的源文件编译成一个静态库，然后通过 `target_include_directories` 将 include 目录添加到工程中，这样 src 目录下的源文件就可以使用 include 目录下的头文件了。

最后，我们通过 `add_subdirectory` 来添加 test 目录，这样 test 目录下的 CMakeLists.txt 就会被执行。

test 目录下的 CMakeLists.txt 内容如下

```CMake
cmake_minimum_required(VERSION 3.10)

project(test LANGUAGES CXX)

find_package(GTest REQUIRED)

add_executable(test_vector_add test.cpp)

target_include_directories(test_vector_add PUBLIC ${GTEST_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
target_link_libraries(test_vector_add ${GTEST_LIBRARIES} vector_add)
```

这里我们通过 `find_package(GTest REQUIRED)` 来查找 googletest 框架，使用 `add_executable` 将 test.cpp 添加为可执行文件，然后通过 `target_include_directories` 将 googletest 的头文件和本项目的头文件添加到工程中，通过 `target_link_libraries` 将 googletest 的库文件链接到 test_vector_add 可执行文件中。

## 编译过程

以上就是整个项目的 CMake 配置，其他代码这里就不赘述了，完整的项目可以在 [GitLab](https://gitlab.com/cuda_exercise/vector-add) 找到。下面我们给出 CMake 的编译过程

```shell
mkdir build
cd build
cmake .. -DGTEST_LIBRARY=~/googletest/lib/gtest.lib -DGTEST_INCLUDE_DIR=~/googletest/include -DGTEST_MAIN_LIBRARY=~/googletest/lib/gtest_main.lib
cmake --build .
```

由于 googletest 属于外部框架，所以我们需要给 CMake 传入 googletest 的路径来确保 CMake 能够找到 googletest。

最后提一下在 Windows 平台使用 googletest 可能会遇到的问题，在默认情况下，VS 动态的链接 C 运行库，而 googletest 是静态链接的，因此会出现类似以下的链接错误

```
...error LNK2038: mismatch detected for ‘RuntimeLibrary’: value ‘MTd_StaticDebug’ doesn't match value ‘MDd_DynamicDebug’ in...
```

为此，官方的解决方案是在编译 googletest 框架时，在 CMake 命令中添加参数 `-Dgtest_force_shared_crt=ON`[1]。

## 参考

[1] 详细内容见 [googletest README](https://github.com/google/googletest/blob/main/googletest/README.md#visual-studio-dynamic-vs-static-runtimes)