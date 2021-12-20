# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake build file for C++ helloworld example.
# Assumes protobuf and gRPC have been installed using cmake.
# See cmake_externalproject/CMakeLists.txt for all-in-one cmake build
# that automatically builds all the dependencies before building helloworld.

cmake_minimum_required(VERSION 3.15)
project(LibTorchClient VERSION 0.10
        DESCRIPTION "Creates a Libtorch client that trains a ResNet18 models on CIFAR10."
        LANGUAGES CXX)

set(EXECUTABLE_NAME client)

# Choose existing gRPC or build a new one
include(cmake/common.cmake)

# Include directories
include_directories(include)
include_directories("../../src/cc/flwr/src")
include_directories("../../src/cc/flwr/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/cmake")

# Include Torchvision, which in turn includes Torch 
find_package(TorchVision REQUIRED)

# Set up Flower libraries
add_subdirectory(${CMAKE_SOURCE_DIR}/cmake)

# Targets flower client
add_executable(${EXECUTABLE_NAME}
                  "src/main.cc"
                  "src/cifar10.cc"
                  "src/torch_client.cc"
                  "src/transform.cc"
                  "include/torch_client.h"
                  "include/cifar10.h"
                  "include/transform.h")

target_link_libraries(${EXECUTABLE_NAME}
     TorchVision::TorchVision
     flwr)

# Dataset fetching
if(DOWNLOAD_DATASETS)
  set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
  file(MAKE_DIRECTORY ${DATA_DIR})

  add_custom_target(cifar10 COMMAND ${CMAKE_COMMAND}
    -D DATA_DIR=${DATA_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_cifar10.cmake)

  add_dependencies(${EXECUTABLE_NAME} cifar10)

endif()

