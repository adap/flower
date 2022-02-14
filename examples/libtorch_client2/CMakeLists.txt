cmake_minimum_required(VERSION 3.16)
project(LIBTORCHCLIENT VERSION 0.10
  DESCRIPTION "Creates a LibTorch client that trains a ResNet18 model on CIFAR10."
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(ABSL_PROPAGATE_CXX_STD ON)

######################
### Add dependencies
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)


######################
### Download gRPC 

include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.43.2
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()


######################
### FLWR_GRPC_PROTO

get_filename_component(FLWR_PROTO "../../src/proto/flwr/proto/transport.proto" ABSOLUTE)
get_filename_component(FLWR_PROTO_PATH "${FLWR_PROTO}" PATH)

set(FLWR_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.cc")
set(FLWR_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.h")
set(FLWR_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.cc")
set(FLAR_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.h")

# External building command to generate gRPC source files.
add_custom_command(
  OUTPUT "${FLWR_PROTO_SRCS}" "${FLWR_PROTO_HDRS}" "${FLWR_GRPC_SRCS}" "${FLWR_GRPC_HDRS}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
	-I "${FLWR_PROTO_PATH}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${FLWR_PROTO}"
  DEPENDS "${FLWR_PROTO}"
)

add_library(flwr_grpc_proto 
  ${FLWR_GRPC_SRCS}
  ${FLWR_GRPC_HDRS}
  ${FLWR_PROTO_SRCS}
  ${FLWR_PROTO_HDRS}
)

target_include_directories(flwr_grpc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(flwr_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)

######################
### FLWR_LIB

file(GLOB FLWR_SRCS "../../src/cc/flwr/src/*.cc")
set(FLWR_INCLUDE_DIR "../../src/cc/flwr/include")

add_library(flwr ${FLWR_SRCS})

target_include_directories(flwr PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${FLWR_INCLUDE_DIR}
)

target_link_libraries(flwr
  flwr_grpc_proto
)

######################
### FLWR_CLIENT

file(GLOB FLWR_CLIENT_SRCS src/*.cc)
set(EXECUTABLE_NAME flwr_client)
add_executable(${EXECUTABLE_NAME} ${FLWR_CLIENT_SRCS})

target_include_directories(${EXECUTABLE_NAME} PUBLIC
  include
  ${FLWR_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${EXECUTABLE_NAME}
  TorchVision::TorchVision
  flwr
)

######################
### Helper functions 

function(check_files BASE_DIR FILES_TO_CHECK FILE_MD5S MISSING_FILES)
    foreach(FILE_TO_CHECK ${${FILES_TO_CHECK}})
        if(EXISTS "${BASE_DIR}/${FILE_TO_CHECK}")
            list(FIND ${FILES_TO_CHECK} ${FILE_TO_CHECK} MD5_IDX)
            list(GET ${FILE_MD5S} ${MD5_IDX} EXPECTED_FILE_MD5)
            
            file(MD5 "${BASE_DIR}/${FILE_TO_CHECK}" ACTUAL_FILE_MD5)
            
            if(NOT (EXPECTED_FILE_MD5 STREQUAL ACTUAL_FILE_MD5))
                list(APPEND RESULT_FILES ${FILE_TO_CHECK})
            endif()
        else()
            list(APPEND RESULT_FILES ${FILE_TO_CHECK})
        endif()
    endforeach()
    
    set(${MISSING_FILES} ${RESULT_FILES} PARENT_SCOPE)
endfunction()

function(fetch_cifar10 DATA_DIR)
    set(CIFAR_DIR "${DATA_DIR}/cifar10")
    set(CIFAR_URL "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
    set(CIFAR_EXTRACTED_FILES
        "data_batch_1.bin" "data_batch_2.bin" "data_batch_3.bin"
        "data_batch_4.bin" "data_batch_5.bin" "test_batch.bin")

    set(CIFAR_EXTRACTED_FILE_MD5s
        "5dd7e06a14cb22eb9f671a540d1b7c25" "5ea93a67294ea407fff1d09f752e9692"
        "942cd6a4bcdd0dd3c604fbe906cb4421" "ae636b3ba5c66a11e91e8cb52e771fcb"
        "53f37980c15c3d472c316c40844f3f0d" "803d5f7f4d78ea53de84dbe85f74fb6d")

    check_files(${CIFAR_DIR} CIFAR_EXTRACTED_FILES CIFAR_EXTRACTED_FILE_MD5s
                MISSING_FILES)

    if(MISSING_FILES)
        message(STATUS "Fetching CIFAR10 dataset...")
        set(CIFAR10_ARCHIVE_NAME "cifar-10-binary.tar.gz")

        file(REMOVE_RECURSE ${CIFAR_DIR})

        file(
            DOWNLOAD ${CIFAR_URL} "${DATA_DIR}/${CIFAR10_ARCHIVE_NAME}"
            SHOW_PROGRESS
            EXPECTED_MD5 "c32a1d4ab5d03f1284b67883e8d87530")

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar x
                    "${DATA_DIR}/${CIFAR10_ARCHIVE_NAME}"
            WORKING_DIRECTORY ${DATA_DIR})

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E rename
                    "${DATA_DIR}/cifar-10-batches-bin" ${CIFAR_DIR})

        file(REMOVE "${DATA_DIR}/${CIFAR10_ARCHIVE_NAME}")
        message(STATUS "Fetching CIFAR10 dataset - done")
    endif()
endfunction()

if(DOWNLOAD_DATASETS)
  set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
  file(MAKE_DIRECTORY ${DATA_DIR})
  fetch_cifar10(${DATA_DIR})
endif()

