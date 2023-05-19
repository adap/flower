cmake_minimum_required(VERSION 3.16)
project(flwr_lib VERSION 0.10
  DESCRIPTION "Generate Flower library"
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(ABSL_PROPAGATE_CXX_STD ON)

######################
### Download gRPC 

include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.43.2
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext
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

get_filename_component(FLWR_PROTO "../../proto/flwr/proto/transport.proto" ABSOLUTE)
get_filename_component(FLWR_PROTO_PATH "${FLWR_PROTO}" PATH)

set(FLWR_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.cc")
set(FLWR_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.h")
set(FLWR_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.cc")
set(FLWR_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.h")

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

set(FLWR_GRPC_LIB "flwr_grpc_proto")
add_library(flwr_grpc_proto SHARED
  ${FLWR_GRPC_SRCS}
  ${FLWR_GRPC_HDRS}
  ${FLWR_PROTO_SRCS}
  ${FLWR_PROTO_HDRS}
)

target_include_directories(flwr_grpc_proto PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(flwr_grpc_proto
  $<BUILD_INTERFACE:${_REFLECTION}>
  $<BUILD_INTERFACE:${_GRPC_GRPCPP}>
  $<BUILD_INTERFACE:${_PROTOBUF_LIBPROTOBUF}>
)

install(FILES ${FLWR_GRPC_HDRS} ${FLWR_PROTO_HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(TARGETS ${FLWR_GRPC_LIB}  DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT ${FLWR_GRPC_LIB} )
install(EXPORT ${FLWR_GRPC_LIB}  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${FLWR_GRPC_LIB})


######################
### FLWR_LIB

file(GLOB FLWR_SRCS "src/*.cc")
#aux_source_directory(src/ FLWR_SRCS)

file(GLOB HEADERS "include/*.h")
set(FLWR_INCLUDE_DIR "include")

add_library(flwr_lib SHARED ${FLWR_SRCS})

target_include_directories(flwr_lib PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${FLWR_INCLUDE_DIR}
)

target_link_libraries(flwr_lib PRIVATE
  flwr_grpc_proto
  #$<BUILD_INTERFACE:flwr_grpc_proto>
)

#intall
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${CMAKE_PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}  DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT ${PROJECT_NAME} )
install(EXPORT ${PROJECT_NAME}  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flwrLib)

#generate flwrLibConfig.cmake
configure_file(./lib_config/flwrConfig.cmake.in ${PROJECT_BINARY_DIR}/flwrLibConfig.cmake @ONLY)

#install flwrLibConfig.cmake
install(FILES ${PROJECT_BINARY_DIR}/flwrLibConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flwrLib)
