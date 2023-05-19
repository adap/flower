cmake_minimum_required(VERSION 3.16)
project(SimpleCppFlowerClient VERSION 0.10
  DESCRIPTION "Creates a Simple C++ Flower client that trains a linear model on synthetic data."
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(ABSL_PROPAGATE_CXX_STD ON)

find_package(flwrLib REQUIRED)

######################
### FLWR_CLIENT
file(GLOB FLWR_CLIENT_SRCS src/*.cc)
set(EXECUTABLE_NAME flwr_client)
add_executable(${EXECUTABLE_NAME} ${FLWR_CLIENT_SRCS})

target_include_directories(${EXECUTABLE_NAME} PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${FLWR_INCLUDE_DIRS}
)


target_link_libraries(${EXECUTABLE_NAME}
  ${FLWR_LIBRARIES}
)
