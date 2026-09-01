cmake_minimum_required(VERSION 3.16)
project(maze_sample_pool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

set(SOURCES
    main/main.cpp
    src/config/config_loader.cpp
    src/grpc/sample_pool_service.cpp
    src/store/sample_pool_coordinator.cpp
    src/store/sample_store_backend.cpp
)

set(CONTRACT_CPP_DIR "" CACHE PATH "Directory containing generated C++ bindings")
if(NOT CONTRACT_CPP_DIR)
    message(FATAL_ERROR "CONTRACT_CPP_DIR must point to an rl-training-contracts C++ artifact")
endif()

set(PROTO_GEN_DIR ${CONTRACT_CPP_DIR})
set(PROTO_SOURCES
    ${PROTO_GEN_DIR}/common.pb.cc
    ${PROTO_GEN_DIR}/training.pb.cc
    ${PROTO_GEN_DIR}/training.grpc.pb.cc
)

add_executable(${PROJECT_NAME} ${SOURCES} ${PROTO_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROTO_GEN_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    protobuf::libprotobuf
    OpenSSL::Crypto
    ${GRPC_LIBRARIES}
    pthread
)

include(CTest)
if(BUILD_TESTING)
    add_executable(sample_pool_development_test
        tests/sample_store_test.cpp
        src/config/config_loader.cpp
        src/store/sample_pool_coordinator.cpp
        src/store/sample_store_backend.cpp
        ${PROTO_GEN_DIR}/common.pb.cc
        ${PROTO_GEN_DIR}/training.pb.cc
    )
    target_include_directories(sample_pool_development_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GEN_DIR}
        ${Protobuf_INCLUDE_DIRS}
    )
    target_link_libraries(sample_pool_development_test PRIVATE
        protobuf::libprotobuf
        OpenSSL::Crypto
        pthread
    )
    add_test(
        NAME sample_pool_development_contract
        COMMAND sample_pool_development_test
    )
endif()
