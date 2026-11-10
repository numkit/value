cmake_minimum_required(VERSION 3.20)
project(numkit_value VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NUMKIT_BUILD_TESTS "Build GoogleTest suite" OFF)
option(NUMKIT_VALUE_BUILD_TESTS "Build C-ABI GoogleTest suite for numkit/value" ${NUMKIT_BUILD_TESTS})

add_library(numkit_value STATIC
    src/data_buffer.cpp
    src/dims.cpp
    src/heap_object.cpp
    src/value.cpp
    src/value_type.cpp
)

add_library(numkit::value ALIAS numkit_value)

target_include_directories(numkit_value PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(EMSCRIPTEN)
    target_compile_options(numkit_value PRIVATE -fexceptions)
endif()

# Shared C-ABI library owning `numkit_val*` allocations across all modules
add_library(numkit_value_c SHARED
    src/c_api.cpp
)
add_library(numkit::value_c ALIAS numkit_value_c)

target_compile_definitions(numkit_value_c PRIVATE NUMKIT_VALUE_C_BUILD=1)
target_link_libraries(numkit_value_c PUBLIC numkit_value)

if(MSVC)
    target_compile_options(numkit_value_c PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_value_c PRIVATE -Wall -Wextra -fPIC)
endif()

# Installation rules
include(GNUInstallDirs)
install(TARGETS numkit_value numkit_value_c
    EXPORT numkit_value_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(NUMKIT_VALUE_BUILD_TESTS AND NUMKIT_BUILD_TESTS)
    enable_testing()
    if(NOT TARGET GTest::gtest_main)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(numkit_value_c_test
        tests/c_api/value_c_api.cpp
    )
    target_link_libraries(numkit_value_c_test PRIVATE numkit_value_c GTest::gtest_main)
    add_test(NAME numkit_value_c_test COMMAND numkit_value_c_test)
endif()
