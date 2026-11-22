cmake_minimum_required(VERSION 3.20)
project(numkit_value VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NUMKIT_BUILD_TESTS "Build GoogleTest suite" OFF)
option(NUMKIT_VALUE_BUILD_TESTS "Build C-ABI GoogleTest suite for numkit/value" ${NUMKIT_BUILD_TESTS})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
    if(NOT TARGET keel::keel)
        if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
            foreach(_keel_cand
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../../keelcpp/keel"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../keelcpp/keel"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../keelcpp/keel"
            )
                if(EXISTS "${_keel_cand}/CMakeLists.txt")
                    set(FETCHCONTENT_SOURCE_DIR_KEEL "${_keel_cand}" CACHE PATH "")
                    break()
                endif()
            endforeach()
        endif()
        include(FetchContent)
        set(KEEL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(KEEL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            keel
            GIT_REPOSITORY https://github.com/keelcpp/keel.git
            GIT_TAG        869a5c07a5d77341e5f6acbac5f8e024b274ea08
        )
        FetchContent_MakeAvailable(keel)
    endif()
endif()

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

target_link_libraries(numkit_value PUBLIC keel::keel)

if(EMSCRIPTEN)
    target_compile_options(numkit_value PRIVATE -fexceptions)
endif()

if(COMMAND keel_enforce_guidelines)
    keel_enforce_guidelines(numkit_value
        SCOPE ALL
        BAN_RAW_INTS ALL
        ALLOW_STDEXCEPT OFF
        ALLOW_STDEXCEPT_IN_HEADERS OFF
        HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ALLOWED_ANGLE_PREFIXES "numkit/"
        EXCLUDE_FILES "c_api\\.h$" "c_api\\.cpp$"
        STANDALONE_HEADERS ${NUMKIT_VALUE_BUILD_TESTS}
        WARNINGS OFF
    )
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
