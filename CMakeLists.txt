cmake_minimum_required(VERSION 3.20)
project(numkit_value VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(nk_value STATIC
    src/data_buffer.cpp
    src/dims.cpp
    src/heap_object.cpp
    src/value.cpp
    src/value_type.cpp
)

add_library(nk::value ALIAS nk_value)

target_include_directories(nk_value PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(EMSCRIPTEN)
    target_compile_options(nk_value PRIVATE -fexceptions)
endif()

# Installation rules
include(GNUInstallDirs)
install(TARGETS nk_value
    EXPORT numkit_value_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
