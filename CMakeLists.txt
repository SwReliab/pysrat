cmake_minimum_required(VERSION 3.18)
project(pysrat LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Cコア（例として同梱）
add_library(srat_core STATIC
  csrc/srat_core.c
)
target_include_directories(srat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

# Python拡張モジュール
pybind11_add_module(_core
  src/bindings.cpp
  cppsrc/em_exp.cpp
  cppsrc/em_tnorm.cpp
)
target_link_libraries(_core PRIVATE srat_core Python::Python)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

install(TARGETS _core
  LIBRARY DESTINATION pysrat
  ARCHIVE DESTINATION pysrat
  RUNTIME DESTINATION pysrat
)
