cmake_minimum_required(VERSION 3.18)
project(pysrat LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# C core (bundled for convenience)
add_library(srat_core STATIC
  csrc/numlib.c
  csrc/inv_log_psi.c
  csrc/gauss_inte.c
  csrc/logistic.c
  csrc/norm.c
)
target_include_directories(srat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

# Python extension module
pybind11_add_module(_core
  cppsrc/basic_pybind.cpp
  cppsrc/em_exp.cpp
  cppsrc/em_tnorm.cpp
  cppsrc/em_pareto.cpp
  cppsrc/em_lnorm.cpp
  cppsrc/em_tlogis.cpp
  cppsrc/em_llogis.cpp
  cppsrc/em_gamma.cpp
  cppsrc/em_txvmax.cpp
  cppsrc/gumbel.cpp
  cppsrc/txvmin.cpp
  cppsrc/lxvmax.cpp
  cppsrc/lxvmin.cpp
)
target_link_libraries(_core PRIVATE srat_core Python::Python)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

# CF1 span backend module (C++20)
pybind11_add_module(marlib_cf1
  phsrc/cf1_pybind.cpp
  phsrc/poisson.cpp
)
target_compile_features(marlib_cf1 PRIVATE cxx_std_20)
target_include_directories(marlib_cf1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phsrc)

install(TARGETS _core
  LIBRARY DESTINATION pysrat
  ARCHIVE DESTINATION pysrat
  RUNTIME DESTINATION pysrat
)

install(TARGETS marlib_cf1
  LIBRARY DESTINATION pysrat
  ARCHIVE DESTINATION pysrat
  RUNTIME DESTINATION pysrat
)
