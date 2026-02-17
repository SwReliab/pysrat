cmake_minimum_required(VERSION 3.18)
project(pysrat LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(srat_core STATIC
  csrc/utils/numlib.c
  csrc/utils/inv_log_psi.c
  csrc/utils/gauss_inte.c
  csrc/utils/logistic.c
  csrc/utils/norm.c
)
target_include_directories(srat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils)
set_target_properties(srat_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_core
  csrc/basic/basic_pybind.cpp
  csrc/basic/em_exp.cpp
  csrc/basic/em_tnorm.cpp
  csrc/basic/em_pareto.cpp
  csrc/basic/em_lnorm.cpp
  csrc/basic/em_tlogis.cpp
  csrc/basic/em_llogis.cpp
  csrc/basic/em_gamma.cpp
  csrc/basic/em_txvmax.cpp
  csrc/basic/gumbel.cpp
  csrc/basic/txvmin.cpp
  csrc/basic/lxvmax.cpp
  csrc/basic/lxvmin.cpp
)
target_link_libraries(_core PRIVATE srat_core)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils)

pybind11_add_module(marlib_cf1
  csrc/cf1/cf1_pybind.cpp
  csrc/cf1/poisson.cpp
)
target_compile_features(marlib_cf1 PRIVATE cxx_std_20)
target_link_libraries(marlib_cf1 PRIVATE srat_core)
target_include_directories(marlib_cf1 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cf1
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils
)

install(TARGETS _core marlib_cf1
  LIBRARY DESTINATION pysrat
  ARCHIVE DESTINATION pysrat
  RUNTIME DESTINATION pysrat
)