#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_glm_binomial(py::module_& m);
void bind_glm_poisson(py::module_& m);
void bind_glmnet_binomial(py::module_& m);
void bind_glmnet_poisson(py::module_& m);
