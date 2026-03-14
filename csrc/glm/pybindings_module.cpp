#include <pybind11/pybind11.h>

#include "bind_glm.h"

namespace py = pybind11;

PYBIND11_MODULE(_glm, m) {
  m.doc() = "pysrat GLM helpers";
  bind_glm_binomial(m);
  bind_glm_binomial_elasticnet(m);
  bind_glm_poisson(m);
  bind_glmnet_poisson(m);
}
