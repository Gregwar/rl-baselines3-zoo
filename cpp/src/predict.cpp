#include "baselines3_models/preprocessing.h"
#include "baselines3_models/approach_v0.h"
#include <chrono>
#include <iostream>
#include <memory>
#include "cmrc/cmrc.hpp"

using namespace baselines3_models;
using namespace torch::indexing;

int main(int argc, const char *argv[]) {
  approach_v0 approach;

  torch::Tensor observation = torch::tensor({-1., 0., 0., 1., 0., 1., 0., 0., 0.});
  torch::Tensor action = approach.predict(observation);

  std::cout << (action) << std::endl;
}