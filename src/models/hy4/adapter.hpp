// src/models/hy4/adapter.hpp - the Hy4-flash adapter: its own start, serving the line protocol over hy4::Engine.
#pragma once

#include "basalt/model/adapter.hpp"

#include <memory>

namespace basalt::models::hy4 {
std::unique_ptr<model::ModelAdapter> make_adapter();
}
