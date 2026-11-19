#include "basalt/models/builtin.hpp"

#include "basalt/model/adapter.hpp"
#include "models/hy4/adapter.hpp"
#include "models/qwen4exp/adapter.hpp"

namespace basalt::models {

void register_builtin_models() {
    if (model::find_adapter("qwen4exp") == nullptr) model::register_adapter(qwen4exp::make_adapter());
    if (model::find_adapter("hy4") == nullptr) model::register_adapter(hy4::make_adapter());
}

}  // namespace basalt::models
