// tests/model/adapter_builtin_test.cpp - the program's own registration: the real adapter, once, and its load contract.
// Linking the models needs the CUDA runtime, so this runs where that is installed; no GPU work.
#include "basalt/model/adapter.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/models/builtin.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <memory>

int main() {
    using namespace basalt;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    models::register_builtin_models();
    models::register_builtin_models();
    check(model::find_adapter("qwen4exp") != nullptr, "the qwen4exp adapter is registered");
    check(model::find_adapter("hy4") != nullptr, "the hy4 adapter is registered");
    check(model::adapter_names().size() == 2, "registering twice keeps one adapter per name");
    if (const model::ModelAdapter* qw = model::find_adapter("qwen4exp")) {
        std::unique_ptr<model::LoadedModel> loaded;
        check(qw->load({}, loaded).status == model::Status::NotImplemented, "qwen4exp has no shared-server load: it runs its own start");
    }
    if (const model::ModelAdapter* hy = model::find_adapter("hy4")) {
        std::unique_ptr<model::LoadedModel> loaded;
        check(hy->load({}, loaded).status == model::Status::NotImplemented, "hy4 has no shared-server load: it runs its own start");
        auto opts = hy->make_options();
        check(opts != nullptr, "hy4 make_options returns options");
        check(hy->planned_spec(*opts) == 6, "hy4 default planned_spec is 6");
        int code = 0;
        int i1 = 0;
        char* f1[] = {(char*) "--trunk", (char*) "q6_k"};
        model::FlagArgs a1(2, f1, i1);
        check(hy->parse_flag("--trunk", a1, *opts, code), "hy4 parses --trunk q6_k");
        check(code == 0, "code is 0 after valid --trunk");
        int i2 = 0;
        char* f2[] = {(char*) "--spec", (char*) "8"};
        model::FlagArgs a2(2, f2, i2);
        check(hy->parse_flag("--spec", a2, *opts, code), "hy4 parses --spec 8");
        check(hy->planned_spec(*opts) == 6, "hy4 planned_spec with --spec 8 clamps to 6");
        int i3 = 0;
        char* f3[] = {(char*) "--spec", (char*) "0"};
        model::FlagArgs a3(2, f3, i3);
        check(hy->parse_flag("--spec", a3, *opts, code), "hy4 parses --spec 0");
        check(hy->planned_spec(*opts) == 0, "hy4 planned_spec with --spec 0 is 0");
        int i4 = 0;
        char* f4[] = {(char*) "--relaxed-drafting", (char*) "0.5"};
        model::FlagArgs a4(2, f4, i4);
        check(hy->parse_flag("--relaxed-drafting", a4, *opts, code), "hy4 parses --relaxed-drafting");
    }
    return failures ? 1 : 0;
}
