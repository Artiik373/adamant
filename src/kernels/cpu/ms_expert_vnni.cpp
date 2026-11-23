// src/kernels/cpu/ms_expert_vnni.cpp - ms_rows.inc built for AVX-VNNI.  Only called behind cpu_avxvnni_ok().
#include "basalt/kernels/cpu/ms_expert.hpp"

#include <immintrin.h>

#include <cstring>

#define BASALT_MS_VNNI 1
#define BASALT_MS_ROWS rows_vnni
#include "ms_rows.inc"
