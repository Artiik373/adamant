// src/kernels/cpu/exl3_vnni.cpp - exl3_cols.inc built for AVX-VNNI.  Only called behind cpu_avxvnni_ok().
#include "exl3_ring.hpp"

#include <immintrin.h>

#include <cstring>

#define BASALT_EXL3_VNNI 1
#define BASALT_EXL3_COLS cols_avx2_vnni
#include "exl3_cols.inc"
