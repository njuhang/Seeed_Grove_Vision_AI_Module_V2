/*
 * cvapp_model_benchmark.h
 *
 * C bridge to the TFLM-based benchmark implementation in cvapp_model_benchmark.cpp.
 * Keeping all C++/TFLM code behind this header lets model_benchmark.c stay pure C.
 */

#ifndef SCENARIO_APP_MODEL_BENCHMARK_CVAPP_H_
#define SCENARIO_APP_MODEL_BENCHMARK_CVAPP_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int cv_model_benchmark_runtime_init(bool security_enable, bool privilege_enable);
int cv_model_benchmark_load_model(uint32_t model_addr, uint32_t arena_size);
void cv_model_benchmark_fill_zero_input(void);
uint32_t cv_model_benchmark_arena_bytes_used(void);
uint32_t cv_model_benchmark_model_bytes(void);
void cv_model_benchmark_unload_model(void);

/**
 * @brief Legacy wrapper that initializes the runtime and optionally loads one
 * model immediately.
 * @return 0 on success, -1 on failure.
 */
int cv_model_benchmark_init(bool security_enable, bool privilege_enable,
                            uint32_t model_addr, uint32_t arena_size);

/** Fill the model input tensor with random int8 data. */
void cv_model_benchmark_fill_random_input(void);

/**
 * @brief Run a single inference and return elapsed cycles.
 * @return cycle count, or 0 if Invoke() failed.
 */
uint32_t cv_model_benchmark_run_inference(void);

#ifdef __cplusplus
}
#endif

#endif /* SCENARIO_APP_MODEL_BENCHMARK_CVAPP_H_ */
