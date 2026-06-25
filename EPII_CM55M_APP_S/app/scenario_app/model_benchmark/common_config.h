/*
 * common_config.h
 *
 * Model Benchmark Configuration
 */

#ifndef SCENARIO_APP_MODEL_BENCHMARK_COMMON_CONFIG_H_
#define SCENARIO_APP_MODEL_BENCHMARK_COMMON_CONFIG_H_

// Default single-model address kept for compatibility during the transition to
// the flash model table flow.
#define MODEL_FLASH_ADDR 0x3AB7B000

// Flash model table address.
#define MODEL_TABLE_FLASH_ADDR    0x3A200000

// Max number of model records read from flash.
#define MAX_MODEL_COUNT           20

// Tensor arena size used by the legacy single-model path.
#define TENSOR_ARENA_SIZE (1024 * 1024)

// Upper bound used by the multi-model benchmark runtime.
#define TENSOR_ARENA_SIZE_MAX     (2 * 1024 * 1024)

// Number of benchmark runs for averaging
#define NUM_BENCHMARK_RUNS 10

// Warm-up runs before measured inference.
#define WARMUP_RUNS               2

// Watchdog timeout in ms
#define WATCH_DOG_TIMEOUT_TH (500)

#endif /* SCENARIO_APP_MODEL_BENCHMARK_COMMON_CONFIG_H_ */
