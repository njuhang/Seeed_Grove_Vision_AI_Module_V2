/*
 * model_benchmark.h
 *
 * Model Benchmark Application Header
 */

#ifndef SCENARIO_APP_MODEL_BENCHMARK_H_
#define SCENARIO_APP_MODEL_BENCHMARK_H_

#define APP_BLOCK_FUNC() do{ \
	__asm volatile("b    .");\
	}while(0)

int app_main(void);

#endif /* SCENARIO_APP_MODEL_BENCHMARK_H_ */