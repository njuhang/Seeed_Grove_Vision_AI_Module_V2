/*
 * model_benchmark.c
 *
 * TFLM Model Benchmark Application
 * Benchmarks model inference using random input data.
 *
 * All TFLM (C++) work lives in cvapp_model_benchmark.cpp; this file stays in
 * plain C and talks to it through cvapp_model_benchmark.h.
 */

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "powermode_export.h"

#define WATCH_DOG_TIMEOUT_TH	(500) //ms

#ifdef TRUSTZONE_SEC
#ifdef FREERTOS
#else
#if (__ARM_FEATURE_CMSE & 1) == 0
#error "Need ARMv8-M security extensions"
#elif (__ARM_FEATURE_CMSE & 2) == 0
#error "Compile with --cmse"
#endif
#include "arm_cmse.h"
#endif
#endif

#include "WE2_device.h"

#include "spi_master_protocol.h"
#include "hx_drv_spi.h"
#include "spi_eeprom_comm.h"
#include "board.h"
#include "xprintf.h"
#include "model_benchmark.h"
#include "board.h"
#include "WE2_core.h"
#include "hx_drv_scu.h"
#include "hx_drv_swreg_aon.h"
#ifdef IP_sensorctrl
#include "hx_drv_sensorctrl.h"
#endif
#ifdef IP_xdma
#include "hx_drv_xdma.h"
#include "sensor_dp_lib.h"
#endif
#ifdef IP_cdm
#include "hx_drv_cdm.h"
#endif
#ifdef IP_gpio
#include "hx_drv_gpio.h"
#endif
#include "hx_drv_pmu_export.h"
#include "hx_drv_pmu.h"
#include "powermode.h"
#include "BITOPS.h"

#include "event_handler.h"
#include "memory_manage.h"
#include "hx_drv_watchdog.h"

#include "common_config.h"
#include "cvapp_model_benchmark.h"
#include "model_table.h"

// SystemTick wraps a 24-bit down-counter (0xffffff) plus a loop counter that
// increments on each wrap. CPU_CLK mirrors cvapp_model_benchmark.cpp so the
// cycles arithmetic here is identical to the per-inference path.
#define CPU_CLK (0xffffff + 1)

#ifdef EPII_FPGA
#define DBG_APP_LOG             (1)
#else
#define DBG_APP_LOG             (0)
#endif
#if DBG_APP_LOG
    #define dbg_app_log(fmt, ...)       xprintf(fmt, ##__VA_ARGS__)
#else
    #define dbg_app_log(fmt, ...)
#endif

static uint32_t g_mm_start_addr = 0;
static uint32_t g_mm_size = 0;
static uint32_t g_cpu_freq = 0;
static model_table_t g_model_table;

typedef struct {
	uint32_t arena_used_bytes;
	uint32_t avg_latency_ms_x1000;
	uint32_t min_latency_ms_x1000;
	uint32_t max_latency_ms_x1000;
	uint32_t runs;
	const char *status;
} model_benchmark_result_t;

// Snapshot of the SystemTick at app_main start, used to derive a monotonic
// uptime (ms since power-on). The board has no RTC, so the JSON timestamp
// field reports uptime rather than wall-clock time.
static uint32_t g_boot_systick = 0;
static uint32_t g_boot_loop_cnt = 0;

void pinmux_init();

/* Init SPI master pin mux (share with SDIO) */
void spi_m_pinmux_cfg(SCU_PINMUX_CFG_T *pinmux_cfg)
{
	pinmux_cfg->pin_pb2 = SCU_PB2_PINMUX_SPI_M_DO_1;
	pinmux_cfg->pin_pb3 = SCU_PB3_PINMUX_SPI_M_DI_1;
	pinmux_cfg->pin_pb4 = SCU_PB4_PINMUX_SPI_M_SCLK_1;
	pinmux_cfg->pin_pb11 = SCU_PB11_PINMUX_SPI_M_CS;
}

void pinmux_init()
{
	SCU_PINMUX_CFG_T pinmux_cfg;

	hx_drv_scu_get_all_pinmux_cfg(&pinmux_cfg);
	spi_m_pinmux_cfg(&pinmux_cfg);
	hx_drv_scu_set_all_pinmux_cfg(&pinmux_cfg, 1);
}

/**
 * @brief Run benchmark and print results
 */
static void benchmark_memory_reset(void)
{
	mm_set_initial(g_mm_start_addr, g_mm_size);
}

static uint32_t benchmark_arena_size_limit(void)
{
	if (g_mm_size < TENSOR_ARENA_SIZE_MAX) {
		return g_mm_size;
	}

	return TENSOR_ARENA_SIZE_MAX;
}

static uint32_t cycles_to_ms_x1000(uint32_t cycles)
{
	if (g_cpu_freq == 0) {
		return 0;
	}

	return (uint32_t)(((uint64_t)cycles * 1000000ULL) / g_cpu_freq);
}

/**
 * @brief Like cycles_to_ms() but for a 64-bit cycle count.
 *
 * The per-inference path stays on the uint32 helper above; this one is for the
 * cumulative uptime, where the cycle count is too wide for uint32.
 */
static uint64_t cycles_to_ms_u64(uint64_t cycles)
{
	if (g_cpu_freq == 0) {
		return 0;
	}

	return (cycles * 1000ULL) / g_cpu_freq;
}

/**
 * @brief Milliseconds elapsed since the boot tick recorded in app_main.
 *
 * Uses the same SystemTick arithmetic as cv_model_benchmark_run_inference():
 * the 24-bit systick is a down-counter so a later reading is smaller, while
 * loop_cnt increments on each wrap. cycles therefore equals
 * (loop_now - loop_boot) * CPU_CLK + (systick_boot - systick_now). No RTC is
 * available, so this monotonic uptime is what the JSON timestamp reports.
 *
 * Unlike the per-inference path, this is a *cumulative* count (ms since boot),
 * so it is computed in uint64_t. CPU_CLK is ~16.77M (0xffffff+1) and the
 * per-wrap term (loop_cnt_diff * CPU_CLK) would overflow uint32 after ~10.7s;
 * uint64_t widens that to roughly millions of years, so the timestamp stays
 * monotonic well beyond any realistic device session. The operands are cast to
 * uint64_t *before* the multiply so the product never wraps in uint32 either.
 */
static uint32_t uptime_ms(void)
{
	uint32_t systick_now;
	uint32_t loop_cnt_now;

	SystemGetTick(&systick_now, &loop_cnt_now);

	uint64_t cycles =
		(uint64_t)(loop_cnt_now - g_boot_loop_cnt) * (uint64_t)CPU_CLK +
		(uint64_t)(g_boot_systick - systick_now);

	return (uint32_t)cycles_to_ms_u64(cycles);
}

static void run_legacy_benchmark(void) {
	xprintf("\r\n=== Model Benchmark ===\r\n");
	xprintf("Running %d inference iterations...\r\n", NUM_BENCHMARK_RUNS);

	uint32_t min_time = 0xFFFFFFFF;
	uint32_t max_time = 0;
	uint32_t total_time = 0;

	for (int i = 0; i < NUM_BENCHMARK_RUNS; i++) {
		cv_model_benchmark_fill_random_input();

		uint32_t time = cv_model_benchmark_run_inference();

		if (time > 0) {
			total_time += time;
			if (time < min_time) min_time = time;
			if (time > max_time) max_time = time;
		}

		xprintf("Run %d: %u cycles\r\n", i + 1, time);
	}

	uint32_t avg_time = total_time / NUM_BENCHMARK_RUNS;

	xprintf("\r\n=== Benchmark Results ===\r\n");
	xprintf("Average: %u cycles\r\n", avg_time);
	xprintf("Min:     %u cycles\r\n", min_time);
	xprintf("Max:     %u cycles\r\n", max_time);
	xprintf("========================\r\n\r\n");
}

static void print_model_result_prefix(const model_table_entry_t *entry, bool first_entry)
{
	const char *task = entry->task[0] == '\0' ? "unknown" : entry->task;
	if (!first_entry) {
		xprintf(",");
	}

	xprintf(
		"{\"name\":\"%s\",\"task\":\"%s\",\"model_size_bytes\":%u,",
		entry->name,
		task,
		entry->model_size_bytes);
}

static void print_ms_x1000(uint32_t value)
{
	xprintf("%u.%03u", value / 1000U, value % 1000U);
}

static void benchmark_one_model(const model_table_entry_t *entry, model_benchmark_result_t *out_result)
{
	uint32_t min_cycles = 0xFFFFFFFF;
	uint32_t max_cycles = 0;
	uint64_t total_cycles = 0;
	uint32_t successful_runs = 0;

	out_result->arena_used_bytes = 0;
	out_result->avg_latency_ms_x1000 = 0;
	out_result->min_latency_ms_x1000 = 0;
	out_result->max_latency_ms_x1000 = 0;
	out_result->runs = 0;
	out_result->status = "invoke_failed";

	xprintf("benchmark_one_model: start %s @0x%08x\r\n", entry->name, 0x3A000000 + entry->flash_addr);
	benchmark_memory_reset();
	if (cv_model_benchmark_load_model(0x3A000000 + entry->flash_addr,
			benchmark_arena_size_limit()) != 0) {
		xprintf("benchmark_one_model: load_failed %s\r\n", entry->name);
		out_result->status = "load_failed";
		return;
	}
	xprintf("benchmark_one_model: load_ok %s arena=%u\r\n", entry->name, cv_model_benchmark_arena_bytes_used());

	out_result->arena_used_bytes = cv_model_benchmark_arena_bytes_used();
	cv_model_benchmark_fill_zero_input();
	for (int i = 0; i < WARMUP_RUNS; ++i) {
		xprintf("benchmark_one_model: warmup %s %d/%d\r\n", entry->name, i + 1, WARMUP_RUNS);
		if (cv_model_benchmark_run_inference() == 0) {
			xprintf("benchmark_one_model: warmup_invoke_failed %s\r\n", entry->name);
			cv_model_benchmark_unload_model();
			return;
		}
	}

	for (int i = 0; i < NUM_BENCHMARK_RUNS; ++i) {
		xprintf("benchmark_one_model: run %s %d/%d\r\n", entry->name, i + 1, NUM_BENCHMARK_RUNS);
		uint32_t cycles = cv_model_benchmark_run_inference();
		if (cycles == 0) {
			xprintf("benchmark_one_model: measured_invoke_failed %s at run %d\r\n", entry->name, i + 1);
			cv_model_benchmark_unload_model();
			return;
		}

		total_cycles += cycles;
		if (cycles < min_cycles) min_cycles = cycles;
		if (cycles > max_cycles) max_cycles = cycles;
		++successful_runs;
	}

	if (successful_runs == 0) {
		cv_model_benchmark_unload_model();
		return;
	}

	out_result->avg_latency_ms_x1000 = cycles_to_ms_x1000((uint32_t)(total_cycles / successful_runs));
	out_result->min_latency_ms_x1000 = cycles_to_ms_x1000(min_cycles);
	out_result->max_latency_ms_x1000 = cycles_to_ms_x1000(max_cycles);
	out_result->runs = successful_runs;
	out_result->status = "ok";
	xprintf("benchmark_one_model: done %s runs=%u\r\n", entry->name, successful_runs);

	cv_model_benchmark_unload_model();
}

static void print_model_result(
	const model_table_entry_t *entry,
	const model_benchmark_result_t *result,
	bool first_entry)
{
	print_model_result_prefix(entry, first_entry);
	xprintf("\"arena_used_bytes\":%u,\"latency_ms\":{\"avg\":", result->arena_used_bytes);
	print_ms_x1000(result->avg_latency_ms_x1000);
	xprintf(",\"min\":");
	print_ms_x1000(result->min_latency_ms_x1000);
	xprintf(",\"max\":");
	print_ms_x1000(result->max_latency_ms_x1000);
	xprintf("},\"runs\":%u,\"status\":\"%s\"}", result->runs, result->status);
}

static void run_multi_model_benchmark(void)
{
	model_benchmark_result_t results[MAX_MODEL_COUNT];

	for (uint32_t i = 0; i < g_model_table.count; ++i) {
		benchmark_one_model(&g_model_table.entries[i], &results[i]);
	}

	xprintf(
		"{\"benchmark\":{\"device\":\"himax_hx6538\",\"firmware\":\"model_benchmark_v1\","
		"\"timestamp\":%lu,\"timestamp_unit\":\"ms_since_boot\"},\"models\":[",
		(unsigned long)uptime_ms());

	for (uint32_t i = 0; i < g_model_table.count; ++i) {
		print_model_result(&g_model_table.entries[i], &results[i], i == 0);
	}

	xprintf("]}\r\n");
}

/**
 * @brief Main application entry point
 */
int app_main(void) {
	uint32_t wakeup_event;
	uint32_t wakeup_event1;
	uint32_t freq = 0;

	hx_drv_pmu_get_ctrl(PMU_pmu_wakeup_EVT, &wakeup_event);
	hx_drv_pmu_get_ctrl(PMU_pmu_wakeup_EVT1, &wakeup_event1);

	hx_drv_swreg_aon_get_pllfreq(&freq);
	g_cpu_freq = freq;
	// Record the boot tick as early as possible so uptime stays monotonic and
	// distinguishes repeated runs. Must run after g_cpu_freq is set so that
	// uptime_ms() can convert cycles.
	SystemGetTick(&g_boot_systick, &g_boot_loop_cnt);
	xprintf("wakeup_event=0x%x, WakeupEvt1=0x%x, freq=%d\n", wakeup_event, wakeup_event1, freq);

	pinmux_init();

	// Initialize memory manager
#ifdef __GNU__
	xprintf("__GNUC \n");
	extern char __mm_start_addr__;
	xprintf("__mm_start_addr__ address: %x\r\n", &__mm_start_addr__);
	g_mm_start_addr = (uint32_t)(&__mm_start_addr__);
	g_mm_size = 0x00200000 - (((uint32_t)(&__mm_start_addr__)) - 0x34000000);
#else
	static uint8_t mm_start_addr __attribute__((section(".bss.mm_start_addr")));
	xprintf("mm_start_addr address: %x \r\n", &mm_start_addr);
	g_mm_start_addr = (uint32_t)(&mm_start_addr);
	g_mm_size = 0x00200000 - (((uint32_t)(&mm_start_addr)) - 0x34000000);
#endif
	benchmark_memory_reset();

	// Initialize SPI for flash access
	hx_lib_spi_eeprom_open(USE_DW_SPI_MST_Q);
	hx_lib_spi_eeprom_enable_XIP(USE_DW_SPI_MST_Q, true, FLASH_QUAD, true);

	xprintf("\r\n========================================\r\n");
	xprintf("  Model Benchmark Application\r\n");
	xprintf("  Model address: 0x%x\r\n", MODEL_FLASH_ADDR);
	xprintf("========================================\r\n\r\n");

	if (model_table_load(MODEL_TABLE_FLASH_ADDR, &g_model_table)) {
		xprintf("Loaded model table with %u model(s)\r\n", g_model_table.count);
		if (cv_model_benchmark_runtime_init(true, true) != 0) {
			xprintf("Model benchmark runtime init failed!\r\n");
			APP_BLOCK_FUNC();
		}

		while (1) {
			run_multi_model_benchmark();
			for (volatile int i = 0; i < 10000000; i++);
		}
	}

	if (cv_model_benchmark_init(true, true, MODEL_FLASH_ADDR, TENSOR_ARENA_SIZE) != 0) {
		xprintf("Model benchmark init failed!\r\n");
		APP_BLOCK_FUNC();
	}

	while (1) {
		run_legacy_benchmark();

		// Wait before next benchmark
		for (volatile int i = 0; i < 10000000; i++);
	}

	return 0;
}
