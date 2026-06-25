/*
 * cvapp_model_benchmark.cpp
 *
 * TFLM (C++) glue for the model_benchmark scenario app. The pure-C entry point
 * in model_benchmark.c calls into this file through cvapp_model_benchmark.h.
 */

#include <cstdio>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <new>

#include "WE2_device.h"
#include "WE2_core.h"
#include "board.h"
#include "xprintf.h"
#include "memory_manage.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

#include "cvapp_model_benchmark.h"

// NPU base address (mirrors the original logic in model_benchmark.c)
#ifdef TRUSTZONE_SEC
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE	BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE	BASE_ADDR_APB_U55_CTRL
#endif
#endif

#define CPU_CLK	(0xffffff + 1)

namespace {
struct ethosu_driver ethosu_drv;
bool runtime_initialized = false;
bool runtime_security_enable = true;
bool runtime_privilege_enable = true;
tflite::MicroInterpreter *interpreter_ptr = nullptr;
TfLiteTensor *input_tensor = nullptr;
uint32_t tensor_arena_addr = 0;
uint32_t tensor_arena_size = 0;
uint32_t current_model_bytes = 0;
uint32_t current_arena_bytes = 0;
alignas(tflite::MicroInterpreter) uint8_t interpreter_storage[sizeof(tflite::MicroInterpreter)];
bool op_resolver_initialized = false;
// Spec 4.8: register a comprehensive op set so models with CPU fallback ops
// (not 100% Vela-compiled) run too. Capacity must stay >= the number of ops
// registered in cv_model_benchmark_runtime_init(); the tools/tests contract
// test guards this. ReduceMin is unavailable in this TFLM build (only
// AddMinimum), so it is intentionally omitted.
static tflite::MicroMutableOpResolver<34> op_resolver;

void _arm_npu_irq_handler(void)
{
	ethosu_irq_handler(&ethosu_drv);
}

void _arm_npu_irq_init(void)
{
	const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
	EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
	NVIC_EnableIRQ(ethosu_irqnum);
}

int _arm_npu_init(bool security_enable, bool privilege_enable)
{
	int err = 0;

	_arm_npu_irq_init();

#if TFLM2209_U55TAG2205
	const void * ethosu_base_address = (void *)(U55_BASE);
#else
	void * const ethosu_base_address = (void *)(U55_BASE);
#endif

	if (0 != (err = ethosu_init(
						&ethosu_drv,
						ethosu_base_address,
						NULL,
						0,
						security_enable,
						privilege_enable))) {
		return err;
	}

	return 0;
}

void reset_runtime_state(void)
{
	interpreter_ptr = nullptr;
	input_tensor = nullptr;
	tensor_arena_addr = 0;
	tensor_arena_size = 0;
	current_model_bytes = 0;
	current_arena_bytes = 0;
}
}  // namespace

extern "C" int cv_model_benchmark_runtime_init(bool security_enable, bool privilege_enable)
{
	if (runtime_initialized) {
		return 0;
	}

	if (_arm_npu_init(security_enable, privilege_enable) != 0) {
		return -1;
	}

	if (!op_resolver_initialized) {
		// Comprehensive op set per spec 4.8. Vela-routable ops are fused into
		// the EthosU custom op at compile time; these registrations cover the
		// ops that remain as native CPU ops (fallback) in real models.
		op_resolver.AddAdd();
		op_resolver.AddBroadcastTo();
		op_resolver.AddConv2D();
		op_resolver.AddDepthwiseConv2D();
		op_resolver.AddDequantize();
		op_resolver.AddReshape();
		op_resolver.AddSoftmax();
		op_resolver.AddTranspose();
		op_resolver.AddMaxPool2D();
		op_resolver.AddAveragePool2D();
		op_resolver.AddFullyConnected();
		op_resolver.AddConcatenation();
		op_resolver.AddMul();
		op_resolver.AddRelu();
		op_resolver.AddResizeBilinear();
		op_resolver.AddResizeNearestNeighbor();
		op_resolver.AddSplit();
		op_resolver.AddPad();
		op_resolver.AddPrelu();
		op_resolver.AddStridedSlice();
		op_resolver.AddGather();
		op_resolver.AddGatherNd();
		op_resolver.AddExp();
		op_resolver.AddLog();
		op_resolver.AddLeakyRelu();
		op_resolver.AddHardSwish();
		op_resolver.AddQuantize();
		op_resolver.AddReduceMax();
		op_resolver.AddSum();
		op_resolver.AddMean();
		op_resolver.AddArgMax();
		op_resolver.AddBatchMatMul();
		op_resolver.AddL2Normalization();
		if (kTfLiteOk != op_resolver.AddEthosU()) {
			return -1;
		}
		op_resolver_initialized = true;
	}

	runtime_security_enable = security_enable;
	runtime_privilege_enable = privilege_enable;
	runtime_initialized = true;
	reset_runtime_state();
	return 0;
}

extern "C" void cv_model_benchmark_unload_model(void)
{
	if (interpreter_ptr != nullptr) {
		interpreter_ptr->~MicroInterpreter();
	}
	reset_runtime_state();
}

extern "C" int cv_model_benchmark_load_model(uint32_t model_addr, uint32_t arena_size)
{
	if (!runtime_initialized) {
		xprintf("cv_model_benchmark_load_model: runtime_not_initialized\r\n");
		return -1;
	}

	cv_model_benchmark_unload_model();
	if (_arm_npu_init(runtime_security_enable, runtime_privilege_enable) != 0) {
		xprintf("cv_model_benchmark_load_model: npu_init_failed\r\n");
		return -1;
	}

	tensor_arena_addr = mm_reserve_align(arena_size, 0x20);
	tensor_arena_size = arena_size;
	if (tensor_arena_addr == 0) {
		xprintf("cv_model_benchmark_load_model: arena_alloc_failed size=%u\r\n", arena_size);
		return -1;
	}
	xprintf("cv_model_benchmark_load_model: arena_reserved addr=0x%08x size=%u\r\n",
		tensor_arena_addr, tensor_arena_size);

	xprintf("cv_model_benchmark_load_model: get_model addr=0x%08x\r\n", model_addr);
	const tflite::Model *model = tflite::GetModel((const void *)model_addr);
	xprintf("cv_model_benchmark_load_model: model_version=%d expected=%d\r\n",
		model->version(), TFLITE_SCHEMA_VERSION);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		xprintf("cv_model_benchmark_load_model: schema_mismatch model=%d expected=%d\r\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return -1;
	}

	xprintf("cv_model_benchmark_load_model: construct_interpreter\r\n");
	interpreter_ptr = new (interpreter_storage) tflite::MicroInterpreter(
		model, op_resolver, (uint8_t *)tensor_arena_addr, arena_size);
	xprintf("cv_model_benchmark_load_model: allocate_tensors_begin\r\n");
	if (interpreter_ptr->AllocateTensors() != kTfLiteOk) {
		xprintf("cv_model_benchmark_load_model: allocate_tensors_failed\r\n");
		interpreter_ptr->~MicroInterpreter();
		reset_runtime_state();
		return -1;
	}
	xprintf("cv_model_benchmark_load_model: allocate_tensors_ok\r\n");

	input_tensor = interpreter_ptr->input(0);
	current_arena_bytes = (uint32_t)interpreter_ptr->arena_used_bytes();
	return 0;
}

extern "C" int cv_model_benchmark_init(bool security_enable, bool privilege_enable,
                                       uint32_t model_addr, uint32_t arena_size)
{
	if (cv_model_benchmark_runtime_init(security_enable, privilege_enable) != 0) {
		return -1;
	}

	if (model_addr == 0) {
		return 0;
	}

	return cv_model_benchmark_load_model(model_addr, arena_size);
}

extern "C" void cv_model_benchmark_fill_random_input(void)
{
	if (input_tensor == nullptr) return;

	switch (input_tensor->type) {
	case kTfLiteFloat32: {
		float *data = input_tensor->data.f;
		const size_t count = input_tensor->bytes / sizeof(float);
		for (size_t i = 0; i < count; i++) {
			data[i] = (float)(rand() & 0xFF) / 255.0f;
		}
		break;
	}
	case kTfLiteInt8:
		for (size_t i = 0; i < input_tensor->bytes; i++) {
			input_tensor->data.int8[i] = (int8_t)(rand() % 256 - 128);
		}
		break;
	case kTfLiteUInt8:
		for (size_t i = 0; i < input_tensor->bytes; i++) {
			input_tensor->data.uint8[i] = (uint8_t)(rand() & 0xFF);
		}
		break;
	default:
		memset(input_tensor->data.raw, 0, input_tensor->bytes);
		break;
	}
}

extern "C" void cv_model_benchmark_fill_zero_input(void)
{
	if (input_tensor == nullptr) return;

	memset(input_tensor->data.raw, 0, input_tensor->bytes);
}

extern "C" uint32_t cv_model_benchmark_arena_bytes_used(void)
{
	return current_arena_bytes;
}

extern "C" uint32_t cv_model_benchmark_model_bytes(void)
{
	return current_model_bytes;
}

extern "C" uint32_t cv_model_benchmark_run_inference(void)
{
	if (interpreter_ptr == nullptr) return 0;

	uint32_t systick_1, systick_2;
	uint32_t loop_cnt_1, loop_cnt_2;

	SystemGetTick(&systick_1, &loop_cnt_1);

	if (interpreter_ptr->Invoke() != kTfLiteOk) {
		return 0;
	}

	SystemGetTick(&systick_2, &loop_cnt_2);

	uint32_t cycles = (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2);
	return cycles;
}

