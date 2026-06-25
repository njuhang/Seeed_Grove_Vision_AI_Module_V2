/*
 * model_table.h
 *
 * Flash model table contract shared by PC-side tooling and firmware.
 */

#ifndef SCENARIO_APP_MODEL_BENCHMARK_MODEL_TABLE_H_
#define SCENARIO_APP_MODEL_BENCHMARK_MODEL_TABLE_H_

#include <stdbool.h>
#include <stdint.h>

#include "common_config.h"

#define MODEL_TABLE_MAGIC   0x4C444F4D
#define MODEL_TABLE_VERSION 1
#define MODEL_NAME_MAX_LEN  32
#define MODEL_TASK_MAX_LEN  32

typedef struct {
    char name[MODEL_NAME_MAX_LEN];
    char task[MODEL_TASK_MAX_LEN];
    uint32_t flash_addr;
    uint32_t model_size_bytes;
    uint32_t input_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t tier;
} model_table_entry_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t reserved;
    model_table_entry_t entries[MAX_MODEL_COUNT];
} model_table_t;

bool model_table_load(uint32_t flash_addr, model_table_t *out_table);

#endif /* SCENARIO_APP_MODEL_BENCHMARK_MODEL_TABLE_H_ */
