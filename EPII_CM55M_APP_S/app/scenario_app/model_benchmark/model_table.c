/*
 * model_table.c
 *
 * Helpers for reading the benchmark model table from XIP flash.
 */

#include "model_table.h"

#include <string.h>

bool model_table_load(uint32_t flash_addr, model_table_t *out_table)
{
    const model_table_t *flash_table = (const model_table_t *)flash_addr;

    if (out_table == 0) {
        return false;
    }

    if (flash_table->magic != MODEL_TABLE_MAGIC) {
        return false;
    }

    if (flash_table->version != MODEL_TABLE_VERSION) {
        return false;
    }

    if (flash_table->count > MAX_MODEL_COUNT) {
        return false;
    }

    memcpy(out_table, flash_table, sizeof(model_table_t));
    return true;
}
