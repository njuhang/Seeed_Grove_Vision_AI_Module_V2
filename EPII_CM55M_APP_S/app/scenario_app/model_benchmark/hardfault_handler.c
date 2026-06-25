/*
 * hardfault_handler.c
 *
 * Exception handlers for debugging
 */

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "WE2_device.h"

void HardFault_Handler(void) {
	printf("\r\nEntering HardFault interrupt!\r\n");
	if (SAU->SFSR != 0) {
		if (SAU->SFSR & SAU_SFSR_INVEP_Msk) {
			printf("SAU->SFSR:INVEP fault: Invalid entry point to secure world.\r\n");
		} else if (SAU->SFSR & SAU_SFSR_AUVIOL_Msk) {
			printf("SAU->SFSR:AUVIOL fault: SAU violation. Access to secure memory from normal world.\r\n");
		} else if (SAU->SFSR & SAU_SFSR_INVTRAN_Msk) {
			printf("SAU->SFSR:INVTRAN fault: Invalid transition from secure to normal world.\r\n");
		} else {
			printf("Another SAU error.\r\n");
		}
		if (SAU->SFSR & SAU_SFSR_SFARVALID_Msk) {
			printf("Address that caused SAU violation is 0x%X.\r\n", SAU->SFAR);
		}
	}

	if (SCB->CFSR != 0) {
		if (SCB->CFSR & SCB_CFSR_IBUSERR_Msk) {
			printf("SCB->BFSR:IBUSERR fault: Instruction bus error on an instruction prefetch.\r\n");
		} else if (SCB->CFSR & SCB_CFSR_PRECISERR_Msk) {
			printf("SCB->BFSR:PRECISERR fault: Precise data access error.\r\n");
		} else {
			printf("Security Another secure bus error 1.\r\n");
		}
		if (SCB->CFSR & SCB_CFSR_BFARVALID_Msk) {
			printf("Address that caused secure bus violation is 0x%X.\r\n", SCB->BFAR);
		}
	}

	if (SCB_NS->CFSR != 0) {
		if (SCB_NS->CFSR & SCB_CFSR_IBUSERR_Msk) {
			printf("SCB_NS->BFSR:IBUSERR fault: Instruction bus error on an instruction prefetch.\r\n");
		} else if (SCB_NS->CFSR & SCB_CFSR_PRECISERR_Msk) {
			printf("SCB_NS->BFSR:PRECISERR fault: Precise data access error.\r\n");
		} else {
			printf("Security Another secure bus error 2.\r\n");
		}
		if (SCB_NS->CFSR & SCB_CFSR_BFARVALID_Msk) {
			printf("Address that caused secure bus violation is 0x%X.\r\n", SCB_NS->BFAR);
		}
	}

	printf("SCB->CFSR:0x%08x\n", SCB->CFSR);
	printf("SCB->BFAR:0x%08x\n", SCB->BFAR);
	printf("SCB->HFSR:0x%08x\n", SCB->HFSR);
	for (;;) {
	}
}

void NMI_Handler(void) {
	printf("\r\nEntering NMI_Handler interrupt!\r\n");
	for (;;) {
	}
}

void MemManage_Handler(void) {
	printf("\r\nEntering MemManage_Handler interrupt!\r\n");
	for (;;) {
	}
}

void BusFault_Handler(void) {
	printf("\r\nEntering BusFault_Handler interrupt!\r\n");
	printf("SCB->CFSR:0x%08x\n", SCB->CFSR);
	printf("SCB->BFAR:0x%08x\n", SCB->BFAR);
	printf("SCB->HFSR:0x%08x\n", SCB->HFSR);
	for (;;) {
	}
}

void UsageFault_Handler(void) {
	printf("\r\nEntering UsageFault_Handler interrupt!\r\n");
	for (;;) {
	}
}

void SecureFault_Handler(void) {
	printf("\r\nEntering SecureFault_Handler interrupt!\r\n");
	for (;;) {
	}
}
