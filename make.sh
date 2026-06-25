#!/usr/bin/env bash
# make.sh
#
# Incremental build + image-gen for the Seeed Grove Vision AI Module V2.
#
# What it does (in order):
#   1. Wipe stale image-gen inputs (old ELF/map) and the whole output dir,
#      so a half-finished previous run can never produce a misleading image.
#   2. Force-rebuild just the application sources (model_benchmark.c,
#      cvapp_model_benchmark.cpp, main.c) by deleting their .o files. This
#      avoids `make clean`, which would also nuke the TFLM library and cost
#      ~10+ minutes to rebuild.
#   3. Run `make` to relink with the fresh app objects.
#   4. Copy the new ELF into the image-gen input dir and run
#      `we2_local_image_gen` to produce output.img.
#
# Optional knobs:
#   FORCE_INFERENCE_REBUILD=1
#     Also invalidate the TFLM inference library objects/archive. Use this
#     when editing sources under library/inference/, otherwise make.sh only
#     recompiles app-layer sources.
#
# Exit on any error so we don't flash a stale image.
set -euo pipefail

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
APP_DIR="${REPO_ROOT}/EPII_CM55M_APP_S"
IMG_DIR="${REPO_ROOT}/we2_image_gen_local"
OBJ_DIR="${APP_DIR}/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65"

ELF_NAME="EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"
MAP_NAME="EPII_CM55M_gnu_epii_evb_WLCSP65_s.map"
ELF_PATH="${OBJ_DIR}/${ELF_NAME}"
IMG_INPUT_DIR="${IMG_DIR}/input_case1_secboot"
IMG_OUTPUT_DIR="${IMG_DIR}/output_case1_sec_wlcsp"
IMG_PROJECT="project_case1_blp_wlcsp.json"
FINAL_IMG="${IMG_OUTPUT_DIR}/output.img"
FORCE_INFERENCE_REBUILD="${FORCE_INFERENCE_REBUILD:-0}"
ARM_GNU_TOOLCHAIN_BIN="${ARM_GNU_TOOLCHAIN_BIN:-${HOME}/tool/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/bin}"

if [[ -d "${ARM_GNU_TOOLCHAIN_BIN}" ]]; then
	export PATH="${ARM_GNU_TOOLCHAIN_BIN}:${PATH}"
fi

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
	echo "ERROR: arm-none-eabi-gcc not found. Set ARM_GNU_TOOLCHAIN_BIN to the toolchain bin directory." >&2
	exit 1
fi

# Which APP_TYPE is configured (used only to know which .o files to nuke).
APP_TYPE="$(awk -F= '/^APP_TYPE[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "${APP_DIR}/makefile")"
echo "==> APP_TYPE = ${APP_TYPE}"

# ---- 1. wipe stale image-gen artefacts ----
echo "==> Cleaning previous image-gen outputs"
rm -f "${IMG_INPUT_DIR}/${ELF_NAME}" "${IMG_INPUT_DIR}/${MAP_NAME}"
rm -rf "${IMG_OUTPUT_DIR}"

# ---- 2. force-rebuild only the app sources ----
# Removing these .o files (and the final ELF) makes `make` recompile just the
# scenario_app + main + relink, while every prebuilt library .a stays cached.
echo "==> Invalidating app object files (incremental rebuild, no make clean)"
APP_SCENARIO_OBJ_DIR="${OBJ_DIR}/app/scenario_app/${APP_TYPE}"
if [[ -d "${APP_SCENARIO_OBJ_DIR}" ]]; then
	rm -f "${APP_SCENARIO_OBJ_DIR}"/*.o
fi
rm -f "${OBJ_DIR}/app/main.o"
rm -f "${ELF_PATH}"

if [[ "${FORCE_INFERENCE_REBUILD}" == "1" ]]; then
	echo "==> FORCE_INFERENCE_REBUILD=1, invalidating TFLM inference library"
	rm -rf "${OBJ_DIR}/library/inference/tflmtag2412_u55tag2411"
	rm -f "${OBJ_DIR}/libtflmtag2412_u55tag2411_cmsisnn_gnu.a"
fi

# ---- 3. build ----
echo "==> Building firmware"
make -C "${APP_DIR}" -j"$(nproc)"

if [[ ! -f "${ELF_PATH}" ]]; then
	echo "ERROR: expected ELF not produced: ${ELF_PATH}" >&2
	exit 1
fi

# ---- 4. generate flashable image ----
echo "==> Copying ELF into image-gen input dir"
cp "${ELF_PATH}" "${IMG_INPUT_DIR}/"
# .map is optional for image generation but handy to keep alongside.
[[ -f "${OBJ_DIR}/${MAP_NAME}" ]] && cp "${OBJ_DIR}/${MAP_NAME}" "${IMG_INPUT_DIR}/" || true

echo "==> Running we2_local_image_gen"
cd "${IMG_DIR}"
./we2_local_image_gen "${IMG_PROJECT}"

if [[ ! -f "${FINAL_IMG}" ]]; then
	echo "ERROR: image generation did not produce ${FINAL_IMG}" >&2
	exit 1
fi

echo
echo "==> Done."
echo "    ELF  : ${ELF_PATH}"
echo "    Image: ${FINAL_IMG}"
