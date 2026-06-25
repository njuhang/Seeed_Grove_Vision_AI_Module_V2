/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/kernels/internal/reference/dequantize.h"

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/dequantize.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

float Float16ToFloat32(TfLiteFloat16 value) {
  const uint32_t half = value.data;
  const uint32_t sign = (half & 0x8000u) << 16;
  uint32_t exponent = (half >> 10) & 0x1fu;
  uint32_t mantissa = half & 0x03ffu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int normalized_exponent = -14;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --normalized_exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | (static_cast<uint32_t>(normalized_exponent + 127) << 23) |
             (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    exponent = exponent + (127 - 15);
    bits = sign | (exponent << 23) | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

}  // namespace

void* DequantizeInit(TfLiteContext* context, const char* buffer,
                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(DequantizeOpData));
}

TfLiteStatus DequantizeEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  DequantizeOpData* data = static_cast<DequantizeOpData*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  // Output type ensured to be kTfLiteFloat32 at the Prepare stage
  TFLITE_DCHECK(output->type == kTfLiteFloat32);

  switch (input->type) {
    case kTfLiteInt8:
      reference_ops::Dequantize(data->quantization_params,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<int8_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt16:
      reference_ops::Dequantize(data->quantization_params,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<int16_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteUInt8:
      reference_ops::Dequantize(data->quantization_params,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<uint8_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteFloat16: {
      const int flat_size =
          MatchingFlatSize(tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorShape(output));
      const TfLiteFloat16* input_data =
          tflite::micro::GetTensorData<TfLiteFloat16>(input);
      float* output_data = tflite::micro::GetTensorData<float>(output);
      for (int i = 0; i < flat_size; ++i) {
        output_data[i] = Float16ToFloat32(input_data[i]);
      }
      break;
    }
    default:
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TFLMRegistration Register_DEQUANTIZE() {
  return tflite::micro::RegisterOp(DequantizeInit, DequantizePrepare,
                                   DequantizeEval);
}

}  // namespace tflite
