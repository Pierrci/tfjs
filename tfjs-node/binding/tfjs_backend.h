/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

#ifndef TF_NODEJS_TFJS_BACKEND_H_
#define TF_NODEJS_TFJS_BACKEND_H_

#include <node_api.h>
#include <uv.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include "ctpl.h"
#include "tf_auto_status.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

namespace tfnodejs
{

class TFJSBackend
{
public:
  // Creates, initializes, and returns a TFJSBackend instance. If initialization
  // fails, a nullptr is returned.
  static TFJSBackend *Create(napi_env env, int num_threads);

  // Creates a new Tensor with given shape and data and returns an ID that
  // refernces the new Tensor.
  // - shape_value (number[])
  // - dtype_value (number)
  // - array_value (TypedArray|Array)
  napi_value CreateTensor(napi_env env, napi_value shape_value,
                          napi_value dtype_value, napi_value array_value);

  // Deletes a created Tensor.
  // - tensor_id_value (number)
  void DeleteTensor(napi_env env, napi_value tensor_id_value);

  // Returns a typed-array as a `napi_value` with the data associated with the
  // TF/TFE pointers.
  // - tensor_id_value (number)
  napi_value GetTensorData(napi_env env, napi_value tensor_id_value);

  // Executes a TFE Op and returns an array of objects containing tensor
  // attributes (id, dtype, shape).
  // - op_name_value (string)
  // - op_attr_inputs (array of TFE Op attributes)
  // - input_tensor_ids (array of input tensor IDs)
  // - num_output_values (number)
  napi_value ExecuteOp(napi_env env, napi_value op_name_value,
                       napi_value op_attr_inputs, napi_value input_tensor_ids,
                       napi_value num_output_values);

  // Load a SavedModel from a path:
  // - export_dir (string)
  // - tags_value (string)
  napi_value LoadSavedModel(napi_env env, napi_value export_dir,
                            napi_value tags_value);

  // Delete the SavedModel corresponding TF_Session and TF_Graph
  // - saved_model_id (number)
  void DeleteSavedModel(napi_env env, napi_value saved_model_id);

  // Execute a session from SavedModel with the provided inputs:
  // - saved_model_id (number)
  // - input_tensor_ids (array of input tensor IDs)
  // - input_op_names (array of input op names)
  // - output_op_names (array of output op names)
  napi_value RunSavedModel(napi_env env, napi_value saved_model_id,
                           napi_value input_tensor_ids,
                           napi_value input_op_names,
                           napi_value output_op_names,
                           napi_ref js_cb);

  // Get number of loaded SavedModel in the backend:
  napi_value GetNumOfSavedModels(napi_env env);

  napi_value GenerateOutputTensorInfo(napi_env env, TFE_TensorHandle *handle);

private:
  TFJSBackend(napi_env env, int num_threads);
  ~TFJSBackend();

  napi_value RunSavedModelInternal(napi_env env, napi_value saved_model_id,
                                   napi_value input_tensor_ids,
                                   napi_value input_op_names,
                                   napi_value output_op_names,
                                   napi_ref js_cb);

  int32_t InsertHandle(TFE_TensorHandle *tfe_handle);
  int32_t InsertSavedModel(TF_Session *tf_session, TF_Graph *tf_graph);

  TFE_Context *tfe_context_;
  std::unordered_map<int32_t, TFE_TensorHandle *> tfe_handle_map_;
  std::unordered_map<int32_t, std::pair<TF_Session *, TF_Graph *>>
      tf_savedmodel_map_;
  std::unordered_map<int32_t, napi_threadsafe_function>
      tf_savedmodel_tsfn_;
  int32_t next_tensor_id_;
  int32_t next_savedmodel_id_;
  std::string device_name;

  ctpl::thread_pool pool;

public:
  bool is_gpu_device;
};

struct ThreadData
{
  // napi_env env;
  TF_Session *session;
  std::vector<TF_Output> inputs;
  std::vector<TF_Tensor *> input_values;
  uint32_t num_input_ids;
  std::vector<TF_Output> outputs;
  std::vector<TF_Tensor *> output_values;
  std::vector<const char *> output_op_name_array;
  TF_AutoStatus tf_status;
  napi_async_work work;
  napi_ref js_cb;
  TFJSBackend &backend;
  std::thread thread;
  napi_threadsafe_function tsfn;
  int savedmodel_id;

  ThreadData(TFJSBackend &bk) : backend{bk} {}
};
// ThreadData::ThreadData(const TFJSBackend &bk) : backend(bk) {};

struct SessionResult
{
  std::vector<TF_Output> inputs;
  std::vector<TF_Tensor *> input_values;
  uint32_t num_input_ids;
  std::vector<TF_Output> outputs;
  std::vector<TF_Tensor *> output_values;
  std::vector<const char *> output_op_name_array;
  TF_AutoStatus tf_status;
};

void RunSession(int id, ThreadData *data);
void ParseSessionResult(napi_env env, napi_value js_callback, void *context, void *data);
} // namespace tfnodejs

#endif // TF_NODEJS_TFJS_BACKEND_H_
