#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_slice.h"

using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;
using tensorflow::Scope;
using tensorflow::ClientSession;

class ModelLoader{
	private:
		SavedModelBundle bundle;
		SessionOptions session_options;
		RunOptions run_options;
		void make_prediction(std::vector<Tensor> &image_output, Tensor &pred);
	public:
		ModelLoader(string);
		void predict(string filename, Tensor &out_pred);
};


Status ReadImageFile(const string &filename, std::vector<Tensor>* out_tensors){
	using namespace ::tensorflow::ops;
	Scope root = Scope::NewRootScope();
	auto output = tensorflow::ops::ReadFile(root.WithOpName("file_reader"), filename);

	tensorflow::Output image_reader;
	const int wanted_channels = 1;
	image_reader = tensorflow::ops::DecodePng(root.WithOpName("file_decoder"), output, DecodePng::Channels(wanted_channels));

	auto image_float = Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
	auto image_expanded = ExpandDims(root.WithOpName("expand_dims"), image_float, 0);
	auto image_normalized = Div(root.WithOpName("float_div"), image_expanded, 255.f);

	tensorflow::GraphDef graph;
	auto s = (root.ToGraphDef(&graph));

	if (!s.ok()){
		printf("Error in loading image from file\n");
	}
	else{
		printf("Input image loaded correctly!\n");
	}

	ClientSession session(root);

	auto run_status = session.Run({image_normalized}, out_tensors);
	if (!run_status.ok()){
		printf("Error in running session \n");
	}
	return Status::OK();

}

ModelLoader::ModelLoader(string path){		
	session_options.config.mutable_gpu_options()->set_allow_growth(true);	
	auto status = tensorflow::LoadSavedModel(session_options, run_options, path, {"serve"},
			&bundle);
    //https://github.com/tensorflow/tensorflow/issues/35614
    //Status status;
    //SavedModelBundle bundle
    //    status = LoadSavedModel(SessionOptions(), RunOptions(), pathToGraph, {tensorflow::kSavedModelTagServe}, &bundle);
    ////...define inputs and outputs
    //Status run_status = bundle.session->Run({{"serving_default_input_1:0", input_tensor_state}, {"serving_default_input_2:0", input_tensor_action}}, {outputlayer}, {}, &outputs);

	if (status.ok()){
		printf("Model loaded successfully...\n");
	}
	else {
		printf("Error in loading model\n");
	}

}

void ModelLoader::predict(string filename, Tensor &out_pred){
	std::vector<Tensor> image_output;
	auto read_status = ReadImageFile(filename, &image_output);
	make_prediction(image_output, out_pred);
}

void ModelLoader::make_prediction(std::vector<Tensor> &image_output, Tensor &out_pred){
    const string input_node = "serving_default_input_1:0";
	std::vector<std::pair<string, Tensor>> inputs_data  = {{input_node, image_output[0]}};
	std::vector<string> output_nodes = {{"StatefulPartitionedCall:0"}}; //conv2d_45

	
	std::vector<Tensor> predictions;
	auto status = this->bundle.GetSession()->Run(inputs_data, output_nodes, {}, &predictions);
	if (status.ok()){
		printf("Run reference successfully...\n");
	}
	else {
        std::cout << status.ToString() << std::endl;
	}

	auto predicted_mask = predictions[0];
    out_pred = predicted_mask;
}
