#!/usr/bin/env bash
# Method 1 for facenet repo based backbones
mode="$1"
model_name="$2"
output_inference_graph_path="$3"
num_classes="$4"
checkpoint_path="$5"
frozen_output_model_path="$6"
output_node_names="$7"

display_usage() {
  echo
  echo "Usage: $0"
  echo "[mode] [model_name] [output_inference_graph_path] [num_classes] [checkpoint_path] [output_model_path] [output_node_names]"
  echo " -h, --help   Display usage instructions"
  echo " -p, --print  Print welcome message"
  echo
}


case $mode in
    -h|--help)
      display_usage
      ;;
    -p|--print)
      echo "print_message"
      ;;
    -f|--facenet)
      python freeze_tf_model.py $checkpoint_path $frozen_output_model_path --output_node_names $output_node_names
      ;;
    -m|--models)
      # Method 2 for offical models repo based backbones
      python export_inference_graph.py  --alsologtostderr --model_name=inception_resnet_v1 --output_file=$output_inference_graph_path --num_classes=$num_classes
      freeze_graph  --input_graph=$output_inference_graph_path --input_checkpoint=$checkpoint_path --input_binary=true --output_graph=$frozen_output_model_path  --output_node_names=$output_node_names
      ;;
    *)
      raise_error "Unknown argument: ${argument}"
      echo "display_usage"
      ;;
esac
