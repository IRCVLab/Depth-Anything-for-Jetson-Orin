import argparse

import time

import os
from pathlib import Path

import torch
import tensorrt as trt
from depth_anything import DepthAnything


def export(
    weights_path: str,  
    save_path: str,
    input_size: int,
    onnx: bool = True,
):  
    """
    weights_path: str -> Path to the PyTorch model(local / hub)
    save_path: str -> Directory to save the model
    input_size: int -> Width and height of the input image(e.g. 308, 364, 406, 518)
    onnx: bool -> Export the model to ONNX format
    """
    weights_path = Path(weights_path)
    
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    model = DepthAnything.from_pretrained(weights_path).to('cpu').eval()
    
    # Create a dummy input
    dummy_input = torch.ones((3, input_size, input_size)).unsqueeze(0)
    _ = model(dummy_input)
    onnx_path = Path(save_path) / f"{weights_path.stem}_{input_size}.onnx"
    
    # Export the PyTorch model to ONNX format
    if onnx:
        torch.onnx.export(
            model,
            dummy_input, 
            onnx_path, 
            opset_version=11    , 
            input_names=["input"], 
            output_names=["output"], 
        )
        print(f"Model exported to {onnx_path}", onnx_path)
        print("Saving the model to ONNX format...")
        time.sleep(2)
    
    # ONNX to TensorRT
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('Failed to parse the ONNX model.')
    
    # Set up the builder config
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16) # FP16
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2 GB
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(onnx_path.with_suffix(".trt"), "wb") as f:
        f.write(serialized_engine)
    
if __name__ == '__main__':
    # args = argparse.ArgumentParser()
    # args.add_argument("--weights_path", type=str, default="LiheYoung/depth_anything_vits14")
    # args.add_argument("--save_path", type=str, default="weights")
    # args.add_argument("--input_size", type=int, default=406)
    
    # export(
    #     weights_path=args.weights_path,
    #     save_path=args.save_path,
    #     input_size=args.input_size,
    #     onnx=True,
    # )
    
    export(
        weights_path="LiheYoung/depth_anything_vits14", # local hub or online
        save_path="weights", # folder name
        input_size=308, # 308 | 364 | 406 | 518
        onnx=True,
    )