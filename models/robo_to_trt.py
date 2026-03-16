import torch
import tensorrt as trt
import numpy as np
import time
import os

# Assuming your model class and dependencies are in the same directory or PYTHONPATH
from robofirefusenet import RoboFireFuseNet

def build_engine(model_path, engine_path, input_res=(256, 256)):
    """
    Builds a TensorRT engine from an ONNX model with dynamic batch size.
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Increase workspace for transformer blocks
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    
    # Load ONNX model
    with open(model_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Define optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    # Set (Min, Opt, Max) shapes
    profile.set_shape(input_name, (1, 4, *input_res), (5, 4, *input_res), (10, 4, *input_res))
    config.add_optimization_profile(profile)

    # Use FP16 if supported for maximum speed
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 Precision")

    print(f"Building TensorRT engine: {engine_path}...")
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print("Engine built successfully.")

def benchmark_trt(engine_path, batch_size=1, input_res=(256, 256)):
    """
    Benchmarks the inference speed of the TensorRT engine.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Set the input shape for this specific inference run
    context.set_input_shape(engine.get_tensor_name(0), (batch_size, 4, *input_res))
    
    # Prepare buffers
    device = torch.device('cuda:0')
    dummy_input = torch.randn(batch_size, 4, *input_res, device=device)
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))
    # Adjust output shape for dynamic batch
    output_shape = (batch_size, *output_shape[1:])
    output = torch.empty(output_shape, device=device)

    # Warmup
    print(f"Warming up batch_size={batch_size}...")
    for _ in range(10):
        context.execute_v2([dummy_input.data_ptr(), output.data_ptr()])

    # Measure speed
    print(f"Benchmarking batch_size={batch_size}...")
    torch.cuda.synchronize()
    start_time = time.time()
    iters = 100
    for _ in range(iters):
        context.execute_v2([dummy_input.data_ptr(), output.data_ptr()])
    torch.cuda.synchronize()
    end_time = time.time()

    avg_latency = (end_time - start_time) / iters
    fps = batch_size / avg_latency
    print(f"Avg Latency: {avg_latency*1000:.2f} ms | FPS: {fps:.2f}")

if __name__ == '__main__':
    device = 'cuda:0'
    input_res = (480, 640)
    num_classes = 3
    onnx_file = "robofirefusenet_dynamic.onnx"
    engine_file = "robofirefusenet_dynamic.trt"

    # 1. Initialize and Export Model
    # Set augment=False for deployment to output a single segmentation map
    model = RoboFireFuseNet(m=2, n=3, num_classes=num_classes, planes=32, 
                            ppm_planes=96, head_planes=128, augment=False, 
                            channels=4, input_resolution=input_res, 
                            window_size=(8, 8), tf_depths=[2, 6]).to(device)
    model.eval()

    dummy_input = torch.randn(1, 4, *input_res).to(device)
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, onnx_file,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14 # High opset recommended for Transformers
    )

    # 2. Build TRT Engine
    if not os.path.exists(engine_file):
        build_engine(onnx_file, engine_file, input_res=input_res)

    # 3. Benchmark across requested batch sizes
    for b in [1, 5, 10]:
        benchmark_trt(engine_file, batch_size=b, input_res=input_res)