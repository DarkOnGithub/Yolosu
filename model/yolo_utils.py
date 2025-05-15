import tensorrt as trt
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ONNX2TensorRT:
    def __init__(
        self,
        onnx_path: str,
        engine_path: str,
        fp16_mode: bool = True,
        max_workspace_size: int = 1 << 30,  
        max_batch_size: int = 1,
        verbose: bool = False
    ):
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.fp16_mode = fp16_mode
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        
    def build_engine(self) -> Optional[trt.ICudaEngine]:
        """
        Build TensorRT engine from ONNX model
        """
        try:
            if not os.path.exists(self.onnx_path):
                raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")
            
            logger.info(f"Building TensorRT engine from {self.onnx_path}")
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            
            parser = trt.OnnxParser(network, self.logger)
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(error)}")
                    return None
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
            if self.fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Using FP16 mode")
            logger.info("Building TensorRT engine...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return None
            with open(self.engine_path, 'wb') as f:
                f.write(serialized_engine)
            logger.info(f"TensorRT engine saved to {self.engine_path}")
            
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            return engine
            
        except Exception as e:
            logger.error(f"Error building TensorRT engine: {str(e)}")
            return None

def convert_onnx_to_engine(
    onnx_path: str = r"runs\detect\train6\weights\best.onnx",
    engine_path: str = r"runs\detect\train6\weights\best.engine",
    fp16_mode: bool = True,
    verbose: bool = False
) -> bool:
    """
    Convert ONNX model to TensorRT engine
    """
    converter = ONNX2TensorRT(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16_mode=fp16_mode,
        verbose=verbose
    )
    
    engine = converter.build_engine()
    return engine is not None

if __name__ == "__main__":
    success = convert_onnx_to_engine(verbose=True)
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")