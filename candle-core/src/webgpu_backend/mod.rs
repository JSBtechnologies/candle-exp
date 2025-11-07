//! WebGPU Backend for Candle
//!
//! This module provides WebGPU support for running Candle models in web browsers
//! and on native platforms with WebGPU support.

mod device;
mod storage;
mod shaders;

pub use device::WebGpuDevice;
pub use storage::WebGpuStorage;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WebGpuError {
    #[error("WebGPU adapter not found")]
    AdapterNotFound,

    #[error("WebGPU device request failed: {0}")]
    DeviceRequestFailed(String),

    #[error("Buffer mapping failed")]
    BufferMappingFailed,

    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    #[error("Invalid buffer size: {0}")]
    InvalidBufferSize(usize),

    #[error("WebGPU operation failed: {0}")]
    OperationFailed(String),

    #[error("Data type not supported: {0:?}")]
    UnsupportedDType(crate::DType),
}

impl From<WebGpuError> for crate::Error {
    fn from(e: WebGpuError) -> Self {
        crate::Error::Msg(format!("WebGPU error: {}", e))
    }
}
