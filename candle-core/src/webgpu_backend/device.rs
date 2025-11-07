//! WebGPU Device Implementation

use super::{WebGpuError, WebGpuStorage};
use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Result, Shape, WithDType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::{
    Adapter, Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device, DeviceDescriptor,
    Instance, Queue,
};

/// Unique identifier for WebGPU devices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// WebGPU device for GPU-accelerated tensor operations
#[derive(Clone)]
pub struct WebGpuDevice {
    /// Unique identifier for this device instance
    pub(crate) id: DeviceId,

    /// WebGPU adapter
    pub(crate) adapter: Arc<Adapter>,

    /// WebGPU device handle
    pub(crate) device: Arc<Device>,

    /// Command queue for GPU operations
    pub(crate) queue: Arc<Queue>,

    /// Cache of compiled compute pipelines
    pub(crate) pipelines: Arc<Mutex<HashMap<String, Arc<ComputePipeline>>>>,

    /// RNG seed for random number generation
    pub(crate) seed: Arc<Mutex<u64>>,
}

impl std::fmt::Debug for WebGpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WebGpuDevice({:?})", self.id)
    }
}

impl WebGpuDevice {
    /// Create a new WebGPU device
    pub async fn new_async(ordinal: usize) -> Result<Self> {
        // Create WebGPU instance
        let instance = Instance::default();

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(WebGpuError::AdapterNotFound)?;

        // Request device and queue
        // For WASM, use downlevel_defaults() for maximum browser compatibility
        // downlevel_webgl2_defaults() still includes unsupported limits like maxInterStageShaderComponents
        #[cfg(target_arch = "wasm32")]
        let limits = wgpu::Limits::downlevel_defaults();

        #[cfg(not(target_arch = "wasm32"))]
        let limits = wgpu::Limits::default();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some(&format!("Candle WebGPU Device {}", ordinal)),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| WebGpuError::DeviceRequestFailed(e.to_string()))?;

        Ok(Self {
            id: DeviceId::new(),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines: Arc::new(Mutex::new(HashMap::new())),
            seed: Arc::new(Mutex::new(299792458)),
        })
    }

    /// Create a new WebGPU device (blocking version using pollster)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(ordinal: usize) -> Result<Self> {
        pollster::block_on(Self::new_async(ordinal))
    }

    /// Create a new WebGPU device (blocking version using pollster)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }

    /// Get the WebGPU device handle
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Create a command encoder
    pub fn create_encoder(&self) -> CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Candle Command Encoder"),
            })
    }

    /// Submit commands to the GPU
    pub fn submit(&self, encoder: CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }

    /// Allocate a GPU buffer
    pub fn alloc_buffer(&self, size: u64, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candle Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Get or create a compute pipeline
    pub fn get_or_create_pipeline(
        &self,
        name: &str,
        shader_source: &str,
        entry_point: &str,
    ) -> Result<Arc<ComputePipeline>> {
        let mut pipelines = self.pipelines.lock().unwrap();

        if let Some(pipeline) = pipelines.get(name) {
            return Ok(Arc::clone(pipeline));
        }

        // Create shader module
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create compute pipeline with automatic layout derivation
        let pipeline = Arc::new(self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None, // Automatic layout derivation from shader
                module: &shader_module,
                entry_point,
                compilation_options: Default::default(),
                cache: None,
            }));

        pipelines.insert(name.to_string(), Arc::clone(&pipeline));
        Ok(pipeline)
    }
}

impl BackendDevice for WebGpuDevice {
    type Storage = WebGpuStorage;

    #[cfg(not(target_arch = "wasm32"))]
    fn new(ordinal: usize) -> Result<Self> {
        WebGpuDevice::new(ordinal)
    }

    #[cfg(target_arch = "wasm32")]
    fn new(_ordinal: usize) -> Result<Self> {
        crate::bail!("WebGPU device creation in WASM must use Device::new_webgpu_async(). Synchronous creation is not supported in browsers.")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::WebGpu { gpu_id: self.id.0 }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        WebGpuStorage::zeros(self, shape, dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        WebGpuStorage::alloc_uninit(self, shape, dtype)
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        WebGpuStorage::from_slice(self, data)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        WebGpuStorage::from_cpu_storage(self, storage)
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        WebGpuStorage::from_cpu_storage(self, &storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<Self::Storage> {
        WebGpuStorage::rand_uniform(self, shape, dtype, min, max)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        WebGpuStorage::rand_normal(self, shape, dtype, mean, std)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed.lock().unwrap() = seed;
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // WebGPU operations are already synchronous in the sense that
        // queue submissions happen in order. For explicit synchronization,
        // we'd need to wait for the device to be idle.
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}
