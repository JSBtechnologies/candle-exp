//! WebGPU Storage Implementation

use super::{shaders, WebGpuDevice, WebGpuError};
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, WithDType};
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages};

/// Storage for tensors on WebGPU devices
#[derive(Debug)]
pub struct WebGpuStorage {
    /// GPU buffer containing the tensor data
    pub(crate) buffer: Arc<Buffer>,

    /// Reference to the WebGPU device
    pub(crate) device: WebGpuDevice,

    /// Data type of the tensor
    pub(crate) dtype: DType,

    /// Number of elements in the tensor
    pub(crate) elem_count: usize,
}

impl WebGpuStorage {
    /// Create a new WebGPU storage with zeros
    pub fn zeros(device: &WebGpuDevice, shape: &Shape, dtype: DType) -> Result<Self> {
        let elem_count = shape.elem_count();
        let size_in_bytes = elem_count * dtype.size_in_bytes();

        // Allocate buffer filled with zeros
        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candle Zero Buffer"),
            size: size_in_bytes as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create zero-filled staging buffer and copy
        let zeros = vec![0u8; size_in_bytes];
        device.queue.write_buffer(&buffer, 0, &zeros);

        Ok(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            dtype,
            elem_count,
        })
    }

    /// Allocate uninitialized storage
    pub unsafe fn alloc_uninit(device: &WebGpuDevice, shape: &Shape, dtype: DType) -> Result<Self> {
        let elem_count = shape.elem_count();
        let size_in_bytes = elem_count * dtype.size_in_bytes();

        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candle Uninit Buffer"),
            size: size_in_bytes as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            dtype,
            elem_count,
        })
    }

    /// Create storage from a slice
    pub fn from_slice<T: WithDType>(device: &WebGpuDevice, data: &[T]) -> Result<Self> {
        let dtype = T::DTYPE;
        let elem_count = data.len();
        let size_in_bytes = elem_count * dtype.size_in_bytes();

        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candle Data Buffer"),
            size: size_in_bytes as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Convert data to bytes
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, size_in_bytes)
        };

        device.queue.write_buffer(&buffer, 0, data_bytes);

        Ok(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            dtype,
            elem_count,
        })
    }

    /// Create storage from CPU storage
    pub fn from_cpu_storage(device: &WebGpuDevice, storage: &CpuStorage) -> Result<Self> {
        let dtype = storage.dtype();

        // Match on storage type and convert to bytes properly
        let (elem_count, buffer) = match storage {
            CpuStorage::U8(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (U8)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
            CpuStorage::U32(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (U32)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
            CpuStorage::I64(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (I64)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
            CpuStorage::BF16(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (BF16)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
            CpuStorage::F16(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (F16)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
            CpuStorage::F32(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (F32)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
            CpuStorage::F64(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (F64)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
            CpuStorage::F8E4M3(data) => {
                let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Candle CPU->GPU Buffer (F8E4M3)"),
                    size: (data.len() * dtype.size_in_bytes()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * dtype.size_in_bytes())
                };
                device.queue.write_buffer(&buffer, 0, bytes);
                (data.len(), buffer)
            }
        };

        Ok(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            dtype,
            elem_count,
        })
    }

    /// Generate random uniform values
    pub fn rand_uniform(
        device: &WebGpuDevice,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self> {
        // For now, generate on CPU and upload
        // TODO: Implement GPU-based RNG
        use rand::Rng;
        let mut rng = rand::rng();
        let elem_count = shape.elem_count();

        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| rng.random_range(min as f32..max as f32))
                    .collect();
                Self::from_slice(device, &data)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| rng.random_range(min..max))
                    .collect();
                Self::from_slice(device, &data)
            }
            _ => Err(WebGpuError::UnsupportedDType(dtype).into()),
        }
    }

    /// Generate random normal values
    pub fn rand_normal(
        device: &WebGpuDevice,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self> {
        // For now, generate on CPU and upload
        // TODO: Implement GPU-based RNG
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(mean, std).map_err(|e| crate::Error::Msg(e.to_string()))?;
        let mut rng = rand::rng();
        let elem_count = shape.elem_count();

        match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| normal.sample(&mut rng) as f32)
                    .collect();
                Self::from_slice(device, &data)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| normal.sample(&mut rng))
                    .collect();
                Self::from_slice(device, &data)
            }
            _ => Err(WebGpuError::UnsupportedDType(dtype).into()),
        }
    }

    /// Get the size in bytes
    pub fn size_in_bytes(&self) -> usize {
        self.elem_count * self.dtype.size_in_bytes()
    }

    /// Execute a binary operation between two tensors
    fn binary_op(
        &self,
        rhs: &Self,
        op_name: &'static str,
        shader_source: &'static str,
    ) -> Result<Self> {
        // Validate dtypes and get element count
        if self.dtype != rhs.dtype {
            crate::bail!("WebGPU binary op: mismatched dtypes {:?} and {:?}", self.dtype, rhs.dtype);
        }
        if self.elem_count != rhs.elem_count {
            crate::bail!("WebGPU binary op: mismatched element counts {} and {}", self.elem_count, rhs.elem_count);
        }
        if self.dtype != DType::F32 {
            crate::bail!("WebGPU binary op currently only supports F32, got {:?}", self.dtype);
        }

        // Create output buffer
        let output_size = (self.elem_count * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Output Buffer", op_name)),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Get or create pipeline
        let pipeline = self.device.get_or_create_pipeline(op_name, shader_source, op_name)?;

        // Get bind group layout from the pipeline (automatic layout derivation)
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", op_name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.create_encoder();
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Compute Pass", op_name)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 256 threads per workgroup
            let workgroup_size = 256;
            let num_workgroups = (self.elem_count as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.device.submit(encoder);

        Ok(Self {
            buffer: Arc::new(output_buffer),
            device: self.device.clone(),
            dtype: self.dtype,
            elem_count: self.elem_count,
        })
    }

    /// Execute a unary operation on a tensor
    fn unary_op(&self, op_name: &'static str, shader_source: &'static str, entry_point: &'static str) -> Result<Self> {
        // Validate dtype
        if self.dtype != DType::F32 {
            crate::bail!("WebGPU unary op currently only supports F32, got {:?}", self.dtype);
        }

        // Create output buffer
        let output_size = (self.elem_count * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Output Buffer", op_name)),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Get or create pipeline
        let pipeline = self.device.get_or_create_pipeline(op_name, shader_source, entry_point)?;

        // Get bind group layout from the pipeline (automatic layout derivation)
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", op_name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.create_encoder();
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Compute Pass", op_name)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with 256 threads per workgroup
            let workgroup_size = 256;
            let num_workgroups = (self.elem_count as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.device.submit(encoder);

        Ok(Self {
            buffer: Arc::new(output_buffer),
            device: self.device.clone(),
            dtype: self.dtype,
            elem_count: self.elem_count,
        })
    }

    /// Copy data from GPU to CPU as a Vec
    pub(crate) fn to_cpu<T: Clone>(&self) -> Result<Vec<T>> {
        let size_in_bytes = self.elem_count * self.dtype.size_in_bytes();

        // Create staging buffer for readback
        let staging_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candle Staging Buffer"),
            size: size_in_bytes as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy GPU buffer to staging buffer
        let mut encoder = self.device.create_encoder();
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size_in_bytes as u64);
        self.device.submit(encoder);

        // Wait for GPU to finish
        self.device.device.poll(wgpu::Maintain::Wait);

        // Map and read the buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .unwrap()
            .map_err(|_| WebGpuError::BufferMappingFailed)?;

        let data = buffer_slice.get_mapped_range();

        // Convert bytes to Vec<T>
        let result = unsafe {
            let ptr = data.as_ptr() as *const T;
            let len = self.elem_count;
            let mut vec = Vec::with_capacity(len);
            std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
            vec.set_len(len);
            vec
        };

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

impl BackendStorage for WebGpuStorage {
    type Device = WebGpuDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        // Create a new buffer and copy data
        let buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candle Cloned Buffer"),
            size: self.buffer.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_encoder();
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, self.buffer.size());
        self.device.submit(encoder);

        Ok(Self {
            buffer: Arc::new(buffer),
            device: self.device.clone(),
            dtype: self.dtype,
            elem_count: self.elem_count,
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
            DType::F8E4M3 => Ok(CpuStorage::F8E4M3(self.to_cpu()?)),
        }
    }

    // Placeholder implementations for operations
    // These will be implemented with actual GPU kernels

    fn affine(&self, _layout: &Layout, _mul: f64, _add: f64) -> Result<Self> {
        crate::bail!("WebGPU affine not yet implemented")
    }

    fn powf(&self, _layout: &Layout, _exp: f64) -> Result<Self> {
        crate::bail!("WebGPU powf not yet implemented")
    }

    fn elu(&self, _layout: &Layout, _alpha: f64) -> Result<Self> {
        crate::bail!("WebGPU elu not yet implemented")
    }

    fn reduce_op(&self, _op: ReduceOp, _layout: &Layout, _dims: &[usize]) -> Result<Self> {
        crate::bail!("WebGPU reduce_op not yet implemented")
    }

    fn cmp(&self, _op: CmpOp, _rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        crate::bail!("WebGPU cmp not yet implemented")
    }

    fn to_dtype(&self, _layout: &Layout, _dtype: DType) -> Result<Self> {
        crate::bail!("WebGPU to_dtype not yet implemented")
    }

    fn unary_impl<B: UnaryOpT>(&self, _layout: &Layout) -> Result<Self> {
        // Match on the operation type to select the appropriate shader
        match B::KERNEL {
            "urelu" => self.unary_op("relu", shaders::RELU_SHADER, "relu"),
            "ugelu" => self.unary_op("gelu", shaders::GELU_SHADER, "gelu"),
            "utanh" => self.unary_op("tanh", shaders::TANH_SHADER, "tanh_kernel"),
            "uexp" => self.unary_op("exp", shaders::EXP_SHADER, "exp_kernel"),
            "ulog" => self.unary_op("log", shaders::LOG_SHADER, "log_kernel"),
            "usilu" => {
                // Silu = x * sigmoid(x) = x / (1 + exp(-x))
                // For now, we can compute it directly or use sigmoid
                // Let's use sigmoid shader as a base and compute silu
                crate::bail!("WebGPU silu operation not yet implemented (use sigmoid and mul instead)")
            },
            kernel => crate::bail!("WebGPU unary operation '{}' not yet implemented", kernel),
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        _lhs_l: &Layout,
        _rhs_l: &Layout,
    ) -> Result<Self> {
        // Match on the operation type to select the appropriate shader
        match B::KERNEL {
            "badd" => self.binary_op(rhs, "add", shaders::ADD_SHADER),
            "bmul" => self.binary_op(rhs, "mul", shaders::MUL_SHADER),
            "bsub" => self.binary_op(rhs, "sub", shaders::SUB_SHADER),
            "bdiv" => self.binary_op(rhs, "div", shaders::DIV_SHADER),
            kernel => crate::bail!("WebGPU binary operation '{}' not yet implemented", kernel),
        }
    }

    fn where_cond(
        &self,
        _layout: &Layout,
        _t: &Self,
        _t_l: &Layout,
        _f: &Self,
        _f_l: &Layout,
    ) -> Result<Self> {
        crate::bail!("WebGPU where_cond not yet implemented")
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        crate::bail!("WebGPU conv1d not yet implemented")
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        crate::bail!("WebGPU conv_transpose1d not yet implemented")
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        crate::bail!("WebGPU conv2d not yet implemented")
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        crate::bail!("WebGPU conv_transpose2d not yet implemented")
    }

    fn avg_pool2d(&self, _l: &Layout, _k: (usize, usize), _stride: (usize, usize)) -> Result<Self> {
        crate::bail!("WebGPU avg_pool2d not yet implemented")
    }

    fn max_pool2d(&self, _l: &Layout, _k: (usize, usize), _stride: (usize, usize)) -> Result<Self> {
        crate::bail!("WebGPU max_pool2d not yet implemented")
    }

    fn upsample_nearest1d(&self, _l: &Layout, _sz: usize) -> Result<Self> {
        crate::bail!("WebGPU upsample_nearest1d not yet implemented")
    }

    fn upsample_nearest2d(&self, _l: &Layout, _h: usize, _w: usize) -> Result<Self> {
        crate::bail!("WebGPU upsample_nearest2d not yet implemented")
    }

    fn gather(&self, _l: &Layout, _ids: &Self, _ids_l: &Layout, _dim: usize) -> Result<Self> {
        crate::bail!("WebGPU gather not yet implemented")
    }

    fn scatter_set(
        &mut self,
        _l: &Layout,
        _ids: &Self,
        _ids_l: &Layout,
        _src: &Self,
        _src_l: &Layout,
        _dim: usize,
    ) -> Result<()> {
        crate::bail!("WebGPU scatter_set not yet implemented")
    }

    fn scatter_add_set(
        &mut self,
        _l: &Layout,
        _ids: &Self,
        _ids_l: &Layout,
        _src: &Self,
        _src_l: &Layout,
        _dim: usize,
    ) -> Result<()> {
        crate::bail!("WebGPU scatter_add_set not yet implemented")
    }

    fn index_select(&self, _ids: &Self, _l: &Layout, _ids_l: &Layout, _dim: usize) -> Result<Self> {
        crate::bail!("WebGPU index_select not yet implemented")
    }

    fn index_add(
        &self,
        _l: &Layout,
        _ids: &Self,
        _ids_l: &Layout,
        _src: &Self,
        _src_l: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        crate::bail!("WebGPU index_add not yet implemented")
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // For now, only support simple contiguous 2D matmul
        // TODO: Handle batched matmul and non-contiguous layouts
        let (_b, m, n, k) = bmnk;

        // Validate dtypes match and are F32
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            crate::bail!("WebGPU matmul currently only supports F32, got {:?} and {:?}", self.dtype, rhs.dtype);
        }

        // Check layouts are contiguous
        if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
            crate::bail!("WebGPU matmul currently only supports contiguous layouts");
        }

        // Create output buffer
        let output_elem_count = m * n;
        let output_size = (output_elem_count * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matmul Output Buffer"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer for dimensions
        let dims_data = [m as u32, n as u32, k as u32, 0u32]; // Padding for alignment
        let dims_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matmul Dimensions Buffer"),
            size: (dims_data.len() * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write dimensions to uniform buffer
        let dims_bytes = unsafe {
            std::slice::from_raw_parts(
                dims_data.as_ptr() as *const u8,
                dims_data.len() * std::mem::size_of::<u32>(),
            )
        };
        self.device.queue.write_buffer(&dims_buffer, 0, dims_bytes);

        // Get or create compute pipeline
        let pipeline = self.device.get_or_create_pipeline(
            "matmul",
            super::shaders::MATMUL_SHADER,
            "matmul",
        )?;

        // Get bind group layout from the pipeline (automatic layout derivation)
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Create bind group
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch compute shader
        let mut encoder = self.device.create_encoder();
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16x16 workgroup size)
            let workgroup_size_x = 16;
            let workgroup_size_y = 16;
            let num_workgroups_x = (m as u32 + workgroup_size_x - 1) / workgroup_size_x;
            let num_workgroups_y = (n as u32 + workgroup_size_y - 1) / workgroup_size_y;

            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        self.device.submit(encoder);

        Ok(Self {
            buffer: Arc::new(output_buffer),
            device: self.device.clone(),
            dtype: DType::F32,
            elem_count: output_elem_count,
        })
    }

    fn copy_strided_src(&self, _dst: &mut Self, _dst_offset: usize, _src_l: &Layout) -> Result<()> {
        crate::bail!("WebGPU copy_strided_src not yet implemented")
    }

    fn copy2d(
        &self,
        _dst: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_s1: usize,
        _dst_s1: usize,
        _src_o: usize,
        _dst_o: usize,
    ) -> Result<()> {
        crate::bail!("WebGPU copy2d not yet implemented")
    }

    fn const_set(&mut self, _v: crate::scalar::Scalar, _layout: &Layout) -> Result<()> {
        crate::bail!("WebGPU const_set not yet implemented")
    }
}
