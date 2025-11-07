//! WGSL Compute Shaders for WebGPU Backend

/// Matrix multiplication shader using tiled computation
/// Computes C = A * B where A is (M x K) and B is (K x N)
pub const MATMUL_SHADER: &str = r#"
struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    let M = dims.M;
    let N = dims.N;
    let K = dims.K;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < K; i = i + 1u) {
        let a_idx = row * K + i;
        let b_idx = i * N + col;
        sum = sum + input_a[a_idx] * input_b[b_idx];
    }

    output[row * N + col] = sum;
}
"#;

/// Optimized tiled matrix multiplication for larger matrices
pub const MATMUL_TILED_SHADER: &str = r#"
struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn matmul_tiled(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    let M = dims.M;
    let N = dims.N;
    let K = dims.K;

    var sum = 0.0;

    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from A
        let a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            tile_a[local_row][local_col] = input_a[row * K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile from B
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            tile_b[local_row][local_col] = input_b[b_row * N + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial sum for this tile
        for (var i = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row][i] * tile_b[i][local_col];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}
"#;

/// Element-wise addition: out = a + b
pub const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = input_a[idx] + input_b[idx];
}
"#;

/// Element-wise multiplication: out = a * b
pub const MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = input_a[idx] * input_b[idx];
}
"#;

/// Element-wise subtraction: out = a - b
pub const SUB_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = input_a[idx] - input_b[idx];
}
"#;

/// Element-wise division: out = a / b
pub const DIV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = input_a[idx] / input_b[idx];
}
"#;

/// ReLU activation: out = max(0, x)
pub const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = max(0.0, input[idx]);
}
"#;

/// GELU activation (approximate)
pub const GELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let x = input[idx];
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"#;

/// Tanh activation
pub const TANH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn tanh_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = tanh(input[idx]);
}
"#;

/// Exponential function: out = exp(x)
pub const EXP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn exp_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = exp(input[idx]);
}
"#;

/// Natural logarithm: out = log(x)
pub const LOG_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn log_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = log(input[idx]);
}
"#;

/// Sigmoid activation: out = 1 / (1 + exp(-x))
pub const SIGMOID_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let x = input[idx];
    output[idx] = 1.0 / (1.0 + exp(-x));
}
"#;

/// Sum reduction: sum all elements
/// Simple version that sums everything to output[0]
pub const SUM_REDUCE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> input_size: u32;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    if (gid < input_size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s && gid + s < input_size) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (tid == 0u) {
        atomicAdd(&output[0], shared_data[0]);
    }
}
"#;
