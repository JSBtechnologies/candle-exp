//! WebGPU Backend Demo
//!
//! This example demonstrates the WebGPU backend for Candle, showing
//! GPU-accelerated tensor operations that work in browsers and on native platforms.
//!
//! Run with: cargo run --example webgpu_demo --features webgpu

use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    println!("=== Candle WebGPU Backend Demo ===\n");

    // Create WebGPU device
    println!("Creating WebGPU device...");
    let device = Device::new_webgpu(0)?;
    println!("✓ WebGPU device created successfully!\n");

    // Demo 1: Basic tensor operations
    println!("--- Demo 1: Basic Tensor Operations ---");
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], (2, 2), &device)?;

    println!("Tensor A:\n{:?}", a.to_vec2::<f32>()?);
    println!("Tensor B:\n{:?}", b.to_vec2::<f32>()?);

    let sum = (&a + &b)?;
    println!("A + B = \n{:?}", sum.to_vec2::<f32>()?);

    let product = (&a * &b)?;
    println!("A * B (element-wise) = \n{:?}\n", product.to_vec2::<f32>()?);

    // Demo 2: Matrix multiplication
    println!("--- Demo 2: Matrix Multiplication ---");
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let y = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], (2, 2), &device)?;

    let result = x.matmul(&y)?;
    println!("Matrix multiplication result:\n{:?}\n", result.to_vec2::<f32>()?);

    // Demo 3: Activation functions
    println!("--- Demo 3: Activation Functions ---");
    let data = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], 5, &device)?;
    println!("Input: {:?}", data.to_vec1::<f32>()?);

    let relu_result = data.relu()?;
    println!("ReLU:  {:?}", relu_result.to_vec1::<f32>()?);

    let gelu_result = data.gelu()?;
    println!("GELU:  {:?}", gelu_result.to_vec1::<f32>()?);

    let tanh_result = data.tanh()?;
    println!("Tanh:  {:?}\n", tanh_result.to_vec1::<f32>()?);

    // Demo 4: Mathematical functions
    println!("--- Demo 4: Mathematical Functions ---");
    let values = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], 4, &device)?;
    println!("Input: {:?}", values.to_vec1::<f32>()?);

    let exp_result = values.exp()?;
    println!("Exp:   {:?}", exp_result.to_vec1::<f32>()?);

    let log_input = Tensor::from_slice(&[1.0f32, 2.718, 7.389, 20.0], 4, &device)?;
    let log_result = log_input.log()?;
    println!("Log:   {:?}\n", log_result.to_vec1::<f32>()?);

    // Demo 5: Chained operations (all on GPU!)
    println!("--- Demo 5: Chained GPU Operations ---");
    let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let weights = Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], (2, 2), &device)?;

    println!("Input tensor:\n{:?}", input.to_vec2::<f32>()?);
    println!("Weights:\n{:?}", weights.to_vec2::<f32>()?);

    // Chain multiple operations: matmul -> add -> relu
    let intermediate = input.matmul(&weights)?;
    println!("After matmul:\n{:?}", intermediate.to_vec2::<f32>()?);

    let bias = Tensor::from_slice(&[0.1f32, 0.1, 0.1, 0.1], (2, 2), &device)?;
    let added = (&intermediate + &bias)?;
    println!("After add bias:\n{:?}", added.to_vec2::<f32>()?);

    let activated = added.relu()?;
    println!("After ReLU:\n{:?}", activated.to_vec2::<f32>()?);

    println!("\n✓ All operations completed successfully on GPU!");
    println!("\n=== Demo Complete ===");
    println!("The WebGPU backend enables:");
    println!("  • GPU-accelerated tensor operations");
    println!("  • Browser compatibility (via WebGPU)");
    println!("  • Cross-platform support (Windows, Mac, Linux)");
    println!("  • WASM compilation for web deployment");

    Ok(())
}
