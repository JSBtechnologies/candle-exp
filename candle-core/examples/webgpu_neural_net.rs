//! Simple Neural Network on WebGPU
//!
//! This example demonstrates a simple 2-layer MLP running entirely on the GPU
//! using the WebGPU backend.
//!
//! Run with: cargo run --example webgpu_neural_net --features webgpu

use candle_core::{Device, Result, Tensor};

/// Simple 2-layer Multi-Layer Perceptron
struct SimpleMLP {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl SimpleMLP {
    /// Create a new MLP with random weights
    fn new(input_size: usize, hidden_size: usize, output_size: usize, device: &Device) -> Result<Self> {
        // Initialize weights with small random values
        let w1 = Tensor::randn(0f32, 0.1, (input_size, hidden_size), device)?;
        let b1 = Tensor::zeros((1, hidden_size), candle_core::DType::F32, device)?;

        let w2 = Tensor::randn(0f32, 0.1, (hidden_size, output_size), device)?;
        let b2 = Tensor::zeros((1, output_size), candle_core::DType::F32, device)?;

        Ok(Self { w1, b1, w2, b2 })
    }

    /// Forward pass through the network (all operations on GPU!)
    /// Note: For simplicity, we skip bias for now since broadcasting isn't implemented yet
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Layer 1: x * w1 (no bias for simplicity)
        let layer1 = x.matmul(&self.w1)?;
        let layer1_activated = layer1.relu()?;

        // Layer 2: hidden * w2 (no bias for simplicity)
        let output = layer1_activated.matmul(&self.w2)?;

        Ok(output)
    }
}

fn main() -> Result<()> {
    println!("=== Neural Network on WebGPU ===\n");

    // Create WebGPU device
    println!("Initializing WebGPU device...");
    let device = Device::new_webgpu(0)?;
    println!("✓ GPU ready!\n");

    // Network architecture
    let input_size = 4;    // Input features
    let hidden_size = 8;   // Hidden layer neurons
    let output_size = 3;   // Output classes

    println!("Creating neural network:");
    println!("  Input layer:  {} neurons", input_size);
    println!("  Hidden layer: {} neurons (ReLU activation)", hidden_size);
    println!("  Output layer: {} neurons", output_size);

    let model = SimpleMLP::new(input_size, hidden_size, output_size, &device)?;
    println!("✓ Model created with {} parameters\n",
        input_size * hidden_size + hidden_size + hidden_size * output_size + output_size);

    // Create some test input (batch of 2 samples)
    println!("Running inference on GPU...");
    let batch_size = 2;
    let input = Tensor::randn(0f32, 1.0, (batch_size, input_size), &device)?;

    println!("Input shape: {:?}", input.dims());
    println!("Input data:\n{:?}", input.to_vec2::<f32>()?);

    // Forward pass (all on GPU!)
    let output = model.forward(&input)?;

    println!("\nOutput shape: {:?}", output.dims());
    println!("Output data:\n{:?}", output.to_vec2::<f32>()?);

    println!("\n✓ Forward pass completed successfully!");
    println!("\nAll operations executed on GPU:");
    println!("  • Matrix multiplications (2x)");
    println!("  • ReLU activation (1x)");
    println!("  • Total GPU operations: 3");
    println!("\nNote: Bias layers skipped (broadcasting not yet implemented)");

    // Performance test with larger batch
    println!("\n--- Performance Test ---");
    let large_batch = 32;
    let large_input = Tensor::randn(0f32, 1.0, (large_batch, input_size), &device)?;

    println!("Processing batch of {} samples...", large_batch);
    let start = std::time::Instant::now();
    let _ = model.forward(&large_input)?;
    device.synchronize()?; // Wait for GPU to finish
    let duration = start.elapsed();

    println!("✓ Processed {} samples in {:?}", large_batch, duration);
    println!("  Throughput: {:.2} samples/ms", large_batch as f64 / duration.as_millis() as f64);

    println!("\n=== Example Complete ===");
    Ok(())
}
