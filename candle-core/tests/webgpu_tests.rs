#![cfg(feature = "webgpu")]

use candle_core::{DType, Device, Result, Tensor};

#[test]
fn test_webgpu_device_creation() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    assert!(device.is_webgpu());
    Ok(())
}

#[test]
fn test_webgpu_zeros() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let tensor = Tensor::zeros((2, 3), DType::F32, &device)?;

    let data = tensor.to_vec2::<f32>()?;
    assert_eq!(data.len(), 2);
    assert_eq!(data[0].len(), 3);
    for row in &data {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
    Ok(())
}

#[test]
fn test_webgpu_from_slice() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_slice(&data, (2, 3), &device)?;

    let result = tensor.to_vec2::<f32>()?;
    assert_eq!(result, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    Ok(())
}

#[test]
fn test_webgpu_add() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;

    let c = (a + b)?;
    let result = c.to_vec2::<f32>()?;

    assert_eq!(result, vec![vec![11.0, 22.0], vec![33.0, 44.0]]);
    Ok(())
}

#[test]
fn test_webgpu_mul() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_slice(&[2.0f32, 3.0, 4.0, 5.0], (2, 2), &device)?;

    let c = (a * b)?;
    let result = c.to_vec2::<f32>()?;

    assert_eq!(result, vec![vec![2.0, 6.0], vec![12.0, 20.0]]);
    Ok(())
}

#[test]
fn test_webgpu_sub() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;
    let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;

    let c = (a - b)?;
    let result = c.to_vec2::<f32>()?;

    assert_eq!(result, vec![vec![9.0, 18.0], vec![27.0, 36.0]]);
    Ok(())
}

#[test]
fn test_webgpu_div() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;
    let b = Tensor::from_slice(&[2.0f32, 4.0, 5.0, 8.0], (2, 2), &device)?;

    let c = (a / b)?;
    let result = c.to_vec2::<f32>()?;

    assert_eq!(result, vec![vec![5.0, 5.0], vec![6.0, 5.0]]);
    Ok(())
}

#[test]
fn test_webgpu_matmul() -> Result<()> {
    let device = Device::new_webgpu(0)?;

    // Create two 2x2 matrices
    // A = [[1, 2],    B = [[5, 6],
    //      [3, 4]]         [7, 8]]
    // Expected: C = [[19, 22],
    //                [43, 50]]
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], (2, 2), &device)?;

    let c = a.matmul(&b)?;
    let result = c.to_vec2::<f32>()?;

    assert_eq!(result, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    Ok(())
}

#[test]
fn test_webgpu_relu() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], 5, &device)?;

    let b = a.relu()?;
    let result = b.to_vec1::<f32>()?;

    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    Ok(())
}

#[test]
fn test_webgpu_gelu() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[-1.0f32, 0.0, 1.0], 3, &device)?;

    let b = a.gelu()?;
    let result = b.to_vec1::<f32>()?;

    // GELU(-1) ≈ -0.16, GELU(0) = 0, GELU(1) ≈ 0.84
    assert!(result[0] < 0.0 && result[0] > -0.2);
    assert!((result[1] - 0.0).abs() < 0.01);
    assert!(result[2] > 0.8 && result[2] < 0.9);
    Ok(())
}

#[test]
fn test_webgpu_tanh() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[-1.0f32, 0.0, 1.0], 3, &device)?;

    let b = a.tanh()?;
    let result = b.to_vec1::<f32>()?;

    // tanh(-1) ≈ -0.76, tanh(0) = 0, tanh(1) ≈ 0.76
    assert!((result[0] - (-0.76)).abs() < 0.01);
    assert!((result[1] - 0.0).abs() < 0.01);
    assert!((result[2] - 0.76).abs() < 0.01);
    Ok(())
}

#[test]
fn test_webgpu_large_matmul() -> Result<()> {
    let device = Device::new_webgpu(0)?;

    // Test with larger matrices (64x64)
    let size = 64;
    let a_data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();

    let a = Tensor::from_slice(&a_data, (size, size), &device)?;
    let b = Tensor::from_slice(&b_data, (size, size), &device)?;

    let c = a.matmul(&b)?;
    let result = c.flatten_all()?.to_vec1::<f32>()?;

    // Just verify we got the right number of elements and no NaN/Inf
    assert_eq!(result.len(), size * size);
    for &val in &result {
        assert!(val.is_finite());
    }
    Ok(())
}

#[test]
fn test_webgpu_exp() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[0.0f32, 1.0, 2.0], 3, &device)?;

    let b = a.exp()?;
    let result = b.to_vec1::<f32>()?;

    // exp(0) = 1, exp(1) ≈ 2.718, exp(2) ≈ 7.389
    assert!((result[0] - 1.0).abs() < 0.01);
    assert!((result[1] - 2.718).abs() < 0.01);
    assert!((result[2] - 7.389).abs() < 0.01);
    Ok(())
}

#[test]
fn test_webgpu_log() -> Result<()> {
    let device = Device::new_webgpu(0)?;
    let a = Tensor::from_slice(&[1.0f32, 2.718, 7.389], 3, &device)?;

    let b = a.log()?;
    let result = b.to_vec1::<f32>()?;

    // log(1) = 0, log(e) = 1, log(e^2) = 2
    assert!((result[0] - 0.0).abs() < 0.01);
    assert!((result[1] - 1.0).abs() < 0.01);
    assert!((result[2] - 2.0).abs() < 0.01);
    Ok(())
}
