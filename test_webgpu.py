#!/usr/bin/env python3
"""Test WebGPU support in candle Python bindings"""

import candle

print("=== Testing Candle WebGPU Python Bindings ===\n")

# Test device creation
print("Creating tensors and moving to WebGPU...")
try:
    # Create tensors on CPU first
    a_cpu = candle.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b_cpu = candle.Tensor([[5.0, 6.0], [7.0, 8.0]])

    print(f"Created tensors on CPU:")
    print(f"  Device: {a_cpu.device}")

    # Move to WebGPU
    print("\nMoving tensors to WebGPU...")
    a = a_cpu.to_device("webgpu")
    b = b_cpu.to_device("webgpu")

    print("✓ Tensors moved to WebGPU successfully!")
    print(f"  Device: {a.device}")

    # Test basic operations on GPU
    print("\nTesting GPU operations...")
    print(f"  Tensor A: {a.values()}")
    print(f"  Tensor B: {b.values()}")

    # Addition on GPU
    c = a + b
    print(f"  A + B (on GPU) = {c.values()}")

    # Multiplication on GPU
    d = a * b
    print(f"  A * B (element-wise on GPU) = {d.values()}")

    # Matrix multiplication on GPU
    print("\nTesting matrix multiplication on GPU...")
    result = a.matmul(b)
    print(f"  A @ B = {result.values()}")

    print("\n✓ All tests passed!")
    print("\n🎉 WebGPU is now available in candle Python bindings!")
    print("   You can now use GPU acceleration in your Python applications!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
