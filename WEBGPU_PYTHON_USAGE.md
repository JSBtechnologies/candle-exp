# WebGPU Python Bindings - Usage Guide

## ✅ Successfully Added WebGPU Support!

WebGPU is now available in candle's Python bindings. You can use GPU acceleration in your Python applications!

## Installation

### 1. Build the Python Package

From the `candle-pyo3` directory:

```bash
# Create virtual environment (one time)
python3 -m venv .venv
source .venv/bin/activate

# Install maturin (one time)
pipx install maturin

# Build and install candle with WebGPU support
export PATH="$HOME/.cargo/bin:$PATH"
maturin develop --features webgpu --release
```

### 2. Use in Your Python Code

```python
import candle

# Create tensors on CPU
a = candle.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = candle.Tensor([[5.0, 6.0], [7.0, 8.0]])

# Move to WebGPU for GPU acceleration
a_gpu = a.to_device("webgpu")
b_gpu = b.to_device("webgpu")

# All operations run on GPU!
result = a_gpu + b_gpu  # Addition on GPU
result = a_gpu * b_gpu  # Element-wise multiplication on GPU
result = a_gpu.matmul(b_gpu)  # Matrix multiplication on GPU

# Move back to CPU to get results
result_cpu = result.to_device("cpu")
print(result_cpu.values())
```

## Example: RAG Assistant for OpenLogin Migrations

Here's how you could build a RAG-powered migration assistant:

```python
# openlogin_migration_helper.py

import candle
from pathlib import Path
import json

class MigrationHelper:
    """AI-powered assistant for OpenLogin migrations"""

    def __init__(self, docs_path: str):
        # Use WebGPU for GPU acceleration
        self.device = "webgpu"

        # Initialize embedding model and LLM (pseudo-code)
        # self.embedding_model = load_embedding_model(self.device)
        # self.llm = load_llm(self.device)
        # self.vector_db = VectorDB()

        # Index documentation
        self._index_documentation(docs_path)

    def _index_documentation(self, docs_path: str):
        """Index all migration docs, examples, and API references"""
        docs = []
        for file in Path(docs_path).rglob("*.md"):
            with open(file) as f:
                content = f.read()
                docs.append({"file": str(file), "content": content})

        # Create embeddings on GPU
        for doc in docs:
            # chunks = chunk_text(doc['content'])
            # for chunk in chunks:
            #     embedding = self.embedding_model.encode(chunk)
            #     self.vector_db.add(chunk, embedding)
            pass

    def ask(self, question: str) -> str:
        """Ask a question about migrations"""
        # 1. Embed question (GPU)
        # question_embedding = self.embedding_model.encode(question)

        # 2. Find relevant docs (CPU - fast Rust vector search)
        # relevant_chunks = self.vector_db.search(question_embedding, top_k=5)

        # 3. Build context
        # context = "\\n\\n".join(relevant_chunks)

        # 4. Generate response (GPU)
        # prompt = f"Question: {question}\\n\\nContext:\\n{context}\\n\\nAnswer:"
        # response = self.llm.generate(prompt)

        return "Response from LLM (GPU-accelerated)"

    def explain_error(self, error_message: str) -> dict:
        """Explain a migration error and suggest fixes"""
        question = f"I got this error during migration: {error_message}. What does it mean and how do I fix it?"
        explanation = self.ask(question)

        return {
            "error": error_message,
            "explanation": explanation,
            "suggested_fix": "Based on the context from docs..."
        }

    def suggest_migration_steps(self, from_version: str, to_version: str) -> list:
        """Get step-by-step migration guide"""
        question = f"How do I migrate from OpenLogin {from_version} to {to_version}?"
        return self.ask(question)


# Usage
if __name__ == "__main__":
    # Initialize helper
    helper = MigrationHelper("/path/to/openlogin/docs")

    # Ask questions
    print(helper.ask("How do I initialize OpenLogin in v2?"))

    # Explain errors
    error = helper.explain_error("TypeError: Cannot read property 'sessionId' of undefined")
    print(json.dumps(error, indent=2))

    # Get migration guide
    steps = helper.suggest_migration_steps("v1.0", "v2.0")
    print(steps)
```

## Benefits for Your Use Case

### 1. **Privacy**
- All data stays local
- No API keys needed
- No data sent to external services

### 2. **Cost**
- Zero API costs (no OpenAI/Anthropic fees)
- Unlimited queries
- One-time setup

### 3. **Speed**
- GPU acceleration for embeddings and LLM
- No network latency
- Instant responses

### 4. **Offline**
- Works without internet
- Perfect for secure environments
- Reliable for production use

## Architecture

```
┌─────────────────────────────────────┐
│  OpenLogin Migration Tool (Python)  │
│  - CLI interface                    │
│  - Error handling                   │
│  - Migration scripts                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  RAG Assistant (Python)             │
│  - Document indexing                │
│  - Question answering               │
│  - Error explanation                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Candle WebGPU (Rust Core)          │
│  - Embedding model                  │  ← GPU accelerated
│  - LLM inference                    │
│  - Vector operations                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  WebGPU Backend                     │
│  - Cross-platform GPU support       │
│  - Works on Mac, Windows, Linux     │
└─────────────────────────────────────┘
```

## Next Steps

### For Production Use:

1. **Add a proper embedding model** (e.g., sentence-transformers)
2. **Integrate a small LLM** (e.g., Phi-3, Mistral-7B)
3. **Build Rust vector search** for fast retrieval
4. **Create a CLI interface** for easy use

### Example CLI:

```bash
# Index documentation
openlogin-helper index ./docs

# Ask questions
openlogin-helper ask "How do I handle session timeout?"

# Explain errors
openlogin-helper explain "SessionError: Invalid token"

# Interactive mode
openlogin-helper interactive
```

## Testing

Run the test to verify WebGPU works:

```bash
python3 test_webgpu.py
```

You should see:
```
✓ Tensors moved to WebGPU successfully!
✓ All tests passed!
🎉 WebGPU is now available in candle Python bindings!
```

## Available Operations

Currently supported on WebGPU:
- ✅ Tensor creation and transfer
- ✅ Element-wise operations (add, mul, etc.)
- ✅ Matrix multiplication (contiguous tensors)
- ✅ Activation functions (ReLU, GELU, Tanh)
- ✅ Math functions (exp, log, etc.)

## Troubleshooting

### Build Fails
```bash
# Make sure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Clean and rebuild
cargo clean
maturin develop --features webgpu --release
```

### Import Error
```bash
# Make sure virtual environment is activated
source candle-pyo3/.venv/bin/activate
```

### Device Not Found
```bash
# Check WebGPU support
python3 -c "import candle; t = candle.Tensor([1.0]); print(t.to_device('webgpu').device)"
```
