# Cigma-Sim

**Cigma-Sim** is a PyTorch-based simulator for bit-sparsity-aware GEMM accelerators.
It is designed for architecture-level evaluation with two complementary perspectives:

- **Cycle simulation**: estimates execution cost under specific PE/array configurations.
- **Value simulation**: emulates numerical behavior under hardware-style precision and alignment constraints.

> [!WARNING]
> **Performance Note:** The core framework is specifically designed and optimized to leverage `torch.compile` in graph mode. Please be aware that executing in eager mode will incur substantial performance degradation.

## What It Covers

- Quantized GEMM simulation (quant x quant)
- Floating-point GEMM simulation (float x float)
- Configurable encoding, mapping, windowing, and lane-sharing behavior
- End-to-end model profiling demos in the `example/` package

## Project Layout

- `cigmasim/` — core simulation library
  - `config/`: simulation configuration types
  - `core/`: low-level kernels (cycle counting, alignment, aggregation)
  - `simulator/`: high-level simulation entrypoints
  - `type/`: data format and encoding utilities
- `example/` — runnable model-level demos and profiling scripts
- `scripts/download_datasets.py` — dataset preparation helper

> [!NOTE]
> Presets of Bit-Cigma and Bitlet architectures are defined in `example/architectures` as demos.

## Requirements

- Python 3.11+
- PyTorch 2.6+

## Quick Start

Install with demo dependencies:

```bash
pip install -e ".[demo]"
```

Prepare datasets:

```bash
make prepare_data
```

Run demos:

```bash
make run_all_demo

# or run them separately
make run_resnet152_cycle
make run_t5_cycle
make run_t5_value
```

List all available commands:

```bash
make help
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev,demo]"
```

Code quality checks:

```bash
make format
make lint
make check
```

Cleanup:

```bash
make clean
```

## Citation

If you use this simulator, please cite the Bit-Cigma paper:

```bibtex
@article{bitcigma2025,
  author={Zhu, Zixuan and Zhou, Xiaolong and Wang, Chundong and Tian, Li and Huang, Zunkai and Zhu, Yongxin},
  title={Bit-Sparsity Aware Acceleration With Compact CSD Code on Generic Matrix Multiplication},
  journal={IEEE Transactions on Computers},
  year={2025},
  month={Feb},
  volume={74},
  number={2},
  pages={414--426},
  doi={10.1109/TC.2024.3483632},
}
```
