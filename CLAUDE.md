# quantprobe

A pip-installable CLI tool that diagnoses quantization accuracy loss in ONNX vision models and recommends mixed-precision configurations.

## Tech Stack

- Language: Python 3.9+ (strict type hints everywhere)
- CLI: Typer
- Core deps: onnx, onnxruntime, numpy, Pillow
- Reporting: Plotly + Jinja2 → self-contained HTML
- Terminal output: rich
- Testing: pytest + pytest-cov
- Linting: ruff
- Packaging: pyproject.toml (setuptools)
- CI: GitHub Actions

## Workflow

- For **core modules** (metrics, activation_collector, sensitivity, mixed_precision, analyzer): propose the approach in plain language first, then let the developer implement. Do not write the implementation unless explicitly asked.
- For **boilerplate** (CI config, pyproject.toml, Jinja2 templates, README sections, .gitignore): full implementation is fine.
- Always explain *why* before *how*. The developer is rebuilding engineering muscle and needs to understand decisions, not just receive code.
- When asked to review code, point out issues but let the developer fix them.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_metrics.py -v

# Run with coverage
pytest tests/ --cov=src/quantprobe --cov-report=term-missing

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

CLI commands will be added to this section as they are implemented.

## Code Style

- Type hints on every function signature, including return types
- Docstrings on all public classes and functions (Google style)
- No `Any` types unless unavoidable — comment why if used
- Use `pathlib.Path` for all file paths, never strings
- Use `dataclasses` or `NamedTuple` for structured data, not raw dicts
- Module-level constants in `UPPER_SNAKE_CASE`
- Functions stay under 50 lines; extract helpers if longer
- Prefer explicit over implicit: no `import *`, no magic strings

## Error Handling

- Catch specific exceptions, never bare `except:` or `except Exception:` without re-raising
- Define tool-specific exceptions in `src/quantprobe/exceptions.py` (e.g., `ModelLoadError`, `CalibrationError`, `MetricsError`)
- The CLI must show user-friendly error messages — full tracebacks only with a `--debug` flag
- Validate inputs at function boundaries; fail fast with clear messages

## Testing Rules

- Every module in `src/quantprobe/` has a corresponding `tests/test_*.py`
- Test fixtures (tiny ONNX models, sample data) live in `tests/conftest.py`
- Test ONNX models are generated programmatically using `onnx.helper` — never commit real model files
- Metrics tests must use known input/output pairs with exact expected values
- CLI integration tests use `typer.testing.CliRunner`
- Edge cases to always cover: empty inputs, single-layer models, mismatched shapes, invalid model paths
- Target: >80% coverage before v1.0

## Architecture Principles

- No PyTorch or TensorFlow dependency — ONNX + ORT only
- Must work without a GPU; CPU inference is the default path
- HTML reports must be single self-contained files (inline CSS/JS, embedded Plotly)
- ORT-specific logic stays in `model_runner.py`, `activation_collector.py`, and `quantizer.py` — other modules must not import `onnxruntime`
- `metrics.py` is pure numpy — no ORT dependency, fully unit-testable in isolation
- Every new dependency must justify its inclusion against bundle size and maintenance cost

## Prohibited

- No `print()` — use `rich.console` or the `logging` module
- No hardcoded paths — accept as CLI args or function params
- No `os.path` — use `pathlib`
- No `eval()` or `exec()`
- No wildcard imports
- No mutable default arguments (use `None` and assign in body)
- No `# type: ignore` without a comment explaining why
- No non-ASCII characters in code, comments, docstrings, or string literals. Use `->` not arrows, `<=` not less-than-or-equal symbols, plain `"` and `'` for quotes. If a character cannot be typed on a standard US keyboard, it does not belong in source files.

## Git Workflow

- Branch: `main` only for now (solo developer)
- Commit messages: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`)
- Each commit should leave tests passing and lint clean
- Push at end of every working session
- Don't pad commits — meaningful changes only

## Key Domain Context

- **QDQ** = Quantize-DeQuantize nodes inserted into ONNX graphs to simulate quantization
- **Sensitivity analysis** = measuring how much each layer's output degrades after quantization
- **Approach**: modify ONNX graph to expose intermediate tensors as outputs, run FP32 and quantized models on the same calibration data, compare activations per layer
- **Reference**: ORT's internal `qdq_loss_debug` module is the closest existing implementation — study its source for the graph modification pattern
- **Target quantization**: static INT8 (per-tensor and per-channel) via ORT's quantization API
- **Mixed-precision** = keeping sensitive layers at FP16/FP32 while quantizing the rest to INT8

## Important Notes

- This is a developer tool, not a research project. UX matters: clear errors, progress bars, sensible defaults.
- The HTML report is the flagship feature — it must look professional and be genuinely useful.
- Keep dependencies minimal. Every addition is reviewed against necessity.
- Architecture and design decisions live in `docs/architecture.md` (created as the project evolves).