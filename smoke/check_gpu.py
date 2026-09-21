"""Quick diagnostic of the training environment. Run this before anything else on the DGX.

    uv run arca-check-gpu
    # or, inside Docker:
    docker compose run --rm check-gpu

It prints what the code will see: driver, CUDA, GPUs and their memory, library versions,
and it runs a small matrix multiplication on the GPU to confirm it actually computes.
Exit code is 0 when a usable GPU is found and 1 otherwise, so you can use it in scripts.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def nvidia_smi_summary() -> str:
    """Return the driver/GPU line from nvidia-smi, or an explanation of why it is missing."""
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found (no NVIDIA driver visible from this environment)"
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return out.stdout.strip() or "nvidia-smi returned no GPUs"
    except Exception as exc:  # noqa: BLE001 - we want to show whatever went wrong
        return f"nvidia-smi failed: {exc}"


def library_versions() -> Table:
    table = Table(title="Libraries", show_header=True, header_style="bold")
    table.add_column("package")
    table.add_column("version")
    table.add_row("python", platform.python_version())
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            module = __import__(name)
            table.add_row(name, getattr(module, "__version__", "?"))
        except ImportError:
            table.add_row(name, "[red]not installed[/red] (run: uv sync --extra train)")
    return table


def benchmark(device: str, size: int = 4096, repeats: int = 10) -> float:
    """Multiply two size x size matrices ``repeats`` times and return TFLOP/s."""
    import torch

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    torch.matmul(a, b)  # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        torch.matmul(a, b)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    flops = 2 * size**3 * repeats
    return flops / elapsed / 1e12


def main() -> int:
    console.print(Panel.fit("[bold]ARCA · environment check[/bold]", border_style="cyan"))
    console.print(
        f"[bold]Host:[/bold] {platform.node()} · {platform.system()} {platform.release()}"
    )
    console.print(f"[bold]nvidia-smi:[/bold] {nvidia_smi_summary()}")
    console.print(library_versions())

    try:
        import torch
    except ImportError:
        console.print("[red]torch is not installed. Run: uv sync --extra train[/red]")
        return 1

    console.print(f"[bold]torch.version.cuda:[/bold] {torch.version.cuda}")
    if not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        console.print(
            f"[yellow]No CUDA device visible to torch. Falling back to {device}.[/yellow]\n"
            "On the DGX this usually means the SLURM session was launched without a GPU: "
            "check the resources you requested in the Open OnDemand portal. In Docker, check "
            "--gpus / the compose GPU reservation and the driver version."
        )
        tflops = benchmark(device, size=1024, repeats=5)
        console.print(f"{device} matmul: {tflops:.2f} TFLOP/s (just to prove the install works)")
        return 1

    table = Table(title="GPUs visible to torch", show_header=True, header_style="bold")
    table.add_column("index")
    table.add_column("name")
    table.add_column("memory (GB)")
    table.add_column("compute capability")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        table.add_row(
            str(idx),
            props.name,
            f"{props.total_memory / 1e9:.1f}",
            f"{props.major}.{props.minor}",
        )
    console.print(table)

    tflops = benchmark("cuda")
    console.print(f"[bold green]GPU 0 bf16 matmul: {tflops:.1f} TFLOP/s[/bold green]")
    console.print(
        f"bf16 supported: {torch.cuda.is_bf16_supported()} · "
        f"memory allocated after benchmark: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )
    console.print(Panel.fit("[bold green]Ready to train.[/bold green]", border_style="green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
