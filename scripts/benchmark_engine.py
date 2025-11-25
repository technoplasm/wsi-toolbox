#!/usr/bin/env python3
"""
Benchmark DZI tile extraction for OpenSlide vs tifffile engines.

Usage:
    uv run python benchmark_engine.py --iterations 3
    uv run python benchmark_engine.py --iterations 5 --output results.csv
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from wsi_toolbox.wsi_files import OpenSlideFile, PyramidalTiffFile, PyramidalWSIFile

# Relative to repository root
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "benchmarks"

# Extension to engines mapping
# (openslide_supported, tifffile_supported)
ENGINE_BY_EXT = {
    ".svs": (True, False),   # SVS: OpenSlide only
    ".ndpi": (True, True),   # NDPI: both
    ".tif": (True, True),    # TIFF: both
    ".tiff": (True, True),   # TIFF: both
    ".mrxs": (True, False),  # MIRAX: OpenSlide only
    ".vms": (True, False),   # Hamamatsu: OpenSlide only
    ".scn": (True, False),   # Leica: OpenSlide only
    ".bif": (True, False),   # Ventana: OpenSlide only
}


def discover_benchmark_files() -> list[tuple[str, bool, bool]]:
    """Discover benchmark files in DATA_DIR and determine compatible engines."""
    files = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in ENGINE_BY_EXT:
            openslide_ok, tifffile_ok = ENGINE_BY_EXT[ext]
            files.append((path.name, openslide_ok, tifffile_ok))
    return files


def count_total_tiles(wsi: PyramidalWSIFile, tile_size: int = 256) -> int:
    """Count total number of DZI tiles across all levels."""
    total = 0
    max_level = wsi.get_dzi_max_level()
    for level in range(max_level + 1):
        _, _, cols, rows = wsi.get_dzi_level_info(level, tile_size)
        total += cols * rows
    return total


def benchmark_all_tiles(
    wsi: PyramidalWSIFile,
    tile_size: int = 256,
    overlap: int = 0,
    show_progress: bool = True,
) -> float:
    """Benchmark reading all DZI tiles. Returns elapsed time in seconds."""
    start = time.perf_counter()

    max_level = wsi.get_dzi_max_level()

    # Outer loop: levels (high to low resolution)
    level_pbar = tqdm(
        range(max_level, -1, -1),
        disable=not show_progress,
        unit="level",
        desc="Levels",
        position=0,
    )

    for level in level_pbar:
        _, _, cols, rows = wsi.get_dzi_level_info(level, tile_size)
        level_tiles = cols * rows
        level_pbar.set_postfix(level=level, tiles=level_tiles)

        # Inner loop: tiles in this level
        tiles_iter = (
            (row, col)
            for row in range(rows)
            for col in range(cols)
        )

        for row, col in tqdm(
            tiles_iter,
            total=level_tiles,
            disable=not show_progress,
            unit="tiles",
            desc=f"  Level {level:2d}",
            position=1,
            leave=False,
        ):
            tile = wsi.get_dzi_tile(level, col, row, tile_size, overlap)
            del tile

    return time.perf_counter() - start


def run_benchmark(iterations: int) -> list[dict]:
    """Run benchmark for all files and engines."""
    results = []
    benchmark_files = discover_benchmark_files()

    if not benchmark_files:
        print(f"No benchmark files found in {DATA_DIR}")
        return results

    print(f"Found {len(benchmark_files)} files to benchmark")

    for filename, openslide_ok, tifffile_ok in benchmark_files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Skipping {filename}: file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Benchmarking: {filename}")
        print(f"{'='*60}")

        # OpenSlide benchmark
        if openslide_ok:
            print(f"\n[OpenSlide] Loading {filename}...")
            try:
                wsi = OpenSlideFile(str(filepath))
                width, height = wsi.get_original_size()
                max_level = wsi.get_dzi_max_level()
                total_tiles = count_total_tiles(wsi)

                print(f"  Size: {width}x{height}, Levels: {max_level+1}, Tiles: {total_tiles}")

                for i in range(iterations):
                    elapsed = benchmark_all_tiles(wsi)
                    tiles_per_sec = total_tiles / elapsed
                    print(f"  Iteration {i+1}/{iterations}: {elapsed:.2f}s ({tiles_per_sec:.1f} tiles/s)")

                    results.append({
                        "file": filename,
                        "engine": "openslide",
                        "iteration": i + 1,
                        "width": width,
                        "height": height,
                        "max_level": max_level,
                        "total_tiles": total_tiles,
                        "elapsed_sec": elapsed,
                        "tiles_per_sec": tiles_per_sec,
                    })
            except Exception as e:
                print(f"  ERROR: {e}")

        # tifffile benchmark
        if tifffile_ok:
            print(f"\n[tifffile] Loading {filename}...")
            try:
                wsi = PyramidalTiffFile(str(filepath))
                width, height = wsi.get_original_size()
                max_level = wsi.get_dzi_max_level()
                total_tiles = count_total_tiles(wsi)

                print(f"  Size: {width}x{height}, Levels: {max_level+1}, Tiles: {total_tiles}")

                for i in range(iterations):
                    elapsed = benchmark_all_tiles(wsi)
                    tiles_per_sec = total_tiles / elapsed
                    print(f"  Iteration {i+1}/{iterations}: {elapsed:.2f}s ({tiles_per_sec:.1f} tiles/s)")

                    results.append({
                        "file": filename,
                        "engine": "tifffile",
                        "iteration": i + 1,
                        "width": width,
                        "height": height,
                        "max_level": max_level,
                        "total_tiles": total_tiles,
                        "elapsed_sec": elapsed,
                        "tiles_per_sec": tiles_per_sec,
                    })
            except Exception as e:
                print(f"  ERROR: {e}")

    return results


def save_results(results: list[dict], output_path: str):
    """Save results to CSV."""
    if not results:
        print("No results to save")
        return

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def print_summary(results: list[dict]):
    """Print summary statistics."""
    if not results:
        return

    print(f"\n{'='*60}")
    print("Summary (average tiles/sec)")
    print(f"{'='*60}")

    # Group by file and engine
    grouped = defaultdict(list)
    for r in results:
        key = (r["file"], r["engine"])
        grouped[key].append(r["tiles_per_sec"])

    # Print summary
    current_file = None
    for (filename, engine), speeds in sorted(grouped.items()):
        if filename != current_file:
            if current_file is not None:
                print()
            print(f"\n{filename}:")
            current_file = filename
        avg_speed = sum(speeds) / len(speeds)
        print(f"  {engine:12s}: {avg_speed:8.1f} tiles/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DZI tile extraction")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=3,
        help="Number of iterations per engine (default: 3)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file (default: benchmark_results.csv)"
    )
    args = parser.parse_args()

    print(f"Running DZI benchmark with {args.iterations} iterations per engine")
    print(f"Data directory: {DATA_DIR.absolute()}")

    results = run_benchmark(args.iterations)
    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
