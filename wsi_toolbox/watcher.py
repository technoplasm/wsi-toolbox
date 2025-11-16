import os
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Set, Callable, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .utils import plot_umap
from . import commands

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'uni')

class Status:
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    ERROR = "ERROR"

    @classmethod
    def is_processing_state(cls, status: str) -> bool:
        """状態が処理中系かどうかを判定"""
        return status.startswith((cls.PROCESSING, cls.DONE, cls.ERROR))

class Task:
    REQUEST_FILE = "_ROBIEMON.txt"
    LOG_FILE = "_ROBIEMON_LOG.txt"

    @staticmethod
    def parse_request_line(line: str) -> tuple[str, bool]:
        """Parse the request line for model and rotation specifications.
        Returns (model_name, should_rotate)"""
        parts = [p.strip() for p in line.split(',')]
        model_name = parts[0] if parts and parts[0] else DEFAULT_MODEL
        should_rotate = len(parts) > 1 and parts[1].lower() == 'rotate'
        return model_name, should_rotate

    def __init__(self, folder:Path, options_line:str, on_complete:Optional[Callable[[Path], None]] = None):
        self.folder = folder
        self.options_line = options_line
        self.model_name, self.should_rotate = self.parse_request_line(options_line)
        self.on_complete = on_complete
        self.wsi_files = list(folder.glob("**/*.ndpi")) + list(folder.glob("**/*.svs"))
        self.wsi_files.sort()

        commands.set_default_progress('tqdm')
        commands.set_default_model(self.model_name)

    def write_banner(self):
        """処理開始時のバナーをログに書き込み"""
        self.append_log("="*50)
        self.append_log(f"Processing folder: {self.folder}")
        self.append_log(f"Request options: {self.options_line}")
        self.append_log(f"Parsed options:")
        self.append_log(f"  - Model: {self.model_name} (default: {DEFAULT_MODEL})")
        self.append_log(f"  - Rotation: {'enabled' if self.should_rotate else 'disabled'}")
        self.append_log(f"Found {len(self.wsi_files)} WSI files:")
        for i, wsi_file in enumerate(self.wsi_files, 1):
            size_mb = wsi_file.stat().st_size / (1024 * 1024)
            self.append_log(f"  {i}. {wsi_file.name} ({size_mb:.1f} MB)")
        self.append_log("="*50)

    def run(self):
        try:
            # ログファイルをクリア
            with open(self.folder / self.LOG_FILE, "w") as f:
                f.write("")

            self.set_status(Status.PROCESSING)
            self.write_banner()

            # WSIファイルごとの処理
            for i, wsi_file in enumerate(self.wsi_files):
                try:
                    self.append_log(f"Processing [{i+1}/{len(self.wsi_files )}]: {wsi_file.name}")

                    hdf5_tmp_path = wsi_file.with_suffix('.h5.tmp')
                    hdf5_file = wsi_file.with_suffix(".h5")

                    # HDF5変換（既存の場合はスキップ）
                    if not hdf5_file.exists():
                        self.append_log("Converting to HDF5...")
                        # Use new command pattern
                        commands.set_default_progress('tqdm')
                        cmd = commands.Wsi2HDF5Command(rotate=self.should_rotate)
                        result = cmd(str(wsi_file), str(hdf5_tmp_path))
                        os.rename(hdf5_tmp_path, hdf5_file)
                        self.append_log("HDF5 conversion completed.")

                    # 特徴量抽出（既存の場合はスキップ）
                    self.append_log("Extracting features...")
                    # Use new command pattern
                    commands.set_default_device('cuda')
                    emb_cmd = commands.PatchEmbeddingCommand()
                    emb_result = emb_cmd(str(hdf5_file))
                    self.append_log("Feature extraction completed.")

                    # クラスタリングとUMAP生成
                    self.append_log("Starting clustering ...")
                    # Use new command pattern
                    cluster_cmd = commands.ClusteringCommand(
                        resolution=1.0,
                        use_umap=True
                    )
                    cluster_result = cluster_cmd([hdf5_file])
                    self.append_log("Clustering completed.")

                    base = str(wsi_file.with_suffix(""))

                    # UMAPプロット生成
                    self.append_log("Starting UMAP generation...")
                    umap_path = Path(f"{base}_umap.png")
                    if not umap_path.exists():
                        umap_embs = cluster_cmd.get_umap_embeddings()
                        fig = plot_umap(umap_embs, cluster_cmd.total_clusters)
                        fig.savefig(umap_path, bbox_inches='tight', pad_inches=0.5)
                        self.append_log(f"UMAP plot completed. Saved to {os.path.basename(umap_path)}")
                    else:
                        self.append_log(f"UMAP plot already exists. Skipped.")

                    # サムネイル生成
                    self.append_log("Starting thumbnail generation...")
                    thumb_path = Path(f"{base}_thumb.jpg")
                    if not thumb_path.exists():
                        # Use new command pattern
                        preview_cmd = commands.PreviewClustersCommand(size=64)
                        img = preview_cmd(str(hdf5_file), cluster_name='')
                        img.save(thumb_path)
                        self.append_log(f"Thumbnail generation completed. Saved to {thumb_path.name}")
                    else:
                        self.append_log(f"Thumbnail already exists. Skipped.")

                    self.append_log("="*30)

                except Exception as e:
                    self.append_log(f"Error processing {wsi_file}: {str(e)}")
                    self.set_status(Status.ERROR)
                    if self.on_complete:
                        self.on_complete(self.folder)
                    return

            self.set_status(Status.DONE)
            self.append_log("All processing completed successfully")

        except Exception as e:
            self.append_log(f"Error: {str(e)}")

        if self.on_complete:
            self.on_complete(self.folder)

    def set_status(self, status: str):
        self.status = status
        with open(self.folder / self.REQUEST_FILE, "w") as f:
            f.write(f"{status}\n")

    def append_log(self, message: str):
        with open(self.folder / self.LOG_FILE, "a") as f:
            f.write(message + "\n")
            print(message)

class Watcher:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.running_tasks: Dict[Path, Task] = {}
        self.console = Console()

    def run(self, interval: int = 60):
        self.console.print("\n[bold blue]ROBIEMON Watcher started[/]")
        self.console.print(f"[blue]Watching directory:[/] {self.base_dir}")
        self.console.print(f"[blue]Polling interval:[/] {interval} seconds")
        self.console.print("[yellow]Press Ctrl+C to stop[/]\n")

        while True:
            try:
                self.check_folders()

                # カウントダウン表示
                for remaining in range(interval, 0, -1):
                    print(f"\rNext check in {remaining:2d}s", end="", flush=True)
                    time.sleep(1)
                # カウントダウン終了後、同じ行を再利用
                print("\rNext check in  0s", end="", flush=True)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Stopping watcher...[/]")
                break
            except Exception as e:
                self.console.print(f"[red]ERROR:[/] {str(e)}")

    def check_folders(self):
        for folder in self.base_dir.rglob("*"):
            if not folder.is_dir():
                continue

            request_file = folder / Task.REQUEST_FILE
            if not request_file.exists():
                continue

            if folder in self.running_tasks:
                continue

            try:
                with open(request_file, "r") as f:
                    content = f.read()
                    if not content.strip():
                        continue

                    # First line contains model/rotation specs
                    options_line = content.split('\n')[0].strip()

                    # Original status check from the entire file
                    status = content.strip()

            except:
                continue

            if Status.is_processing_state(status):
                continue

            # \rを含むログから改行するため空白行を挿入
            print()
            print()
            print(f"detected: {folder}")
            print(f"Request options: {options_line}")

            task = Task(folder, options_line, on_complete=lambda f: self.running_tasks.pop(f, None))
            self.running_tasks[folder] = task
            task.run()  # 同期実行に変更

BASE_DIR = os.getenv('BASE_DIR', 'data')

def main():
    parser = argparse.ArgumentParser(description="ROBIEMON WSI Processor Watcher")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE_DIR,
        help="Base directory to watch for WSI processing requests"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)"
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory '{args.base_dir}' does not exist")
        return
    if not base_dir.is_dir():
        print(f"Error: '{args.base_dir}' is not a directory")
        return

    watcher = Watcher(args.base_dir)
    watcher.run(interval=args.interval)  # asyncio.runを削除

if __name__ == "__main__":
    main()
