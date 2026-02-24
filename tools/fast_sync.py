#!/usr/bin/env python3
"""
高速并行文件同步工具 - 针对大文件优化 (safetensors/pth 5-10GB)

核心策略: 单个大文件内部分块，多线程并行 pread/pwrite，
把 NFS/CephFS 单连接 ~40MB/s 的瓶颈通过多路并发叠加带宽。

用法:
    python fast_sync.py <src> <dst> [--workers 32] [--chunk-mb 64] [--skip-existing] [--dry-run]

示例:
    python fast_sync.py /group/.../Wan2.2-T2V-A14B /tmp/.../Wan2.2-T2V-A14B --workers 32
    SYNC_WORKERS=64 python fast_sync.py /group/.../models /tmp/.../models --skip-existing
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def scan_files(src_dir: str):
    """扫描源目录，返回 (相对路径, 文件大小) 列表，按大小降序"""
    files = []
    src = Path(src_dir)
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            files.append((str(rel), f.stat().st_size))
    files.sort(key=lambda x: x[1], reverse=True)
    return files


def copy_chunk(src_fd, dst_fd, offset, length):
    """用 pread/pwrite 拷贝一个分块（线程安全，不需要 seek）"""
    buf = os.pread(src_fd, length, offset)
    os.pwrite(dst_fd, buf, offset)
    return len(buf)


def copy_file_parallel(src_path, dst_path, chunk_size, max_workers, progress_cb=None):
    """
    多线程分块并行拷贝单个文件。
    使用临时文件 + rename 保证原子性：中断后不会留下"大小正确但内容残缺"的文件。
    """
    file_size = os.path.getsize(src_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if file_size == 0:
        open(dst_path, "wb").close()
        return 0

    # 写到临时文件，完成后 rename（原子性保证）
    tmp_path = dst_path + ".synctmp"

    try:
        if file_size <= chunk_size:
            # 小文件直接单线程拷贝
            src_fd = os.open(src_path, os.O_RDONLY)
            dst_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                buf = os.pread(src_fd, file_size, 0)
                os.pwrite(dst_fd, buf, 0)
            finally:
                os.close(src_fd)
                os.close(dst_fd)
            if progress_cb:
                progress_cb(file_size)
        else:
            # 大文件: 预分配 + 多线程分块
            src_fd = os.open(src_path, os.O_RDONLY)
            dst_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(dst_fd, file_size)

                chunks = []
                offset = 0
                while offset < file_size:
                    length = min(chunk_size, file_size - offset)
                    chunks.append((offset, length))
                    offset += length

                actual_workers = min(max_workers, len(chunks))
                with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                    futures = [
                        pool.submit(copy_chunk, src_fd, dst_fd, off, ln)
                        for off, ln in chunks
                    ]
                    for f in as_completed(futures):
                        copied = f.result()
                        if progress_cb:
                            progress_cb(copied)
            finally:
                os.close(src_fd)
                os.close(dst_fd)

        # 全部写完，rename 到最终路径（原子操作）
        os.rename(tmp_path, dst_path)

        # 保留源文件的时间戳
        st = os.stat(src_path)
        os.utime(dst_path, (st.st_atime, st.st_mtime))

    except BaseException:
        # 中断/异常时清理临时文件
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise

    return file_size


def format_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def format_speed(bps: float) -> str:
    return format_size(bps) + "/s"


class ProgressTracker:
    """线程安全的进度跟踪器"""
    def __init__(self, total_bytes, total_files):
        self.lock = threading.Lock()
        self.total_bytes = total_bytes
        self.total_files = total_files
        self.copied_bytes = 0
        self.done_files = 0
        self.skipped_files = 0
        self.skipped_bytes = 0
        self.t_start = time.time()
        self.last_report = self.t_start

    def add_bytes(self, n):
        with self.lock:
            self.copied_bytes += n
            self._maybe_report()

    def add_done(self):
        with self.lock:
            self.done_files += 1

    def add_skip(self, size):
        with self.lock:
            self.skipped_files += 1
            self.skipped_bytes += size
            self.done_files += 1
            self._maybe_report()

    def _maybe_report(self):
        now = time.time()
        if now - self.last_report < 2.0:
            return
        self.last_report = now
        elapsed = now - self.t_start
        speed = self.copied_bytes / elapsed if elapsed > 0 else 0
        progress = (self.copied_bytes + self.skipped_bytes) / self.total_bytes * 100 if self.total_bytes > 0 else 100
        print(
            f"  [{progress:5.1f}%] {self.done_files}/{self.total_files} files | "
            f"copied {format_size(self.copied_bytes)} | "
            f"skipped {self.skipped_files} | "
            f"speed {format_speed(speed)} | "
            f"elapsed {elapsed:.0f}s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="高速并行文件同步 (大文件分块优化)")
    parser.add_argument("src", help="源目录路径")
    parser.add_argument("dst", help="目标目录路径")
    parser.add_argument("--workers", type=int, default=32, help="总并发线程数 (默认 32)")
    parser.add_argument("--chunk-mb", type=int, default=64, help="单个分块大小 MB (默认 64)")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在且大小一致的文件")
    parser.add_argument("--dry-run", action="store_true", help="仅扫描不拷贝")
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    chunk_size = args.chunk_mb * 1024 * 1024

    if not os.path.isdir(src):
        print(f"[ERROR] 源目录不存在: {src}")
        return 1

    # 扫描
    print(f"[SCAN] 扫描源目录: {src}")
    t0 = time.time()
    files = scan_files(src)
    total_size = sum(s for _, s in files)
    scan_time = time.time() - t0
    print(f"[SCAN] 共 {len(files)} 个文件, 总计 {format_size(total_size)}, 扫描耗时 {scan_time:.1f}s")

    # 大文件统计
    big_files = [(r, s) for r, s in files if s > 1024 * 1024 * 1024]
    if big_files:
        big_total = sum(s for _, s in big_files)
        print(f"[SCAN] 其中 >1GB 的大文件: {len(big_files)} 个, 共 {format_size(big_total)} ({big_total/total_size*100:.1f}%)")

    if args.dry_run:
        print(f"\n[DRY-RUN] 文件列表 (按大小降序):")
        for rel, size in files[:30]:
            print(f"  {format_size(size):>12}  {rel}")
        if len(files) > 30:
            print(f"  ... 还有 {len(files)-30} 个文件")
        return 0

    # 拷贝
    os.makedirs(dst, exist_ok=True)
    print(f"[COPY] 目标目录: {dst}")
    print(f"[COPY] 并发线程: {args.workers}, 分块大小: {args.chunk_mb}MB")
    print(f"[COPY] 跳过已存在: {args.skip_existing}")
    print()

    tracker = ProgressTracker(total_size, len(files))
    failed = []

    # 策略: 大文件单独用多线程分块拷贝（串行处理每个大文件，内部并行分块）
    #       小文件用线程池并行拷贝
    BIG_THRESHOLD = 512 * 1024 * 1024  # 512MB

    big = [(r, s) for r, s in files if s >= BIG_THRESHOLD]
    small = [(r, s) for r, s in files if s < BIG_THRESHOLD]

    # 第一阶段: 大文件逐个处理，每个文件内部多线程分块
    if big:
        print(f"[PHASE 1] 拷贝 {len(big)} 个大文件 (>= 512MB), 每个文件内部 {args.workers} 线程分块并行...")
    for rel, size in big:
        src_file = os.path.join(src, rel)
        dst_file = os.path.join(dst, rel)

        # 清理上次中断残留的临时文件
        tmp_file = dst_file + ".synctmp"
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)
            print(f"    [CLEANUP] 删除残留临时文件: {os.path.basename(tmp_file)}")

        if args.skip_existing and os.path.exists(dst_file):
            if os.path.getsize(dst_file) == size:
                tracker.add_skip(size)
                continue

        fname = os.path.basename(rel)
        print(f"    >> {fname} ({format_size(size)})", flush=True)
        t_file = time.time()
        try:
            copy_file_parallel(src_file, dst_file, chunk_size, args.workers, progress_cb=tracker.add_bytes)
            tracker.add_done()
            elapsed_f = time.time() - t_file
            speed_f = size / elapsed_f if elapsed_f > 0 else 0
            print(f"    << {fname} done in {elapsed_f:.1f}s ({format_speed(speed_f)})", flush=True)
        except Exception as e:
            failed.append((rel, str(e)))
            tracker.add_done()

    # 第二阶段: 小文件并行拷贝
    if small:
        print(f"\n[PHASE 2] 拷贝 {len(small)} 个小文件 (< 512MB), {args.workers} 线程并行...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def copy_small(rel, size):
                src_file = os.path.join(src, rel)
                dst_file = os.path.join(dst, rel)
                tmp_file = dst_file + ".synctmp"
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
                if args.skip_existing and os.path.exists(dst_file):
                    if os.path.getsize(dst_file) == size:
                        return (rel, size, True)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                copy_file_parallel(src_file, dst_file, chunk_size, 1)
                return (rel, size, False)

            futures = {pool.submit(copy_small, r, s): r for r, s in small}
            for f in as_completed(futures):
                rel = futures[f]
                try:
                    _, size, skipped = f.result()
                    if skipped:
                        tracker.add_skip(size)
                    else:
                        tracker.add_bytes(size)
                        tracker.add_done()
                except Exception as e:
                    failed.append((rel, str(e)))
                    tracker.add_done()

    elapsed = time.time() - tracker.t_start
    avg_speed = tracker.copied_bytes / elapsed if elapsed > 0 else 0

    print()
    print("=" * 60)
    print(f"[DONE] 同步完成!")
    print(f"  拷贝: {tracker.done_files - tracker.skipped_files} 文件, {format_size(tracker.copied_bytes)}")
    print(f"  跳过: {tracker.skipped_files} 文件, {format_size(tracker.skipped_bytes)}")
    print(f"  失败: {len(failed)} 文件")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  平均速度: {format_speed(avg_speed)}")
    print("=" * 60)

    if failed:
        print(f"\n[WARN] 失败文件列表:")
        for rel, err in failed:
            print(f"  {rel}: {err}")

    return 1 if failed else 0


if __name__ == "__main__":
    exit(main())
