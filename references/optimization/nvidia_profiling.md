# NVIDIA GPU Profiling & Benchmarking Reference

This reference guide details system-level and kernel-level GPU profiling workflows using NVIDIA tools when benchmarking standalone PyTorch models and custom CUDA operators outside of runtime environments.

---

## ⏱️ 1. PyTorch Micro-benchmarking Rules

When benchmarking operators inside Python/PyTorch scripts, follow these practices to avoid inaccurate timings:

1. **Warmup Passes:** Always execute a minimum of 10 warmup steps to load CUDA context, compile JIT kernels, and populate memory managers before starting the timer.
2. **Device Synchronization:** PyTorch executes GPU operations asynchronously. You must synchronize the host and device before starting and ending your timer:
   ```python
   # Warmup
   for _ in range(10):
       _ = model(inputs)
   torch.cuda.synchronize()
   
   start_time = time.time()
   for _ in range(100):
       _ = model(inputs)
   torch.cuda.synchronize()
   elapsed = (time.time() - start_time) / 100
   ```
3. **Cache Flushing:** To benchmark raw DRAM access speeds without cache interference, flush the L2 cache between runs:
   ```python
   # Allocate a dummy tensor matching the L2 cache size (usually 32MB - 96MB on modern GPUs)
   dummy = torch.empty(40 * 1024 * 1024, dtype=torch.uint8, device="cuda").zero_()
   ```

---

## 📊 2. Nsight Systems (`nsys`) — Timeline & Concurrency Profiling

Nsight Systems profiles system-wide CPU-to-GPU execution timelines, kernel execution bounds, memory transfers (HtoD/DtoH), and launch latency overhead.

### A. Run Command
Execute the profiler via CLI:
```bash
nsys profile --trace=cuda,cudnn,cublas,osrt -o report_all -w true python script.py
```
* `--trace=cuda,cudnn,cublas,osrt`: Limits tracking to CUDA calls, CuDNN, CuBLAS, and OS runtime APIs.
* `-o report_all`: Saves the output files (`report_all.nsys-rep` and `report_all.sqlite`).

### B. Analyzing the Report
Open the report in the **NVIDIA Nsight Systems UI** or query the SQLite database to analyze:
1. **Host-Device Gap:** If the CPU timeline has large gaps between launching CUDA kernels, the execution is CPU-bound (likely caused by data preprocessing, loader bottlenecks, or excessive Python logic loop overhead).
2. **Memory Swapping:** Look for frequent Host-to-Device (`cudaMemcpyHtoD`) or Device-to-Host transfers occurring inside the loop timeline. These indicate memory-paging bottlenecks that should be resolved by pinning memory (`pin_memory=True`).

---

## 🎯 3. Nsight Compute (`ncu`) — Kernel-Level Deep Profiling

Nsight Compute performs detailed hardware-level profiling of individual GPU kernels, detailing memory access patterns, register pressure, occupancy, and Tensor Core utilization.

### A. Run Command
Execute on specific targets:
```bash
ncu --target-processes all -o kernel_report python script.py
```
* `-o kernel_report`: Saves the report as `kernel_report.ncu-rep`.
* **Important:** NCU introduces significant overhead. Restrict it to profile a single iteration of your target kernel.

### B. Crucial NCU Metrics
Look at these key metrics inside the Nsight Compute UI:
1. **SOL (Speed of Light):** Percentage of the hardware's maximum bandwidth or compute throughput achieved by your kernel. If Compute SOL is high, the kernel is compute-bound; if Memory SOL is high, it is memory-bound.
2. **Occupancy:** The ratio of active warps per multiprocessor to the maximum possible warps. Low occupancy indicates register spills or excessive shared memory allocation.
3. **Tensor Core (TC) Pipe:** Verifies if Tensor Cores are active. If the TC pipeline usage is 0%, your matmul operations are running on slower FP32 CUDA cores.
