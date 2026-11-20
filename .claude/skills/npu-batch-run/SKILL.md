---
name: npu-batch-run
description: Use before running NPU hardware tests or measurement batches (run_tc_all.sh, make run, energy/RAPL measurement) on the Ryzen AI NPU - pre-checks, first-case validation, monitoring, and hang detection/recovery.
---

# NPU batch run protocol

1. **Pre-test environment check** -- Before running any NPU test batch:
   - Verify RAPL is accessible: `cat /sys/class/powercap/intel-rapl:0/energy_uj`
   - Verify NPU driver timeout: `sudo cat /sys/module/amdxdna/parameters/tdr_timeout_ms` (expect 60000; set via `/etc/modprobe.d/amdxdna.conf`, read-only at runtime)
   - Verify no stale `out/tc.json` or `out/build/` from previous runs
   - Close unnecessary applications (browser, IDE) to reduce measurement noise

2. **First-case validation** -- After the first test case completes:
   - Check status is PASS (not RUN_FAIL or timeout)
   - Verify `min_us` and `step_min_us` are reasonable (not -1 or extreme)
   - Verify energy fields are populated (not all -1)
   - If any issue, stop the batch and investigate before continuing

3. **Progress monitoring** -- During long batch runs:
   - Periodically check `tail -5 <result_csv>` for recent statuses
   - Watch for consecutive RUN_FAIL (may indicate driver/NPU hang)
   - If >3 consecutive failures, stop and restart NPU driver or reboot
   - Compare `log.txt` modification time vs current time to detect hangs
   - If `onnx_matmul` process is in `D` (disk sleep) state for >5 min, it is hung

4. **Hang detection and recovery** -- NPU hangs manifest as `D` (disk sleep) state:
   - Check: `pstree -p <batch_pid>` to find `onnx_matmul` PID
   - Check: `cat /proc/<pid>/status | grep State` -- `D (disk sleep)` = hung
   - NPU PCI device: `0000:c5:00.1` (PCI_ID=1022:17F0, driver=amdxdna)
   - `D` state is uninterruptible -- `kill -9`, PCI reset, driver rmmod are all ineffective
   - **Only reliable recovery is reboot**
   - After reboot, resume with `run_tc_all.sh -s <next_case>` (preserves existing CSV)
   - Prevention: ensure `tdr_timeout_ms=60000` before batch start (`setup_env.sh` verifies it)
   - Note: timeout_in_sec=60 does NOT prevent all hangs -- some D states bypass driver timeout (observed with amdxdna 2.20)

5. **Autonomous batch execution** -- When running batches via Claude Code:
   - **Before start**: verify RAPL readable, NPU timeout set, no stale artifacts
   - **First-case validation**: wait for case #1 to finish, check all fields per item 2
   - **Periodic monitoring**: check progress every 10-15 min via background commands
   - **Report**: alert user immediately on any FAIL, RUN_FAIL, or hang detection

