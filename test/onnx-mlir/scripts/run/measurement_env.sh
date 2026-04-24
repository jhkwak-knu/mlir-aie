#!/usr/bin/env bash
# measurement_env.sh - Configure system for stable NPU energy/time measurement.
#
# Usage:
#   sudo bash measurement_env.sh apply   # save snapshot + apply measurement settings
#   sudo bash measurement_env.sh revert  # restore saved snapshot
#   bash measurement_env.sh status       # print current settings (no sudo needed)
#
# All settings are volatile sysfs — a reboot restores kernel defaults.
# The snapshot file is used for same-session revert only.

set -euo pipefail

# Resolve project paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SNAPSHOT="$ROOT_DIR/out/.measurement_env_snapshot.sh"

# Target values for measurement stability
TARGET_GOVERNOR="performance"
TARGET_MAX_FREQ="3290000"
TARGET_MIN_FREQ="3290000"
TARGET_EPP="performance"
TARGET_PLATFORM_PROFILE="performance"
TARGET_NMI_WATCHDOG="0"
# C-states to disable (indices 2 and 3 = C2, C3)
CSTATE_DISABLE_INDICES=(2 3)

# --- Helper functions ---

read_sysfs() {
    local path="$1"
    if [[ -f "$path" && -r "$path" ]]; then
        cat "$path" 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

write_sysfs() {
    local path="$1" value="$2"
    if [[ -f "$path" ]]; then
        echo "$value" > "$path" 2>/dev/null || echo "[warn] failed to write $value to $path" >&2
    else
        echo "[warn] path not found: $path" >&2
    fi
}

# --- Commands ---

do_status() {
    echo "=== Measurement Environment Status ==="
    echo ""

    # Governor (sample cpu0)
    local gov
    gov=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "CPU governor:        $gov"

    # Frequency range
    local min_f max_f cur_f
    min_f=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
    max_f=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
    cur_f=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
    echo "Freq (min/max/cur):  ${min_f} / ${max_f} / ${cur_f} kHz"

    # EPP
    local epp
    epp=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)
    echo "EPP:                 $epp"

    # Platform profile
    local profile
    profile=$(read_sysfs /sys/firmware/acpi/platform_profile)
    echo "Platform profile:    $profile"

    # NMI watchdog
    local nmi
    nmi=$(read_sysfs /proc/sys/kernel/nmi_watchdog)
    echo "NMI watchdog:        $nmi"

    # C-state disable status
    echo -n "C-state disable:     "
    for idx in "${CSTATE_DISABLE_INDICES[@]}"; do
        local name dis
        name=$(read_sysfs "/sys/devices/system/cpu/cpu0/cpuidle/state${idx}/name")
        dis=$(read_sysfs "/sys/devices/system/cpu/cpu0/cpuidle/state${idx}/disable")
        echo -n "${name}=${dis} "
    done
    echo ""

    # RAPL availability
    local rapl_pkg rapl_core
    rapl_pkg=$(read_sysfs /sys/class/powercap/intel-rapl:0/energy_uj)
    rapl_core=$(read_sysfs /sys/class/powercap/intel-rapl:0:0/energy_uj)
    echo "RAPL PKG readable:   $([ "$rapl_pkg" != "N/A" ] && echo yes || echo no)"
    echo "RAPL CORE readable:  $([ "$rapl_core" != "N/A" ] && echo yes || echo no)"

    # Snapshot
    echo ""
    if [[ -f "$SNAPSHOT" ]]; then
        echo "Snapshot:            $SNAPSHOT (exists)"
    else
        echo "Snapshot:            (none)"
    fi
    echo ""
}

do_apply() {
    echo "=== Applying Measurement Environment ==="
    echo ""

    # Save snapshot of current values before changing anything
    mkdir -p "$(dirname "$SNAPSHOT")"
    {
        echo "# measurement_env snapshot — generated $(date -Iseconds)"
        echo "SNAP_GOVERNOR=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
        echo "SNAP_MAX_FREQ=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
        echo "SNAP_MIN_FREQ=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)"
        echo "SNAP_EPP=$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
        echo "SNAP_PLATFORM_PROFILE=$(read_sysfs /sys/firmware/acpi/platform_profile)"
        echo "SNAP_NMI_WATCHDOG=$(read_sysfs /proc/sys/kernel/nmi_watchdog)"
        for idx in "${CSTATE_DISABLE_INDICES[@]}"; do
            echo "SNAP_CSTATE${idx}_DISABLE=$(read_sysfs /sys/devices/system/cpu/cpu0/cpuidle/state${idx}/disable)"
        done
    } > "$SNAPSHOT"
    echo "[ok] Snapshot saved: $SNAPSHOT"

    # Apply governor (all CPUs)
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        write_sysfs "$f" "$TARGET_GOVERNOR"
    done
    echo "[ok] Governor -> $TARGET_GOVERNOR"

    # Apply max frequency first (must be >= current min to avoid EINVAL)
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
        write_sysfs "$f" "$TARGET_MAX_FREQ"
    done
    echo "[ok] Max freq -> $TARGET_MAX_FREQ kHz"

    # Apply min frequency
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
        write_sysfs "$f" "$TARGET_MIN_FREQ"
    done
    echo "[ok] Min freq -> $TARGET_MIN_FREQ kHz"

    # Apply EPP
    for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
        write_sysfs "$f" "$TARGET_EPP"
    done
    echo "[ok] EPP -> $TARGET_EPP"

    # C-state disable
    for idx in "${CSTATE_DISABLE_INDICES[@]}"; do
        for f in /sys/devices/system/cpu/cpu*/cpuidle/state${idx}/disable; do
            write_sysfs "$f" "1"
        done
    done
    echo "[ok] C-states ${CSTATE_DISABLE_INDICES[*]} disabled"

    # NMI watchdog
    write_sysfs /proc/sys/kernel/nmi_watchdog "$TARGET_NMI_WATCHDOG"
    echo "[ok] NMI watchdog -> $TARGET_NMI_WATCHDOG"

    # Platform profile
    write_sysfs /sys/firmware/acpi/platform_profile "$TARGET_PLATFORM_PROFILE"
    echo "[ok] Platform profile -> $TARGET_PLATFORM_PROFILE"

    # Grant RAPL read access (same as setup_env.sh)
    local rapl_path="/sys/class/powercap/intel-rapl:0/energy_uj"
    local rapl_core_path="/sys/class/powercap/intel-rapl:0:0/energy_uj"
    for rp in "$rapl_path" "$rapl_core_path"; do
        if [[ -f "$rp" && ! -r "$rp" ]]; then
            chmod o+r "$rp" 2>/dev/null && echo "[ok] RAPL readable: $rp" || true
        fi
    done

    echo ""
    do_status
}

do_revert() {
    echo "=== Reverting Measurement Environment ==="
    echo ""

    if [[ ! -f "$SNAPSHOT" ]]; then
        echo "[error] No snapshot found at $SNAPSHOT" >&2
        echo "[info]  Reboot to restore kernel defaults, or run 'apply' first." >&2
        exit 1
    fi

    # shellcheck source=/dev/null
    source "$SNAPSHOT"

    # Revert min frequency first (lower min before lowering max to avoid EINVAL)
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
        write_sysfs "$f" "$SNAP_MIN_FREQ"
    done
    echo "[ok] Min freq -> $SNAP_MIN_FREQ kHz"

    # Revert max frequency
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
        write_sysfs "$f" "$SNAP_MAX_FREQ"
    done
    echo "[ok] Max freq -> $SNAP_MAX_FREQ kHz"

    # Revert governor
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        write_sysfs "$f" "$SNAP_GOVERNOR"
    done
    echo "[ok] Governor -> $SNAP_GOVERNOR"

    # Revert EPP
    for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
        write_sysfs "$f" "$SNAP_EPP"
    done
    echo "[ok] EPP -> $SNAP_EPP"

    # Revert C-states
    for idx in "${CSTATE_DISABLE_INDICES[@]}"; do
        local varname="SNAP_CSTATE${idx}_DISABLE"
        local val="${!varname}"
        for f in /sys/devices/system/cpu/cpu*/cpuidle/state${idx}/disable; do
            write_sysfs "$f" "$val"
        done
    done
    echo "[ok] C-states reverted"

    # Revert NMI watchdog
    write_sysfs /proc/sys/kernel/nmi_watchdog "$SNAP_NMI_WATCHDOG"
    echo "[ok] NMI watchdog -> $SNAP_NMI_WATCHDOG"

    # Revert platform profile
    write_sysfs /sys/firmware/acpi/platform_profile "$SNAP_PLATFORM_PROFILE"
    echo "[ok] Platform profile -> $SNAP_PLATFORM_PROFILE"

    echo ""
    do_status
}

# --- Main ---

case "${1:-}" in
    apply)  do_apply  ;;
    revert) do_revert ;;
    status) do_status ;;
    *)
        echo "Usage: $0 {apply|revert|status}" >&2
        echo "  apply  - Save snapshot and apply measurement settings (requires sudo)" >&2
        echo "  revert - Restore settings from snapshot (requires sudo)" >&2
        echo "  status - Print current settings (no sudo needed)" >&2
        exit 1
        ;;
esac
