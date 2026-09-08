#!/usr/bin/env bash

# Directory holding the prepared OSRM graphs, one subdirectory per region.
# Pass as the first argument or export OSRM_DATA_DIR.
PATH_TO_OSRM=${1:-${OSRM_DATA_DIR:-"/path/to/osrm"}}

cd "$PATH_TO_OSRM" || { echo "Directory $PATH_TO_OSRM does not exist."; exit 1; }

set -euo pipefail

OSRM_IMAGE="ghcr.io/project-osrm/osrm-backend:v6.0.0"
SIF_FILE="osrm-backend.sif"
LOCK_FILE="${SIF_FILE}.lock"

# Set FORCE_REFRESH_IMAGE=1 to force re-pull of image.
FORCE_REFRESH_IMAGE="${FORCE_REFRESH_IMAGE:-0}"
LOCK_WAIT_TIMEOUT_SECONDS="${LOCK_WAIT_TIMEOUT_SECONDS:-5}"
LOCK_WAIT_POLL_SECONDS="${LOCK_WAIT_POLL_SECONDS:-1}"

# --------------------
# Expected regions
# --------------------
REGIONS=(
  "us-northeast-latest"
  "us-west-latest"
  "malaysia-singapore-brunei-latest"
)

declare -A RUNNING
declare -A PORT

PORT["us-northeast-latest"]=5001
PORT["us-west-latest"]=5002
PORT["malaysia-singapore-brunei-latest"]=5003

for region in "${REGIONS[@]}"; do
    RUNNING["$region"]=0
done

is_region_process_running() {
    local region="$1"
    pgrep -af "osrm-routed" 2>/dev/null | grep -qiE "osrm-routed.*(/data/${region}/${region}\.osrm|${region}/${region}\.osrm)"
}

is_region_ready() {
    local region="$1"
    curl -sf -o /dev/null --max-time 3 "http://localhost:${PORT[$region]}/route/v1/driving/0,0;0,0"
}

all_required_regions_ready() {
    local region
    for region in "${REGIONS[@]}"; do
        if ! is_region_process_running "$region"; then
            return 1
        fi
        if ! is_region_ready "$region"; then
            return 1
        fi
    done
    return 0
}

print_running_processes() {
    echo -e "\nRunning OSRM processes:"
    pgrep -af "osrm-routed" || true
}

print_lock_help() {
    echo ""
    echo "Lock file in use for too long: $LOCK_FILE"
    echo "Another startup process may still be active, or a stale lock may remain."
    echo "What to do:"
    echo "  1) Check startup processes: pgrep -af 'start_osrm_servers_apptainer.sh|apptainer pull|osrm-routed'"
    echo "  2) If no relevant process is active, remove stale lock: rm -f '$LOCK_FILE'"
    echo "  3) Re-run this script"
}

# --------------------
# Fast path: skip lock/image setup when everything is already running
# --------------------
echo -e "\nChecking whether required OSRM servers are already running..."
if all_required_regions_ready; then
    echo "✅ All required OSRM servers are already running and ready."
    print_running_processes
    exit 0
fi

# --------------------
# Pull SIF image (cached if already exists)
# --------------------
echo -e "\nPulling OSRM Apptainer image..."

if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    echo "Acquiring OSRM image lock: $LOCK_FILE"
    if ! flock -n 9; then
        echo "OSRM image lock is currently held by another process."
        echo "Waiting up to ${LOCK_WAIT_TIMEOUT_SECONDS}s for lock (poll every ${LOCK_WAIT_POLL_SECONDS}s)..."
        waited=0
        while ! flock -n 9; do
            if all_required_regions_ready; then
                echo "✅ Required OSRM servers became ready while waiting; skipping lock wait."
                print_running_processes
                exit 0
            fi

            if (( waited >= LOCK_WAIT_TIMEOUT_SECONDS )); then
                echo "❌ Timed out waiting for OSRM image lock after ${LOCK_WAIT_TIMEOUT_SECONDS}s."
                print_lock_help
                exit 1
            fi

            sleep "$LOCK_WAIT_POLL_SECONDS"
            waited=$((waited + LOCK_WAIT_POLL_SECONDS))
        done
        echo "Acquired OSRM image lock after waiting ~${waited}s."
    fi
else
    echo "⚠️ flock not found; continuing without image lock"
fi

if [[ -f "$SIF_FILE" && "$FORCE_REFRESH_IMAGE" != "1" ]]; then
    echo "✅ OSRM image already exists: $SIF_FILE"
else
    tmp_sif="$(mktemp "${SIF_FILE}.tmp.XXXXXX")"
    trap 'rm -f "$tmp_sif"' EXIT

    if ! apptainer pull "$tmp_sif" "docker://$OSRM_IMAGE"; then
        echo "❌ Failed to pull OSRM image"
        exit 1
    fi

    mv -f "$tmp_sif" "$SIF_FILE"
    trap - EXIT
    echo "✅ OSRM image updated: $SIF_FILE"
fi

# --------------------
# Scan running OSRM processes
# --------------------
echo -e "\nScanning running OSRM processes..."

while read -r process_line; do
    for region in "${REGIONS[@]}"; do
        if echo "$process_line" | grep -qiE "osrm-routed.*(/data/${region}/${region}\.osrm|${region}/${region}\.osrm)"; then
            RUNNING["$region"]=1
        fi
    done
done < <(pgrep -af "osrm-routed" 2>/dev/null || true)

# --------------------
# Start missing OSRM processes
# --------------------
echo -e "\nSetting up OSRM routing servers..."

FAILED=0

for region in "${REGIONS[@]}"; do
    if [[ "${RUNNING[$region]}" -eq 0 ]]; then
        echo -e "\tStarting OSRM $region..."

        log_file="osrm-${region}.log"
        nohup apptainer exec \
            -B "${PATH_TO_OSRM}:/data" \
            "${SIF_FILE}" \
            osrm-routed --algorithm mld "/data/$region/$region.osrm" --port "${PORT[$region]}" \
            >"$log_file" 2>&1 &
        pid=$!

        # Wait for the process to become ready (accept HTTP requests)
        MAX_WAIT=120
        WAIT_INTERVAL=2
        elapsed=0
        ready=0

        echo -e "\tWaiting for OSRM $region to become ready (up to ${MAX_WAIT}s)..."
        while [[ $elapsed -lt $MAX_WAIT ]]; do
            # First check the process is still alive
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "\t❌ OSRM $region process died during startup. Logs:"
                sed 's/^/\t/' "$log_file" || true
                echo
                FAILED=1
                break
            fi

            # Probe the HTTP endpoint
            if curl -sf -o /dev/null --max-time 3 "http://localhost:${PORT[$region]}/route/v1/driving/0,0;0,0"; then
                ready=1
                break
            fi

            sleep $WAIT_INTERVAL
            elapsed=$((elapsed + WAIT_INTERVAL))
        done

        if [[ $FAILED -ne 0 ]]; then
            continue
        fi

        if [[ $ready -eq 1 ]]; then
            echo -e "\t✅ OSRM $region is running and ready (pid $pid, port ${PORT[$region]}, took ~${elapsed}s).\n"
        else
            echo -e "\t❌ OSRM $region started but not responding after ${MAX_WAIT}s. Logs:"
            sed 's/^/\t/' "$log_file" || true
            echo
            FAILED=1
            continue
        fi
    else
        echo -e "\t✅ OSRM $region is already running.\n"
    fi
done


if [[ $FAILED -ne 0 ]]; then
    echo "⚠️ One or more OSRM servers failed to start"
    exit 1
else
    echo "All OSRM servers are running successfully."
    print_running_processes
fi
