#!/usr/bin/env bash

# Directory holding the prepared OSRM graphs, one subdirectory per region.
# Pass as the first argument or export OSRM_DATA_DIR.
PATH_TO_OSRM=${1:-${OSRM_DATA_DIR:-"/path/to/osrm"}}

cd "$PATH_TO_OSRM" || { echo "Directory $PATH_TO_OSRM does not exist."; exit 1; }

set -euo pipefail

OSRM_IMAGE="ghcr.io/project-osrm/osrm-backend:v6.0.0"

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

# --------------------
# Scan running containers
# --------------------
while read -r cid image; do
    cmd=$(docker inspect "$cid" \
        --format '{{join .Config.Entrypoint " "}} {{join .Config.Cmd " "}}')

    for region in "${REGIONS[@]}"; do
        if echo "$cmd" | grep -qi "$region"; then
            RUNNING["$region"]=1
        fi
    done
done < <(docker ps --format '{{.ID}} {{.Image}}' | grep osrm)

# --------------------
# Start missing containers
# --------------------
echo -e "\nSetting up OSRM routing servers..."

FAILED=0

for region in "${REGIONS[@]}"; do
    if [[ "${RUNNING[$region]}" -eq 0 ]]; then
        echo -e "\tStarting OSRM $region..."

        CONTAINER_ID=$(docker run -d \
            -p "${PORT[$region]}:5000" \
            -v "${PATH_TO_OSRM}:/data" \
            "$OSRM_IMAGE" \
            osrm-routed --algorithm mld "/data/$region/$region.osrm")

        # Give the container a moment to start (and possibly crash)
        sleep 5

        # Check if container is actually running
        if ! docker ps -q --no-trunc | grep -q "$CONTAINER_ID"; then
            echo -e "\t❌ OSRM $region failed to start. Container logs:"
            docker logs "$CONTAINER_ID" | sed 's/^/\t/'
            echo
            FAILED=1
            continue
        fi

        echo -e "\t✅ OSRM $region is running (container $CONTAINER_ID).\n"
    else
        echo -e "\t✅ OSRM $region is already running.\n"
    fi
done


if [[ $FAILED -ne 0 ]]; then
    echo "⚠️ One or more OSRM containers failed to start"
    exit 1
else
    echo "All OSRM containers are running successfully."
fi