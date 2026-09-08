#!/usr/bin/env bash

# Common data directory (absolute path recommended)
DATA_DIR=${1:-"/path/to/osrm"}

set -euo pipefail

# ----------------------------
# Configuration
# ----------------------------
OSRM_IMAGE="ghcr.io/project-osrm/osrm-backend:v6.0.0"
PROFILE="/opt/car.lua"

# Regions: name -> download URL
declare -A REGIONS=(
    ["us-northeast-latest"]="https://download.geofabrik.de/north-america/us-northeast-latest.osm.pbf"
    ["us-west-latest"]="https://download.geofabrik.de/north-america/us-west-latest.osm.pbf"
    ["malaysia-singapore-brunei-latest"]="https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf"
)

# ----------------------------
# Setup
# ----------------------------
mkdir -p "${DATA_DIR}"

echo "Using data directory: ${DATA_DIR}"
echo "Using OSRM image: ${OSRM_IMAGE}"
echo

# ----------------------------
# Functions
# ----------------------------
download_if_missing() {
    local url="$1"
    local output="$2"

    if [[ -f "${output}" ]]; then
        echo "✔ ${output} already exists, skipping download"
    else
        echo "⬇ Downloading ${output}"
        mkdir -p "${DATA_DIR}/${REGION}"
        wget -O "${output}" "${url}"
    fi
}

run_osrm_step() {
    local cmd="$1"
    local file="$2"

    echo "▶ Running: ${cmd} ${file}"
    docker run --rm -t \
        -v "${DATA_DIR}:/data" \
        "${OSRM_IMAGE}" \
        ${cmd} "${file}" || {
            echo "✖ ${cmd} failed for ${file}"
            return 1
        }
}

# ----------------------------
# Main loop
# ----------------------------
for REGION in "${!REGIONS[@]}"; do
    echo "========================================"
    echo "Processing region: ${REGION}"
    echo "========================================"

    OSM_FILE="${DATA_DIR}/${REGION}/${REGION}.osm.pbf"
    OSRM_FILE="/data/${REGION}/${REGION}.osrm"

    # Download
    download_if_missing "${REGIONS[$REGION]}" "${OSM_FILE}"

    # Extract
    run_osrm_step "osrm-extract -p ${PROFILE}" "/data/${REGION}/${REGION}.osm.pbf"

    # Partition (for MLD)
    run_osrm_step "osrm-partition" "${OSRM_FILE}"

    # Customize
    run_osrm_step "osrm-customize" "${OSRM_FILE}"

    echo "✔ Finished region: ${REGION}"
    echo
done

echo "🎉 All OSRM data prepared successfully!"
