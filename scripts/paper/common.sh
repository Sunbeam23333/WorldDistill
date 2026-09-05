#!/bin/bash
set -euo pipefail

PAPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "${PAPER_SCRIPT_DIR}")")"
DRY_RUN="${DRY_RUN:-0}"

print_header() {
    local title="$1"
    echo ""
    echo "============================================================"
    echo "${title}"
    echo "============================================================"
}

print_cmd() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: required file not found: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "ERROR: required directory not found: ${path}" >&2
        exit 1
    fi
}

warn_if_template_manifest() {
    local path="$1"
    case "${path}" in
        *.template.json)
            echo "WARNING: using template manifest ${path}; replace placeholder paths before real execution." >&2
            ;;
    esac
}
