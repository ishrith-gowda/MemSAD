#!/usr/bin/env bash
# deploy the memsad gradio demo to hugging face spaces.
#
# usage:
#   ./scripts/deploy_hf_space.sh <hf_username>
#   e.g.: ./scripts/deploy_hf_space.sh ishrith-gowda
#
# prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
#
# the script creates a temporary deployment directory, copies the necessary
# source files, and pushes to the hf space repo.
#
# all comments are lowercase.

set -euo pipefail

HF_USER="${1:?usage: $0 <hf_username>}"
SPACE_NAME="memsad-demo"
REPO_ID="${HF_USER}/${SPACE_NAME}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR=$(mktemp -d)

echo "deploying to hf space: ${REPO_ID}"
echo "staging directory: ${DEPLOY_DIR}"

# copy source modules needed by the app
mkdir -p "${DEPLOY_DIR}/data"
mkdir -p "${DEPLOY_DIR}/defenses"
mkdir -p "${DEPLOY_DIR}/evaluation"
mkdir -p "${DEPLOY_DIR}/memory_systems"
mkdir -p "${DEPLOY_DIR}/attacks"
mkdir -p "${DEPLOY_DIR}/utils"
mkdir -p "${DEPLOY_DIR}/watermark"

# core app
cp "${PROJECT_ROOT}/src/app/app.py" "${DEPLOY_DIR}/app.py"

# data modules
cp "${PROJECT_ROOT}/src/data/__init__.py" "${DEPLOY_DIR}/data/"
cp "${PROJECT_ROOT}/src/data/synthetic_corpus.py" "${DEPLOY_DIR}/data/"
cp "${PROJECT_ROOT}/src/data/corpus_extended.py" "${DEPLOY_DIR}/data/"
cp "${PROJECT_ROOT}/src/data/nq_subset.py" "${DEPLOY_DIR}/data/" 2>/dev/null || true

# defenses
cp "${PROJECT_ROOT}/src/defenses/__init__.py" "${DEPLOY_DIR}/defenses/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/defenses/__init__.py"
cp "${PROJECT_ROOT}/src/defenses/semantic_anomaly.py" "${DEPLOY_DIR}/defenses/"
cp "${PROJECT_ROOT}/src/defenses/base.py" "${DEPLOY_DIR}/defenses/" 2>/dev/null || true

# evaluation
cp "${PROJECT_ROOT}/src/evaluation/__init__.py" "${DEPLOY_DIR}/evaluation/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/evaluation/__init__.py"
cp "${PROJECT_ROOT}/src/evaluation/retrieval_sim.py" "${DEPLOY_DIR}/evaluation/"
cp "${PROJECT_ROOT}/src/evaluation/benchmarking.py" "${DEPLOY_DIR}/evaluation/"

# memory systems
cp "${PROJECT_ROOT}/src/memory_systems/__init__.py" "${DEPLOY_DIR}/memory_systems/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/memory_systems/__init__.py"
cp "${PROJECT_ROOT}/src/memory_systems/vector_store.py" "${DEPLOY_DIR}/memory_systems/"
cp "${PROJECT_ROOT}/src/memory_systems/base.py" "${DEPLOY_DIR}/memory_systems/" 2>/dev/null || true
cp "${PROJECT_ROOT}/src/memory_systems/wrappers.py" "${DEPLOY_DIR}/memory_systems/" 2>/dev/null || true

# attacks (needed by evaluation imports)
cp "${PROJECT_ROOT}/src/attacks/__init__.py" "${DEPLOY_DIR}/attacks/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/attacks/__init__.py"
cp "${PROJECT_ROOT}/src/attacks/base.py" "${DEPLOY_DIR}/attacks/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/attacks/base.py"
cp "${PROJECT_ROOT}/src/attacks/implementations.py" "${DEPLOY_DIR}/attacks/" 2>/dev/null || true
mkdir -p "${DEPLOY_DIR}/attacks/trigger_optimization"
cp "${PROJECT_ROOT}/src/attacks/trigger_optimization/__init__.py" "${DEPLOY_DIR}/attacks/trigger_optimization/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/attacks/trigger_optimization/__init__.py"
cp "${PROJECT_ROOT}/src/attacks/trigger_optimization/optimizer.py" "${DEPLOY_DIR}/attacks/trigger_optimization/" 2>/dev/null || true

# utils
cp "${PROJECT_ROOT}/src/utils/__init__.py" "${DEPLOY_DIR}/utils/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/utils/__init__.py"
cp "${PROJECT_ROOT}/src/utils/logging.py" "${DEPLOY_DIR}/utils/"
cp "${PROJECT_ROOT}/src/utils/config.py" "${DEPLOY_DIR}/utils/" 2>/dev/null || true

# watermark (needed by defense imports)
cp "${PROJECT_ROOT}/src/watermark/__init__.py" "${DEPLOY_DIR}/watermark/" 2>/dev/null || echo "" > "${DEPLOY_DIR}/watermark/__init__.py"
cp "${PROJECT_ROOT}/src/watermark/watermarking.py" "${DEPLOY_DIR}/watermark/" 2>/dev/null || true

# configs (needed by some imports)
if [ -d "${PROJECT_ROOT}/configs" ]; then
    cp -r "${PROJECT_ROOT}/configs" "${DEPLOY_DIR}/configs"
fi

# hf space metadata
cp "${PROJECT_ROOT}/src/app/README.md" "${DEPLOY_DIR}/README.md"
cp "${PROJECT_ROOT}/src/app/requirements.txt" "${DEPLOY_DIR}/requirements.txt"

# fix the app.py path setup for flat deployment
# (in the space, modules are at the same level as app.py)
sed -i.bak 's|_src_dir = Path(__file__).resolve().parent.parent|_src_dir = Path(__file__).resolve().parent|' "${DEPLOY_DIR}/app.py"
rm -f "${DEPLOY_DIR}/app.py.bak"

echo "staged files:"
find "${DEPLOY_DIR}" -type f | sort

# push to hf space
echo ""
echo "pushing to hugging face space: ${REPO_ID}"
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('${REPO_ID}', repo_type='space', space_sdk='gradio', exist_ok=True)
api.upload_folder(
    folder_path='${DEPLOY_DIR}',
    repo_id='${REPO_ID}',
    repo_type='space',
)
print(f'deployed to: https://huggingface.co/spaces/${REPO_ID}')
"

# cleanup
rm -rf "${DEPLOY_DIR}"
echo "done."
