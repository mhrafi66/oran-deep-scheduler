#!/bin/bash

set -euo pipefail


REPO="$(
    git rev-parse \
        --show-toplevel
)"

cd "${REPO}"


source \
    "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler


unset PYTHONPATH

export PYTHONPATH="${REPO}/src"


RUN_ID="rlpolicy3_$(date +%Y%m%d_%H%M%S)"

OUTPUT_ROOT="${HOME}/oran_experiments/rl_eval3"

CHECKPOINT="${HOME}/oran-deep-scheduler/experiments/checkpoints/paper_1lds_ppo_500tti.pt"

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"


if [[ ! -f "${CHECKPOINT}" ]]; then

    echo "ERROR: missing checkpoint:"
    echo "${CHECKPOINT}"

    exit 1
fi


# Refuse to submit modified tracked code.
#
# Untracked result/log files do not matter.

if [[ -n "$(
    git status \
        --porcelain \
        --untracked-files=no
)" ]]; then

    echo "ERROR: tracked files are dirty."
    echo "Commit the Set-3 scripts before submission."

    git status --short

    exit 1
fi


mkdir -p \
    "${OUTPUT_ROOT}" \
    "${RUN_DIR}" \
    output2


printf '%s\n' \
    "${RUN_ID}" \
    > "${OUTPUT_ROOT}/latest_run_id.txt"


git rev-parse HEAD \
    > "${RUN_DIR}/git_commit.txt"


git status --short \
    > "${RUN_DIR}/git_status_at_submit.txt"


printf '%s\n' \
    "${CHECKPOINT}" \
    > "${RUN_DIR}/checkpoint.txt"


printf '%s\n' \
    "30" \
    > "${RUN_DIR}/expected_tasks.txt"


JOB_RAW="$(
    sbatch \
        --parsable \
        --export="ALL,RL_EVAL3_RUN_ID=${RUN_ID},RL_EVAL3_OUTPUT_ROOT=${OUTPUT_ROOT},PPO_CHECKPOINT=${CHECKPOINT}" \
        scripts/run_rl_policy_eval3_array.slurm
)"


JOB_ID="${JOB_RAW%%;*}"


printf '%s\n' \
    "${JOB_ID}" \
    > "${RUN_DIR}/slurm_job_id.txt"


cat <<EOF

============================================================
RL POLICY EVALUATION — SET 3 SUBMITTED
============================================================

Run ID:      ${RUN_ID}
Slurm job:   ${JOB_ID}

Tasks:       30 (0-29)
Concurrency: 12

Worktree:
${REPO}

Commit:
$(git rev-parse HEAD)

Checkpoint:
${CHECKPOINT}

Output:
${RUN_DIR}

============================================================

EOF
