#!/bin/bash
# Submit a phase of this experiment with the environment it needs.
#
# The launcher's defaults are not the configuration any run here uses, and
# submitting without them has cost a run to an assertion unrelated to the work.
# Every setting a phase depends on lives in launch/<phase>.env; anything passed on
# the command line overrides it, so a one-off sweep stays a one-off.
#
#   experiments/dsv4_0731/submit.sh phase1
#   experiments/dsv4_0731/submit.sh phase2 NUM_ROLLOUT=4
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "${here}/../.." && pwd)

phase=${1:-}
if [[ -z ${phase} ]]; then
  echo "usage: $(basename "$0") <phase> [VAR=value ...]" >&2
  echo "phases: $(cd "${here}/launch" && ls *.env | sed 's/\.env$//' | tr '\n' ' ')" >&2
  exit 2
fi
shift

env_file="${here}/launch/${phase}.env"
[[ -f ${env_file} ]] || { echo "no such phase: ${phase} (${env_file})" >&2; exit 2; }

set -a
# shellcheck source=/dev/null
source "${env_file}"
set +a

for override in "$@"; do
  [[ ${override} == *=* ]] || { echo "not a VAR=value override: ${override}" >&2; exit 2; }
  export "${override?}"
done

: "${RUN_ID:=dsv4-0731-${phase}-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_ID

commit=$(git -C "${repo}" rev-parse --short HEAD)
echo "submitting ${phase} at ${commit} as ${RUN_ID}"
sbatch "${repo}/experiments/aws_dsv4_flash/run_rl.sbatch"
