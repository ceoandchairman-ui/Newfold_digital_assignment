#!/usr/bin/env bash
# Run a command inside a PRIVATE AKS cluster from CI, through the Azure control plane.
#
#   scripts/aks_invoke.sh "<command>" [extra az args, e.g. --file .]
#   needs: RESOURCE_GROUP and AKS_NAME in the environment, `az` logged in, and `jq`.
#
# The cluster's API server has no public endpoint, so a GitHub-hosted runner cannot run kubectl or
# helm against it directly. `az aks command invoke` runs the command in a pod inside the cluster and
# returns its output. The CLI's own exit status does not always reflect the command's, so the
# command's exitCode is checked explicitly and this script fails if it is not 0.
set -euo pipefail

cmd="$1"; shift
: "${RESOURCE_GROUP:?RESOURCE_GROUP not set}" "${AKS_NAME:?AKS_NAME not set}"

out="$(az aks command invoke --resource-group "$RESOURCE_GROUP" --name "$AKS_NAME" \
        --command "$cmd" "$@" --output json)"
printf '%s\n' "$out" | jq -r '.logs // ""'

code="$(printf '%s\n' "$out" | jq -r '.exitCode // empty')"
if [ -z "$code" ]; then
  echo "aks_invoke: no exitCode in the response, treating as failure" >&2
  exit 1
fi
exit "$code"
