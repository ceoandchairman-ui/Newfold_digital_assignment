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

field() { printf '%s\n' "$out" | jq -r ".$1 // empty"; }

# After about 5 minutes `invoke` comes back while the command is still running, with no exitCode.
# Keep asking for its result. Until it finishes, `command result` answers in PLAIN TEXT
# ("... status: Running") even with --output json.
id="$(field id)"
deadline=$((SECONDS + ${AKS_INVOKE_WAIT_S:-1200}))
code="$(field exitCode)"
while [ -z "$code" ] && [ -n "$id" ] && [ "$(field provisioningState)" = "Running" ] &&
      [ "$SECONDS" -lt "$deadline" ]; do
  echo "aks_invoke: command $id is still running" >&2
  sleep "${AKS_INVOKE_POLL_S:-20}"
  reply="$(az aks command result --resource-group "$RESOURCE_GROUP" --name "$AKS_NAME" \
            --command-id "$id" --output json)"
  case "$reply" in
    "{"*)
      out="$reply"
      printf '%s\n' "$out" | jq -r '.logs // ""'
      code="$(field exitCode)"
      ;;
    *"status: Running"*) ;;
    *)
      echo "aks_invoke: unexpected reply for command $id: $reply" >&2
      break
      ;;
  esac
done

if [ -z "$code" ]; then
  echo "aks_invoke: no exitCode in the response, treating as failure" >&2
  echo "aks_invoke: command id='$id' state='$(field provisioningState)' reason='$(field reason)'" >&2
  [ -n "$out" ] || echo "aks_invoke: az returned an empty response" >&2
  exit 1
fi
exit "$code"
