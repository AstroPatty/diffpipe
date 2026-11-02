#!/usr/bin/env bash
# Sync .github/workflows/mockgen_opencosmo.yaml with the upstream diffsky
# workflow "mockgen_opencosmo".
#
# The local workflow is an adapted copy of the upstream one (it tests diffpipe
# rather than diffsky, uploads artifacts for the property tests, ...). To pull
# in upstream changes without losing those adaptations we do a three-way merge:
#
#   ours   = .github/workflows/mockgen_opencosmo.yaml  (local, adapted)
#   base   = .github/upstream/mockgen_opencosmo.yaml        (upstream as last synced)
#   theirs = upstream workflow at the current tip of diffsky main
#
# On success the local workflow holds the merged result and the base file is
# replaced with the new upstream copy. Requires: git, gh (authenticated via
# GH_TOKEN), curl.
#
# Writes to $GITHUB_OUTPUT (if set): changed=true|false, conflicts=<n>, sha=<sha>
set -euo pipefail

UPSTREAM_REPO="${UPSTREAM_REPO:-ArgonneCPAC/diffsky}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
UPSTREAM_PATH=".github/workflows/mockgen_opencosmo.yaml"

LOCAL=".github/workflows/mockgen_opencosmo.yaml"
BASE=".github/upstream/mockgen_opencosmo.yaml"
BASE_SHA=".github/upstream/mockgen_opencosmo.sha"

out() { [ -n "${GITHUB_OUTPUT:-}" ] && echo "$1" >> "$GITHUB_OUTPUT" || true; }

# Pin to a commit so the fetched content and the recorded sha agree.
sha=$(gh api "repos/${UPSTREAM_REPO}/commits/${UPSTREAM_BRANCH}" --jq .sha)
new=$(mktemp)
merged=$(mktemp)
trap 'rm -f "$new" "$merged"' EXIT
curl -fsSL "https://raw.githubusercontent.com/${UPSTREAM_REPO}/${sha}/${UPSTREAM_PATH}" -o "$new"

out "sha=${sha}"

if cmp -s "$BASE" "$new"; then
  echo "Upstream workflow unchanged since last sync."
  out "changed=false"
  out "conflicts=0"
  exit 0
fi

# git merge-file exits with the number of conflicts (>0), or a value >=128 on error.
cp "$LOCAL" "$merged"
set +e
git merge-file \
  -L "local (${LOCAL})" \
  -L "upstream (previously synced)" \
  -L "upstream (${UPSTREAM_REPO}@${sha:0:7})" \
  "$merged" "$BASE" "$new"
rc=$?
set -e
if [ "$rc" -ge 128 ]; then
  echo "git merge-file failed (exit $rc)" >&2
  exit 1
fi

cp "$merged" "$LOCAL"
cp "$new" "$BASE"
echo "$sha" > "$BASE_SHA"

echo "Merged upstream ${sha} into ${LOCAL} with ${rc} conflict(s)."
out "changed=true"
out "conflicts=${rc}"
