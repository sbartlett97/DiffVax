#!/usr/bin/env bash
# Wait for the sweep (edit the until-condition for other jobs),
# then pull every result off the pod, verify, and only then delete the pod.
set -u
SP=/private/tmp/claude-501/-Users-samuelbartlett-PycharmProjects-DiffVax/1f825b0d-4b40-4f27-87cd-4be5c061d4e0/scratchpad
FLAG=/nonexistent
POD=47xn19nefpf8ld
KEY=$HOME/.ssh/id_ed25519_runpod_2
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=20 -p 10337 root@154.54.102.49"
DEST=/Users/samuelbartlett/PycharmProjects/DiffVax/outputs/pod_2026-11-11
mkdir -p "$DEST"
echo "$(date) waiting for sweep"
until $SSH 'grep -q ALL_SWEEPS_DONE /workspace/sweep.log || ! pgrep -f "^bash /workspace/sweep.sh" >/dev/null'; do sleep 60; done
echo "$(date) sweep finished; wrapping up now"
echo "$(date) pulling results"
RS="rsync -az -e \"ssh -i $KEY -o IdentitiesOnly=yes -p 10337\""
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -p 10337" root@154.54.102.49:/workspace/outputs/ "$DEST/outputs/" ; r1=$?
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -p 10337" --include='*.log' --include='*.txt' --include='*.sh' --exclude='*' root@154.54.102.49:/workspace/ "$DEST/logs/" ; r2=$?
remote=$($SSH 'find /workspace/outputs -type f | wc -l')
local_n=$(find "$DEST/outputs" -type f | wc -l)
echo "$(date) rsync exit $r1/$r2; files remote=$remote local=$local_n"
if [ "$r1" -ne 0 ] || [ "$r2" -ne 0 ] || [ "$local_n" -lt "$remote" ]; then
  echo "$(date) VERIFY FAILED - NOT deleting pod"; exit 1
fi
url=$(python3 -c "import re;print(re.search(r'^DISCORD_WEBHOOK\s*=\s*(.+)$',open('/Users/samuelbartlett/PycharmProjects/DiffVax/.env').read(),re.M).group(1).strip().strip('\"\''))" 2>/dev/null)
runpodctl pod delete $POD && echo "$(date) pod deleted"
[ -n "$url" ] && curl -s -o /dev/null -H 'Content-Type: application/json' -d "{\"content\":\"**diffvax** results pulled to outputs/pod_2026-11-11 ($local_n files) and pod $POD deleted.\"}" "$url"
echo "$(date) done"
