#!/usr/bin/env bash
# tools/ik2/claim.sh TAG CMD...: run CMD under one line in /tmp/gpu-status (waits while another agent's CLAIM / BUSY
# line is there; the line is removed when CMD ends).  TAG starting with "build" writes a BUSY line (the whole box).
set -u
tag="$1"; shift
f=/tmp/gpu-status
while grep -E '^(CLAIM|BUSY|WAIT) ' "$f" 2>/dev/null | grep -qv 'normal use'; do sleep 20; done
case "$tag" in build*) line="BUSY 0-3 ik2-agent-$tag $(date '+%H:%M:%S %Z')";; *) line="CLAIM ik2-agent: $tag cards 2,3 $(date '+%H:%M:%S %Z')";; esac
echo "$line" >> "$f"
"$@"; rc=$?
grep -vxF "$line" "$f" > "$f.ik2.$$"; cat "$f.ik2.$$" > "$f"; rm -f "$f.ik2.$$"
exit $rc
