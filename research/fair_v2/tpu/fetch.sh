#!/bin/bash
# fetch.sh <ip> <remote_path> <local_path>: copy with md5 verification and retries (internal network corrupts packets).
ip=$1; src=$2; dst=$3
S="ssh -n -F /dev/null -o StrictHostKeyChecking=no -i $HOME/.ssh/sky-cluster-key"
want=$($S $ip "md5sum $src" 2>/dev/null | cut -c1-32)
[ -z "$want" ] && { echo "FAIL no remote md5 for $ip:$src"; exit 1; }
mkdir -p "$(dirname "$dst")"
for t in $(seq 1 30); do
  if [ "$ip" = "10.202.0.134" ]; then cp "$src" "$dst.part"; else $S $ip "cat $src" > "$dst.part" 2>/dev/null; fi
  got=$(md5sum "$dst.part" | cut -c1-32)
  if [ "$got" = "$want" ]; then mv "$dst.part" "$dst"; echo "ok $ip:$src try $t"; exit 0; fi
done
echo "FAIL $ip:$src"; exit 1
