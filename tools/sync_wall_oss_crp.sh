#!/usr/bin/env bash
# Mirror the WALL-OSS-CRP source tree (Stage 1 / Stage 2 code, configs, tests, docs)
# into this repo's WALL-OSS-CRP/ so it goes to GitHub with the rest of the project.
#
# The working copy stays at $SRC (its venvs / Docker mounts use absolute paths);
# WALL-OSS-CRP/ here is a read-only mirror — edit in $SRC, then:
#   tools/sync_wall_oss_crp.sh && git add -A && git commit -m "..." && git push
#
# Left out: env/, .venv-rlt/, .cache/, wheelhouse/, artifacts/, models (symlink),
# upstream clones lerobot/ wall-x/ dmuon/ (recorded in UPSTREAM_REPOS.txt),
# binaries, and logs/ except plain-text reports.
set -euo pipefail

SRC="${WALL_OSS_CRP_SRC:-/home/crp-5070ti-01/workspaces/WALL-OSS-CRP}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DST="$REPO/WALL-OSS-CRP"

[ -d "$SRC" ] || { echo "source not found: $SRC" >&2; exit 1; }
mkdir -p "$DST"

rsync -a --delete --delete-excluded --prune-empty-dirs --no-links --max-size=20M \
  --exclude='/.cache/' --exclude='/env/' --exclude='/.venv-rlt/' --exclude='/wheelhouse/' \
  --exclude='/artifacts/' --exclude='/lerobot/' --exclude='/wall-x/' --exclude='/dmuon/' \
  --exclude='.pytest_cache/' --exclude='__pycache__/' --exclude='*.py[cod]' \
  --include='/logs/' --include='/logs/**/' \
  --include='/logs/**.json' --include='/logs/**.log' --include='/logs/**.txt' --include='/logs/**.md' \
  --include='/logs/**.py' --include='/logs/**.sh' --include='/logs/**.diff' --include='/logs/**.patch' \
  --include='/logs/**.yaml' --include='/logs/**.csv' \
  --exclude='/logs/**' \
  --exclude='*.whl' --exclude='*.deb' --exclude='*.part' --exclude='*.tar' --exclude='*.tar.gz' \
  --exclude='*.tgz' --exclude='*.so' --exclude='*.safetensors' --exclude='*.pt' --exclude='*.pth' \
  --exclude='*.ckpt' --exclude='*.bin' --exclude='*.mp4' \
  "$SRC/" "$DST/"

# H100 login credentials stay local: cut H100使用说明.md at its "## 8.账号密码" section.
f="$DST/H100使用说明.md"
if [ -f "$f" ]; then
  awk -v src="$SRC" '/^##[[:space:]]*8\.[[:space:]]*账号密码/ {
      print "## 8.账号密码"; print "";
      print "（账号、密码与 IP 分配不入库，见本机原文件 " src "/H100使用说明.md）"; exit }
    { print }' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
fi

# Upstream clones are excluded; record where they come from so the tree can be rebuilt.
{
  printf '# upstream clones excluded from the mirror (name, origin, HEAD, local changes)\n'
  for d in lerobot wall-x dmuon; do
    [ -d "$SRC/$d/.git" ] || continue
    printf '%s\t%s\t%s\tlocal_changes=%s\n' "$d" \
      "$(git -C "$SRC/$d" remote get-url origin 2>/dev/null || echo '?')" \
      "$(git -C "$SRC/$d" rev-parse HEAD)" \
      "$(git -C "$SRC/$d" status --porcelain | wc -l)"
  done
} > "$DST/UPSTREAM_REPOS.txt"

echo "synced $SRC -> $DST ($(du -sh "$DST" | cut -f1), $(find "$DST" -type f | wc -l) files)"
