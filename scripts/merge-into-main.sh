#!/bin/bash
# 작업 가지를 main에 빨리감기로 합치고 push한다. 인자: 가지 이름(기본 worktree-night-1105c)
# 이 폴더(main)에 커밋하지 않은 변경이 있으면:
#   - 그 변경을 가지 내용에 합쳐도 가지 내용 그대로면(가지가 이미 담았다) 버리고 진행한다.
#   - 다르면 아무것도 하지 않고 무엇이 다른지 보여 준 뒤 멈춘다.
set -u
B="${1:-worktree-night-1105c}"
# 레포 폴더 안에서 부른다(가지에서 꺼내 `bash <(git show ...)`로 불러도 되게 스크립트 위치에 기대지 않는다)
cd "$(git rev-parse --show-toplevel)" || exit 1
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "멈춤: 지금 가지가 main이 아니다"; exit 1; }
git fetch -q origin "$B" || { echo "멈춤: origin/$B 를 못 받았다"; exit 1; }
T="origin/$B"
bad=0
same=()
while IFS= read -r f; do
  [ -z "$f" ] && continue
  # 로컬 변경(HEAD → 지금 파일)을 가지 내용에 3방향으로 합쳐도 가지 내용 그대로면, 가지가 이미 담은 변경이다
  found=0
  if git cat-file -e "HEAD:$f" 2>/dev/null && git cat-file -e "$T:$f" 2>/dev/null; then
    tmp=$(mktemp -d)
    git show "HEAD:$f" > "$tmp/base"; git show "$T:$f" > "$tmp/theirs"
    if git merge-file -q -p "$f" "$tmp/base" "$tmp/theirs" > "$tmp/out" 2>/dev/null && cmp -s "$tmp/out" "$tmp/theirs"; then
      found=1
    fi
    rm -rf "$tmp"
  fi
  if [ "$found" = 1 ]; then
    same+=("$f")
  else
    echo "== 가지와 다른 변경: $f"
    git diff --stat -- "$f"
    bad=1
  fi
done < <(git diff --name-only HEAD)
if [ "$bad" = 1 ]; then
  echo "멈춤: 위 파일은 가지에 없는 변경이라 버리지 않았다. Claude 세션에 이 출력을 보여 주세요."
  exit 1
fi
if [ "${#same[@]}" -gt 0 ]; then
  echo "가지가 이미 담은 변경이라 정리함: ${same[*]}"
  git checkout -- "${same[@]}"
fi
git merge --ff-only "$T" && git push && echo 합침
