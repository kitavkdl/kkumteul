#!/bin/bash
# 부모의 가르침 셋째 판 — 따름 반응(F1) 관문 −1b′ 세계 점검. 설계 6-1절, 예측은 results.md "따름 반응(F1) 관문 −1b′ 예측".
# teach_chain.sh의 세계 점검(wbranch)과 같은 틀: 바탕 g-p500-a/b 260만에서 3만 스텝 갈라 끝 5,000스텝 창을 읽는다.
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> <메모리 상한 도구> --max 1500M --bg teach-follow \
#     --log "$ROOT/runs/teach-exp/follow.log" -- bash wayfinder/stage2-4/follow_check.sh
# 갈래마다 done/json 표시가 있어 다시 불러도 이어 간다. DRY=1이면 명령만 찍는다.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
E=${OUT:-$R/runs/teach-exp}
PY=$R/.venv/bin/python
J=wayfinder/stage2-4/taste_gene_judge.py
START=ckpt_000002600000.npz
W_END=ckpt_000002630000.npz
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2} PYTHONPATH=.
mkdir -p "$E"
note() { echo "$(TZ=America/New_York date '+%m-%d %H:%M ET') $*" | tee -a "$E/progress.txt"; }
runit() { if [ -n "${DRY:-}" ]; then echo "DRY: $*"; return 0; fi; "$@"; }
cd "$C" || exit 1
base_ckpt() { echo "$R/runs/g-p500-$1/checkpoints/$START"; }
wbranch() {  # 태그 이름 설정...
  local tag=$1 name=$2 s; shift 2
  local dir=$R/runs/t-w-$name-$tag ck out=$E/w-$name-$tag
  local sets=(); for s in "$@"; do sets+=(--set "$s"); done
  if [ ! -f "$dir/done" ]; then
    note "세계 $name-$tag 시작 ($*)"
    if [ -n "${DRY:-}" ]; then echo "DRY: branch $dir ${sets[*]}"; else
      nice -n 15 "$PY" -m kkumteul.run branch --from "$(base_ckpt "$tag")" --run "$dir" --steps 30000 --quiet \
        "${sets[@]}" > "$dir.log" 2>&1 && touch "$dir/done"
    fi
  fi
  ck=$dir/checkpoints/$W_END
  [ -s "$out.json" ] && return 0
  [ -f "$ck" ] || [ -n "${DRY:-}" ] || { note "세계 $name-$tag: 체크포인트 없음 $ck"; return 0; }
  runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "${out}.log" 2>&1
  note "세계 $name-$tag 읽음"
}
G="preset=taste taste_inherit=0 taste_gene_enabled=true kind_flip_period=500 juv_enabled=true"
for t in a b; do
  wbranch $t fol05 $G juv_follow=0.5 juv_move_floor=1.0
  wbranch $t fol10 $G juv_follow=1.0 juv_move_floor=1.0
done
wbranch a fol10s03 $G juv_follow=1.0 juv_move_floor=0.3
note "fol 끝"
