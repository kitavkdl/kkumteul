#!/bin/bash
# 사회 학습 실험 체인(사용자 새 방향 2번). 설계: docs/superpowers/specs/2026-11-07-social-learning-design.md 둘째 판.
# 예측·문턱: wayfinder/stage2-4/results.md "사회 학습 — 예측"(실행 전에 적어 둔다).
# 앞 실험이 끝난 뒤 한 번에 하나(2스레드, nice). 순서: 판정 0(용량 반응) → 관문 → 진화 갈래 → 갈래별 판정.
#
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> [WAIT_FOR=<pgrep -f 패턴>] [DRY=1] \
#     <메모리 상한 도구> --max 1500M --bg soc-chain --log "$ROOT/runs/soc-exp/chain.log" -- bash wayfinder/stage2-4/soc_chain.sh
# 상한은 묶음 전체에 걸린다. 갈래·판정마다 done/json 표시가 있어 다시 불러도 이어 간다.
# DRY=1이면 명령만 찍고(표준출력) 아무것도 돌리지 않는다. 이때 기록 폴더는 임시 폴더(OUT으로 바꿀 수 있다).
# 시간 어림(서버 부하에 따라 달라짐, 추정): 판정 0 약 1.5시간, 갈래 여섯 약 1.5시간, 갈래별 판정 약 1시간.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
if [ -n "${DRY:-}" ]; then
  E=${OUT:-$(mktemp -d)}
  exec 3>&1                      # DRY 줄은 명령별 리다이렉트(> 로그)와 상관없이 화면(표준출력)으로
else
  E=${OUT:-$R/runs/soc-exp}
fi
PY=$R/.venv/bin/python
J=wayfinder/stage2-4/taste_gene_judge.py
START=ckpt_000002600000.npz     # 바탕: 개인 학습 갈래 g-p500-a/b 260만
END=ckpt_000002900000.npz       # 30만 스텝 뒤
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2} PYTHONPATH=.
mkdir -p "$E"
note() { echo "$(TZ=America/New_York date '+%m-%d %H:%M ET') $*" | tee -a "$E/progress.txt"; }
runit() {  # DRY=1이면 명령만 찍는다
  if [ -n "${DRY:-}" ]; then echo "DRY: $*" >&3; return 0; fi
  "$@"
}

if [ -n "${WAIT_FOR:-}" ] && [ -z "${DRY:-}" ]; then
  while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 30; done
  note "앞 실험 끝남 확인 — 체인 시작"
fi
cd "$C" || exit 1

base_ckpt() { echo "$R/runs/g-p500-$1/checkpoints/$START"; }   # 바탕 태그 a|b

pre_check() {  # 체크포인트 설정이 설계가 기대는 세계인지(다르면 멈춘다)
  local tag=$1 ck
  ck=$(base_ckpt "$tag")
  [ -f "$ck" ] || { note "바탕 체크포인트 없음: $ck"; return 1; }
  "$PY" - "${ck%.npz}.json" <<'PYEOF' || return 1
import json, sys
c = json.load(open(sys.argv[1]))["config"]
want = {"food_kinds_enabled": True, "kind_flip_period": 500, "taste_enabled": True,
        "taste_gene_enabled": True, "poison": 0.3, "taste_inherit": 0.0}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad:
    print("체크포인트 설정이 기대와 다름:", bad)
    sys.exit(1)
PYEOF
}

dose() {  # 바탕태그 이름 d [판정 옵션...]  (이름 예: outcome-d0.25)
  local tag=$1 name=$2 d=$3; shift 3
  local ck out
  ck=$(base_ckpt "$tag"); out=$E/j-dose-$name-$tag
  [ -s "$out.json" ] && { note "용량 반응 $name-$tag 이미 있음"; return 0; }
  note "용량 반응 $name-$tag 시작"
  runit nice -n 15 "$PY" $J rate-zero "$ck" --gene soc --soc-d "$d" --set soc_taste_enabled=true --set soc_rate=0.05 \
    "$@" --json "$out.json" > "$out.log" 2>&1
  local rc=$?
  note "용량 반응 $name-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

dose_all() {  # 바탕태그 하나의 판정 0 열한 판. 중립은 첫 판에서만(min_detectable)
  local t=$1
  dose "$t" outcome-d0.05 0.05
  for d in 0.25 1.0; do dose "$t" outcome-d$d $d --no-neutral; done
  for d in 0.05 0.25 1.0; do dose "$t" blind-d$d $d --set soc_taste_mode=blind --no-neutral; done
  # 음성 대조: 뒤의 --set이 앞의 soc_taste_enabled=true를 덮어쓴다(같은 키는 나중 값이 이긴다)
  dose "$t" off-d1.0 1.0 --set soc_taste_enabled=false --no-neutral
  for d in 0.25 1.0; do dose "$t" outcome-nocost-d$d $d --set soc_cost=0 --no-neutral; done   # 비용 치우침 분리
  dose "$t" outcome-noind-d1.0 1.0 --no-individual --no-neutral                    # 개인 학습 없는 판(2×2 한 칸)
  dose "$t" blind-noind-d1.0 1.0 --no-individual --set soc_taste_mode=blind --no-neutral
}

run_branch() {  # 이름 씨앗(a=바탕 a·같은 흐름, b=바탕 b, c=바탕 a·--new-stream) 설정...
  local name=$1 seed=$2; shift 2
  local base=$seed; [ "$seed" = c ] && base=a
  local dir=$R/runs/s-$name-$seed
  [ -f "$dir/done" ] && { note "$name-$seed 이미 끝남, 건너뜀"; return 0; }
  [ -f "$dir/run.json" ] && { note "$name-$seed 중간에 끊긴 흔적 — 건너뜀"; return 0; }
  local extra=(); [ "$seed" = c ] && extra=(--new-stream)
  local s sets=(); for s in "$@"; do sets+=(--set "$s"); done
  note "$name-$seed 시작 ($*)"
  runit nice -n 15 "$PY" -m kkumteul.run branch --from "$(base_ckpt "$base")" --run "$dir" --steps 300000 --quiet \
    "${extra[@]}" "${sets[@]}" > "$dir.log" 2>&1
  local rc=$?
  runit mkdir -p "$dir"; runit touch "$dir/done"
  note "$name-$seed 끝 (종료 코드 $rc): $(tail -1 "$dir.log" 2>/dev/null | cut -c1-160)"
}

judge() {  # 갈래-씨앗 무엇(out|blind|ctrl|nocost|noind|noindb|obs)
  local tag=$1 what=$2
  local ck=$R/runs/s-$tag/checkpoints/$END out=$E/j-$what-$tag
  [ -s "$out.json" ] && { note "판정 $what-$tag 이미 있음"; return 0; }
  [ -f "$ck" ] || [ -n "${DRY:-}" ] || { note "판정 $what-$tag: 체크포인트 없음 $ck"; return 0; }
  note "판정 $what-$tag 시작"
  local opts=()
  case $what in
    out)     ;;
    blind)   opts=(--set soc_taste_mode=blind) ;;
    ctrl)    opts=(--set soc_taste_enabled=false) ;;          # 음성 대조(soc 꺼짐)
    nocost)  opts=(--set soc_cost=0) ;;
    noind)   opts=(--no-individual) ;;
    noindb)  opts=(--no-individual --set soc_taste_mode=blind) ;;
  esac
  if [ "$what" = obs ]; then
    runit nice -n 15 "$PY" $J observe "$ck" --json "$out.json" > "$out.log" 2>&1
  else
    runit nice -n 15 "$PY" $J rate-zero "$ck" --gene soc "${opts[@]}" --json "$out.json" > "$out.log" 2>&1
  fi
  local rc=$?
  note "판정 $what-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

pre_check a && pre_check b || { note "바탕 확인 실패 — 체인 중단"; exit 1; }

# 1. 판정 0: 용량 반응(진화 전)
for t in a b; do dose_all $t; done
runit "$PY" $J gate --dir "$E" --tags a b > "$E/gate.json" 2> "$E/gate.log"
rc=$?
note "관문 종료 코드 $rc (0 통과, 3 검출 불능, 4 도구 깨짐): $(tr -d '\n ' < "$E/gate.json" 2>/dev/null | cut -c1-200)"
if [ "$rc" != 0 ]; then
  note "관문 못 넘음 — 진화 갈래를 돌리지 않고 닫는다(검출 불능·도구 깨짐은 결과 문서에 그대로 적는다)"
  exit 0
fi

# 2. 진화 갈래: 같은 체크포인트에서 갈라 30만 스텝(씨앗 a·b). 설정은 체크포인트에서 오되 감사용으로 박는다
G="preset=taste taste_inherit=0 taste_gene_enabled=true kind_flip_period=500"
for s in a b; do
  run_branch on    $s $G soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.05
  run_branch inert $s $G soc_taste_enabled=true soc_taste_mode=inert   soc_rate=0.05
  run_branch off   $s $G soc_taste_enabled=false
done
# 3. 판정: 켠 갈래는 판정 나(outcome·blind·음성 대조·비용 없음·개인 학습 없음)+단서, 불활성·끔은 단서(observe)만
for s in a b; do
  for w in out blind ctrl nocost noind noindb obs; do judge on-$s $w; done
  judge inert-$s obs
  judge off-$s obs
done
# 4. 시간이 남으면: 씨앗 c, 첫걸음 위치 0.2 갈래
run_branch on    c $G soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.05
run_branch inert c $G soc_taste_enabled=true soc_taste_mode=inert   soc_rate=0.05
run_branch off   c $G soc_taste_enabled=false
for w in out blind ctrl obs; do judge on-c $w; done
judge inert-c obs
judge off-c obs
for s in a b; do
  run_branch on02 $s $G soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.2
  for w in out blind ctrl obs; do judge on02-$s $w; done
done
note "체인 끝"
