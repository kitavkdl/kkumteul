#!/bin/bash
# 부모의 학습 보조 투자("가르침" 후보) 실험 체인(사용자 새 방향 3번). 설계: docs/superpowers/specs/2026-11-26-parent-teaching-design.md 셋째 판(따름 반응 FOLLOW, 기본 켬).
# 예측·문턱: wayfinder/stage2-4/results.md "부모의 가르침 — 예측"(실행 전에 적어 둔다).
# 앞 실험(사회 학습 체인 포함)이 끝난 뒤 한 번에 하나(2스레드, nice).
# 순서: 선결(사회 학습) → 세계 점검(−1) → 판정 0(용량 반응) → 관문(−2·α·β) → 진화 갈래 → 갈래별 판정.
#
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> [WAIT_FOR=<pgrep -f 패턴>] [DRY=1] \
#     <메모리 상한 도구> --max 1500M --bg teach-chain --log "$ROOT/runs/teach-exp/chain.log" -- bash wayfinder/stage2-4/teach_chain.sh
# 셋째 판은 OUT=<레포>/runs/teach-exp-f1을 꼭 준다(기본 기록 폴더 runs/teach-exp에는 둘째 판 w-*.json이 있어 세계 점검이 건너뛰어진다).
# 상한은 묶음 전체에 걸린다. 갈래·판정마다 done/json 표시가 있어 다시 불러도 이어 간다.
# DRY=1이면 명령만 찍고(표준출력) 아무것도 돌리지 않는다. 이때 기록 폴더는 임시 폴더(OUT으로 바꿀 수 있다).
#   DRY=1 EXP_OK=1이면 정보 비싼 세계(−1c 통과) 칸까지 찍는다.
# 시간 어림(서버 부하에 따라 달라짐, 추정): 세계 점검 약 20분, 판정 0 약 3시간, 진화 갈래 열두 개 약 3.5시간, 갈래별 판정 약 1.5시간.
# 관문에서 닫히면 판정 0까지 약 3.5시간으로 끝난다.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
if [ -n "${DRY:-}" ]; then
  E=${OUT:-$(mktemp -d)}
  exec 3>&1                      # DRY 줄은 명령별 리다이렉트(> 로그)와 상관없이 화면(표준출력)으로
else
  E=${OUT:-$R/runs/teach-exp}
fi
SOCE=${SOCE:-$R/runs/soc-exp}   # 사회 학습 판정 0 원자료(선결 읽기)
PY=$R/.venv/bin/python
J=wayfinder/stage2-4/taste_gene_judge.py
START=ckpt_000002600000.npz     # 바탕: 개인 학습 갈래 g-p500-a/b 260만
W_END=ckpt_000002630000.npz     # 세계 점검 갈래(3만 스텝) 끝
END=ckpt_000002900000.npz       # 진화 갈래(30만 스텝) 끝
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2} PYTHONPATH=.
# 셋째 판(설계 6-1절): 곁 머무름 장치. FOLLOW="" 이면 둘째 판 그대로. RUN_TAG는 갈래 폴더 이름 앞에 붙어 둘째 판 갈래와 안 겹치게 한다.
FOL=${FOLLOW-juv_follow=0.5 juv_move_floor=1.0}
RT=${RUN_TAG-f1-}
FOLSET=(); for s in $FOL; do FOLSET+=(--set "$s"); done
mkdir -p "$E"
logof() {  # 갈래 로그 자리. DRY면 임시 폴더로(리다이렉트는 runit보다 먼저 열려 진짜 로그를 자른다)
  if [ -n "${DRY:-}" ]; then echo "$E/$(basename "$1").log"; else echo "$1.log"; fi
}
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
        "taste_gene_enabled": True, "poison": 0.3, "taste_inherit": 0.0, "food_cap_replace": True,
        "taste_rate": 0.5, "social_enabled": False}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad:
    print("체크포인트 설정이 기대와 다름:", bad)
    sys.exit(1)
PYEOF
}

# ---- 세계 점검(판정 −1): 같은 체크포인트에서 갈라 3만 스텝, 끝에서 5,000스텝 창을 읽는다
wbranch() {  # 태그 이름 설정...   (이름: off=juv_measure만, on=미숙기 기본, exp=독 배수 3)
  local tag=$1 name=$2 s; shift 2
  local dir=$R/runs/t-w-$RT$name-$tag ck out=$E/w-$name-$tag
  local sets=(); for s in "$@"; do sets+=(--set "$s"); done
  if [ ! -f "$dir/done" ]; then
    note "세계 $name-$tag 시작 ($*)"
    runit nice -n 15 "$PY" -m kkumteul.run branch --from "$(base_ckpt "$tag")" --run "$dir" --steps 30000 --quiet \
      "${sets[@]}" > "$(logof "$dir")" 2>&1
    runit mkdir -p "$dir"; runit touch "$dir/done"
  fi
  ck=$dir/checkpoints/$W_END
  [ -s "$out.json" ] && return 0
  [ -f "$ck" ] || [ -n "${DRY:-}" ] || { note "세계 $name-$tag: 체크포인트 없음 $ck"; return 0; }
  runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1
  note "세계 $name-$tag 읽음: $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

# ---- 용량 반응(판정 0)
dose() {  # 바탕태그 이름 d [판정 옵션...]  (이름 예: kin-d0.5). 기본 옵션 뒤의 --set이 앞을 덮어쓴다
  local tag=$1 name=$2 d=$3; shift 3
  local ck out
  ck=$(base_ckpt "$tag"); out=$E/j-dose-$name-$tag
  [ -s "$out.json" ] && { note "용량 반응 $name-$tag 이미 있음"; return 0; }
  note "용량 반응 $name-$tag 시작"
  runit nice -n 15 "$PY" $J rate-zero "$ck" --gene "${GENE:-show}" --teach-d "$d" \
    --set juv_enabled=true --set soc_taste_enabled=true --set soc_taste_mode=outcome --set soc_rate=0.05 \
    --set teach_enabled=true --set teach_mode=kin "${FOLSET[@]}" "$@" --json "$out.json" > "$out.log" 2>&1
  local rc=$?
  note "용량 반응 $name-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

dose_all() {  # 바탕태그 하나의 판정 0 열아홉 칸. 중립은 첫 칸에서만(min_detectable)
  local t=$1 d
  dose "$t" kin-d0.1 0.1
  for d in 0.5 1.0; do dose "$t" kin-d$d $d --no-neutral; done
  for d in 0.1 0.5 1.0; do dose "$t" costonly-d$d $d --set teach_mode=costonly --no-neutral; done
  for d in 0.5 1.0; do dose "$t" any-d$d $d --set teach_mode=any --no-neutral; done
  dose "$t" knower-d1.0 1.0 --set teach_mode=knower --no-neutral
  dose "$t" off-d1.0 1.0 --set teach_enabled=false --no-neutral                       # 음성 대조
  dose "$t" nosoc-kin-d1.0 1.0 --soc-none --no-neutral                                 # 보고 배우기 없음(양쪽 soc_g = −soc_rate)
  for d in 0.5 1.0; do dose "$t" kin-nocost-d$d $d --set teach_cost=0 --no-neutral; done
  for d in 0.5 1.0; do dose "$t" kin-meal-d$d $d --set teach_cost_when=meal --no-neutral; done
  dose "$t" costonly-meal-d1.0 1.0 --set teach_mode=costonly --set teach_cost_when=meal --no-neutral
  GENE=feed dose "$t" feed-d1.0 1.0 --set teach_enabled=false --set feed_enabled=true --no-neutral   # 돌봄(feed_g를 d로)
  if [ "${EXP_OK:-0}" = 1 ]; then   # 관문 −1c 통과 때만(정보 비싼 세계)
    dose "$t" kin-exp-d1.0 1.0 --set juv_poison_mult=3 --no-neutral
    dose "$t" costonly-exp-d1.0 1.0 --set teach_mode=costonly --set juv_poison_mult=3 --no-neutral
  fi
}

run_branch() {  # 이름 씨앗(a=바탕 a·같은 흐름, b=바탕 b, c=바탕 a·--new-stream) 설정...
  local name=$1 seed=$2 s; shift 2
  local base=$seed; [ "$seed" = c ] && base=a
  local dir=$R/runs/t-$RT$name-$seed
  [ -f "$dir/done" ] && { note "$name-$seed 이미 끝남, 건너뜀"; return 0; }
  [ -f "$dir/run.json" ] && { note "$name-$seed 중간에 끊긴 흔적 — 건너뜀"; return 0; }
  local extra=(); [ "$seed" = c ] && extra=(--new-stream)
  local sets=(); for s in "$@"; do sets+=(--set "$s"); done
  note "$name-$seed 시작 ($*)"
  runit nice -n 15 "$PY" -m kkumteul.run branch --from "$(base_ckpt "$base")" --run "$dir" --steps 300000 --quiet \
    "${extra[@]}" "${sets[@]}" > "$(logof "$dir")" 2>&1
  local rc=$?
  runit mkdir -p "$dir"; runit touch "$dir/done"
  note "$name-$seed 끝 (종료 코드 $rc): $(tail -1 "$dir.log" 2>/dev/null | cut -c1-160)"
}

judge() {  # 갈래-씨앗 무엇(kin|costonly|any|ctrl|nocost|nosoc|obs)
  local tag=$1 what=$2
  local ck=$R/runs/t-$RT$tag/checkpoints/$END out=$E/j-$what-$tag
  [ -s "$out.json" ] && { note "판정 $what-$tag 이미 있음"; return 0; }
  [ -f "$ck" ] || [ -n "${DRY:-}" ] || { note "판정 $what-$tag: 체크포인트 없음 $ck"; return 0; }
  note "판정 $what-$tag 시작"
  local opts=()
  case $what in
    kin)      ;;
    costonly) opts=(--set teach_mode=costonly) ;;
    any)      opts=(--set teach_mode=any) ;;
    ctrl)     opts=(--set teach_enabled=false) ;;            # 음성 대조
    nocost)   opts=(--set teach_cost=0) ;;
    nosoc)    opts=(--soc-none) ;;
  esac
  if [ "$what" = obs ]; then
    runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1
  else
    runit nice -n 15 "$PY" $J rate-zero "$ck" --gene show "${opts[@]}" --json "$out.json" > "$out.log" 2>&1
  fi
  local rc=$?
  note "판정 $what-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

pre_check a && pre_check b || { note "바탕 확인 실패 — 체인 중단"; exit 1; }

# 0. 선결: 사회 학습 판정 0의 직접 지표가 작동이어야 kin·knower·any를 돈다(아니면 feed·세계 점검만)
SOC_OK=1
if [ -n "${DRY:-}" ]; then
  echo "DRY: $PY $J prereq-soc --dir $SOCE --tags a b" >&3
else
  "$PY" $J prereq-soc --dir "$SOCE" --tags a b > "$E/prereq.json" 2> "$E/prereq.log" || SOC_OK=0
fi
note "선결(사회 학습 직접 지표) SOC_OK=$SOC_OK"

# 1. 세계 점검(−1): 끔(측정만)·켬·정보 비싼 세계, 씨앗 a·b
G="preset=taste taste_inherit=0 taste_gene_enabled=true kind_flip_period=500"
for t in a b; do
  wbranch $t off  $G juv_measure=true
  wbranch $t on   $G juv_enabled=true $FOL
  wbranch $t exp  $G juv_enabled=true juv_poison_mult=3 $FOL
done
runit "$PY" $J gate-world --dir "$E" --tags a b > "$E/gate-world.json" 2> "$E/gate-world.log"
rc=$?
note "관문 −1 종료 코드 $rc (0 통과, 3 세계 못 섬, 4 곁 부족): $(tr -d '\n ' < "$E/gate-world.json" 2>/dev/null | cut -c1-200)"
if [ -z "${DRY:-}" ] || [ -z "${EXP_OK:-}" ]; then
  EXP_OK=$("$PY" -c "import json,sys; print(int(json.load(open('$E/gate-world.json')).get('exp_ok', False)))" 2>/dev/null || echo 0)
fi
export EXP_OK
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 −1 못 넘음 — 판정 0을 돌리지 않고 닫는다(설계 6절: 판 바꿈은 문서를 고쳐 쓰고 정한다)"
  exit 0
fi

# 2. 판정 0: 용량 반응(진화 전). 선결이 안 서면 feed 칸 외에는 건너뜀
for t in a b; do
  if [ "$SOC_OK" = 1 ]; then dose_all $t
  else dose $t off-d1.0 1.0 --set teach_enabled=false --no-neutral
       GENE=feed dose $t feed-d1.0 1.0 --set teach_enabled=false --set feed_enabled=true --no-neutral; fi
done
runit "$PY" $J gate-teach --dir "$E" --tags a b > "$E/gate.json" 2> "$E/gate.log"
rc=$?
note "관문(−2·α·β) 종료 코드 $rc (0 진화 갈래로 갈 만함, 3 가를 수 없음, 4 도구 깨짐): $(tr -d '\n ' < "$E/gate.json" 2>/dev/null | cut -c1-200)"
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 못 넘음 — 진화 갈래를 돌리지 않고 닫는다(가를 수 없음·도구 깨짐은 결과 문서에 그대로 적는다)"
  exit 0
fi

# 3. 진화 갈래: 같은 체크포인트에서 갈라 30만 스텝. kin·drift·off는 a·b·c, any·feed는 a·b
S="juv_enabled=true soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.05 $FOL"
for s in a b c; do
  run_branch kin   $s $G $S teach_enabled=true teach_mode=kin
  run_branch drift $s $G $S teach_enabled=true teach_mode=drift
  run_branch off   $s $G $S teach_enabled=false
done
for s in a b; do
  run_branch any  $s $G $S teach_enabled=true teach_mode=any
  run_branch feed $s $G $S teach_enabled=false feed_enabled=true
done
# 4. 판정: kin 갈래는 판정 나(kin·costonly·any·음성 대조·비용 없음·보고 배우기 없음)+단서, 나머지는 단서(observe)만
for s in a b; do
  for w in kin costonly any ctrl nocost nosoc obs; do judge kin-$s $w; done
done
for s in a b c; do
  judge drift-$s obs
  judge off-$s obs
done
judge kin-c obs
for s in a b; do judge any-$s obs; judge feed-$s obs; done
# 5. 시간이 남으면: 정보 비싼 세계(−1c 통과 때)에서 kin·drift 한 번 더
if [ "${EXP_OK:-0}" = 1 ]; then
  for s in a b; do
    run_branch kin-exp   $s $G $S teach_enabled=true teach_mode=kin juv_poison_mult=3
    run_branch drift-exp $s $G $S teach_enabled=true teach_mode=drift juv_poison_mult=3
    judge kin-exp-$s kin
    judge kin-exp-$s obs
    judge drift-exp-$s obs
  done
fi
note "체인 끝"
