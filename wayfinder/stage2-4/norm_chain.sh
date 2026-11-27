#!/bin/bash
# 규범의 씨앗(가로채기·벌주기) 실험 체인. 설계: docs/superpowers/specs/2026-11-26-norms-seed-design.md 둘째 판.
# 예측·문턱: wayfinder/stage2-4/results.md "규범의 씨앗 — 예측"(실행 전에 적어 둔다. 없으면 이 스크립트는 멈춘다).
# 앞 실험(사회 학습·가르침 체인)이 끝난 뒤 한 번에 하나(2스레드, nice). WAIT_FOR 기본값이 그 체인들을 기다린다(비우려면 WAIT_FOR=).
# 순서: 바탕 확인 → 관문 −2b(문턱) → 세계 점검(−1: take·inert·full·noimp + 유병률 맞춘 sig·none) → 판정 0(핵심·δ 칸 먼저)
#       → 관문(α·β·γ·δ·−2) → 나머지 칸 → 진화 갈래 → 판정 나·다·라·단서.
#
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> [BASE=<바탕 체크포인트>] [WAIT_FOR=<pgrep -f 패턴>] [DRY=1] [OUT=<기록 폴더>] \
#     [TAU=50] [IMP=0.015] <메모리 상한 도구> --max 1500M --bg norm-chain --log "$ROOT/runs/norm-exp/chain.log" -- bash wayfinder/stage2-4/norm_chain.sh
# 상한은 묶음 전체에 걸린다. 같은 기록 폴더에서 두 번 띄우면 flock이 막는다.
# 다시 불러도 이어 가는 범위(정직하게): 칸·판정은 json이 있으면 건너뛰고 없는 것만 돈다. 갈래(세계 점검·진화)는 done 표시가 있으면 건너뛴다.
#   done은 **종료 코드 0이고 끝 체크포인트가 있을 때만** 만든다. 도는 중에 끊겨 run.json만 남은 갈래는 자동으로 이어 돌리지 않는다 —
#   note에 "끊긴 흔적"을 남기고 건너뛴다(폴더를 지우고 다시 부르거나, python -m kkumteul.run resume으로 이어 간 뒤 done을 손으로 만든다).
# 관문 −1에서 τ·norm_impulse를 조정해 다시 돌릴 때는 TAU=25 IMP=...로 **OUT을 다른 폴더**로 불러 같은 폴더의 done·json과 안 섞는다
#   (조정은 결과 문서에 먼저 적고 한다).
# DRY=1이면 명령만 찍고(표준출력) 아무것도 돌리지 않는다. 이때 기록 폴더는 임시 폴더(OUT으로 바꿀 수 있다)이고 갈래 로그도 거기에 쓴다.
# 시간 어림(2026-11-26 짧게 잼, 2스레드·nice 15, 다른 실행과 함께 돈 서버라 흔들린다): 바탕에서 가른 갈래(약 1,300마리)
#   3,000스텝 187초(스텝당 약 0.06초, 300스텝만은 약 0.01초) → 세계 점검 갈래(3만) 약 30분, 진화 갈래(30만) 약 5시간.
#   겨루기 한 판(800마리 × 1,000스텝) 34초 → 8,000스텝 약 4분, 칸 하나(반복 6) 약 25분(base는 중립 6판을 더해 약 50분).
#   세계 점검 열두 갈래 + 관찰 ≈ 7시간, 판정 0 CORE + DELTA(18칸 × 표준·섞은) ≈ 15시간, REST ≈ 6시간, 진화 열여섯 갈래 ≈ 80시간
#   (기울기 시작판 여덟 갈래 + 40시간), 판정 나(block-compete, 끝 무리 하나에 겨루기 8종 × 반복 6 ≈ 3시간) 여덟 번 ≈ 26시간.
#   최대 메모리는 프로세스 하나 약 0.5GB(갈래·겨루기 모두). 관문에서 닫히면 그 앞까지만 돈다.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
if [ -n "${DRY:-}" ]; then
  E=${OUT:-$(mktemp -d)}
  exec 3>&1                      # DRY 줄은 명령별 리다이렉트(> 로그)와 상관없이 화면(표준출력)으로
else
  E=${OUT:-$R/runs/norm-exp}
fi
PY=$R/.venv/bin/python
J=wayfinder/stage2-4/norm_judge.py
BASE=${BASE:-$R/runs/world4/checkpoints/ckpt_000089611957.npz}   # 바탕 B1(마음·사회·판, 공격 선택압 없음)
TAU=${TAU:-50}
IMP=${IMP:-0.015}
bn=$(basename "$BASE" .npz); S0=$((10#${bn#ckpt_}))
W_END=$(printf 'ckpt_%012d.npz' $((S0 + 30000)))     # 세계 점검 갈래(3만 스텝) 끝
END=$(printf 'ckpt_%012d.npz' $((S0 + 300000)))      # 진화 갈래(30만 스텝) 끝
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2} PYTHONPATH=.
mkdir -p "$E"
note() { echo "$(TZ=America/New_York date '+%m-%d %H:%M ET') $*" | tee -a "$E/progress.txt"; }
runit() {  # DRY=1이면 명령만 찍는다
  if [ -n "${DRY:-}" ]; then echo "DRY: $*" >&3; return 0; fi
  "$@"
}
logf() {   # 갈래 로그 경로. DRY에서는 진짜 로그를 자르지 않게 임시 기록 폴더로 돌린다
  if [ -n "${DRY:-}" ]; then echo "$E/dry-$(basename "$1")"; else echo "$1"; fi
}

if [ -z "${DRY:-}" ]; then
  exec 9>"$E/.lock"
  flock -n 9 || { echo "같은 기록 폴더($E)에서 이미 체인이 돈다"; exit 1; }
fi
WAIT_FOR=${WAIT_FOR-'soc_chain.sh|teach_chain.sh'}    # 기본: 앞의 두 체인. (kkumteul.run을 기다리면 상시 도는 세계에 영영 걸린다)
if [ -n "$WAIT_FOR" ] && [ -z "${DRY:-}" ]; then
  while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 30; done
  note "앞 실험 끝남 확인 — 체인 시작"
fi
cd "$C" || exit 1
grep -q "규범의 씨앗 — 예측" wayfinder/stage2-4/results.md || { note "예측 절이 results.md에 없음 — 체인 중단"; exit 1; }

pre_check() {  # 바탕 체크포인트 설정이 설계가 기대는 세계인지(다르면 멈춘다)
  [ -f "$BASE" ] || { note "바탕 체크포인트 없음: $BASE"; return 1; }
  "$PY" - "$BASE" <<'PYEOF' || return 1
import sys
from pathlib import Path
from kkumteul import checkpoint as ckpt
c = ckpt.read_config(Path(sys.argv[1])).to_dict()   # 옛 json에 없는 키는 Config 기본값(그냥 json을 읽으면 None이라 False와 다르다)
want = {"mind_enabled": True, "social_enabled": True, "niche_enabled": True, "conflict_enabled": False,
        "vision_sectors": 8, "n_spare_in": 13, "carry_enabled": False, "taste_enabled": False,
        "soc_taste_enabled": False}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad:
    print("체크포인트 설정이 기대와 다름:", bad)
    sys.exit(1)
PYEOF
}

# 모든 갈래·점검에 같은 설정을 박는다(soc_*·taste_*는 갈래마다 섞지 않는다: 격자 재구성이 attack 이웃을 바꾼다)
G="preset=conflict preset=norm_size norm_impulse=$IMP norm_mark_tau=$TAU"
NS=(--set "norm_impulse=$IMP" --set "norm_mark_tau=$TAU")   # 마당 명령(hand-dose·arena-check·calibrate)에도 같은 값

# ---- 관문 −2b(진화 시작점의 공격 문턱): 시뮬레이션 없이 저장된 h로 문턱 위 개체 비율을 잰다
if [ ! -s "$E/threshold.json" ]; then
  runit "$PY" $J threshold "$BASE" --json "$E/threshold.json" > "$E/threshold.log" 2>&1
  note "관문 −2b: $(tail -1 "$E/threshold.log" 2>/dev/null | cut -c1-160)"
fi
# 문턱 위가 1% 아래면(threshold.json의 slope_start) 진화 갈래에 기울기 시작판(공격 편향 b0 = −0.3) 넷을 더한다(아래 3번)
SLOPE=${SLOPE:-$("$PY" -c "import json; print(int(json.load(open('$E/threshold.json')).get('slope_start', 0)))" 2>/dev/null || echo 0)}

# ---- 세계 점검(판정 −1): 같은 체크포인트에서 갈라 3만 스텝, 끝에서 5,000스텝 창을 읽는다(씨앗 a·b)
wbranch() {  # 씨앗 이름 설정...   (이름: meas=끔+계수만, inert, full=take 원천, noimp=위반 공급 끔, sig·none=유병률 맞춘 임의 표지)
  local seed=$1 name=$2; shift 2
  local dir=$R/runs/n-w-$name-$seed ck out=$E/w-$name-$seed rc
  local extra=(); [ "$seed" = b ] && extra=(--new-stream)
  local kv sets=(); for kv in "$@"; do sets+=(--set "$kv"); done   # 반복 변수를 local로(호출자의 s를 덮어쓰지 않게)
  ck=$dir/checkpoints/$W_END
  if [ ! -f "$dir/done" ]; then
    if [ -f "$dir/run.json" ]; then
      note "세계 $name-$seed: 끊긴 흔적(run.json 있고 done 없음) — 건너뜀. 폴더를 지우고 다시 부르거나 resume으로 이어 간 뒤 done을 만든다"
      return 0
    fi
    note "세계 $name-$seed 시작 ($*)"
    runit nice -n 15 "$PY" -m kkumteul.run branch --from "$BASE" --run "$dir" --steps 30000 --quiet \
      --reseed-attack-row "${extra[@]}" "${sets[@]}" > "$(logf "$dir.log")" 2>&1
    rc=$?
    if [ -z "${DRY:-}" ] && { [ "$rc" != 0 ] || [ ! -f "$ck" ]; }; then
      note "세계 $name-$seed 실패(종료 코드 $rc 또는 끝 체크포인트 없음) — done을 안 만든다"
      return 0
    fi
    runit touch "$dir/done"
  fi
  [ -s "$out.json" ] && return 0
  [ -f "$ck" ] || { note "세계 $name-$seed: 체크포인트 없음 $ck"; return 0; }
  runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1
  note "세계 $name-$seed 읽음: $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

# ---- 판정 0: 용량 반응(진화 없이 겨루기만). 칸 정의는 norm_judge.CELLS에 있다
CORE=(base hand-phi0.1 hand-phi0.3 hand-phi1.0 grp-1 grp-2 grp-3 hand3-phi0.3 hand-phi0.3-inert hand-phi0.3-ret0
      hand-phi0.3-ret0.5 hand-phi0.3-L1 off noimp)                                  # 열네 칸
DELTA=(hand-phi0.3-sig-ret0 hand-phi0.3-sig-ret0.1 hand-phi0.3-none-q-ret0 hand-phi0.3-none-q-ret0.1)  # 관문 δ가 읽는 칸: 관문 전에
REST=(hand-phi0.3-nearest hand-phi0.3-inh0 hand-phi0.3-inh0.4 hand-phi0.3-L10 hand-phi0.3-tau1
      hand-phi0.3-noise0.0005 hand-phi0.3-noise0.002)                               # 나머지 일곱 칸: 관문 뒤에
dose() {  # 칸 마당(std|mix)
  local cell=$1 arena=$2 out=$E/h-$1-$2
  [ -s "$out.json" ] && { note "용량 반응 $cell-$arena 이미 있음"; return 0; }
  note "용량 반응 $cell-$arena 시작"
  runit nice -n 15 "$PY" $J hand-dose "$BASE" --cell "$cell" --arena "$arena" --n 800 --steps 8000 --reps 6 \
    --calib "$E/calib.json" "${NS[@]}" --json "$out.json" > "$out.log" 2>&1
  local rc=$?
  note "용량 반응 $cell-$arena 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}
dose_list() {  # 칸 이름들
  local c
  for c in "$@"; do dose "$c" std; dose "$c" mix; done
}

run_branch() {  # 이름 씨앗(a=같은 흐름, b=--new-stream) 설정...   (환경변수 ATTACK_B0가 있으면 공격 편향 b0를 그 값으로 다시 뽑는다)
  local name=$1 seed=$2; shift 2
  local dir=$R/runs/n-$name-$seed rc
  [ -f "$dir/done" ] && { note "$name-$seed 이미 끝남, 건너뜀"; return 0; }
  [ -f "$dir/run.json" ] && { note "$name-$seed 끊긴 흔적(run.json 있고 done 없음) — 건너뜀. 폴더를 지우고 다시 부르거나 resume으로 이어 간 뒤 done을 만든다"; return 0; }
  local extra=(); [ "$seed" = b ] && extra=(--new-stream)
  local b0=(); [ -n "${ATTACK_B0:-}" ] && b0=(--attack-b0 "$ATTACK_B0")
  local kv sets=(); for kv in "$@"; do sets+=(--set "$kv"); done
  note "$name-$seed 시작 ($*)"
  runit nice -n 15 "$PY" -m kkumteul.run branch --from "$BASE" --run "$dir" --steps 300000 --quiet \
    --reseed-attack-row "${b0[@]}" "${extra[@]}" "${sets[@]}" > "$(logf "$dir.log")" 2>&1
  rc=$?
  if [ -z "${DRY:-}" ] && { [ "$rc" != 0 ] || [ ! -f "$dir/checkpoints/$END" ]; }; then
    note "$name-$seed 실패(종료 코드 $rc 또는 끝 체크포인트 없음) — done을 안 만든다"
    return 0
  fi
  runit touch "$dir/done"
  note "$name-$seed 끝 (종료 코드 $rc): $(tail -1 "$(logf "$dir.log")" 2>/dev/null | cut -c1-160)"
}

judge() {  # 갈래-씨앗 무엇(block|bottom|obs|knock-norm|knock-vis)
  local tag=$1 what=$2
  local ck=$R/runs/n-$tag/checkpoints/$END out=$E/j-$what-$tag
  [ -s "$out.json" ] && { note "판정 $what-$tag 이미 있음"; return 0; }
  [ -f "$ck" ] || { note "판정 $what-$tag: 체크포인트 없음 $ck"; return 0; }
  note "판정 $what-$tag 시작"
  case $what in
    block)      runit nice -n 15 "$PY" $J block-compete "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
    bottom)     runit nice -n 15 "$PY" $J bottom "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
    obs)        runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1 ;;
    knock-norm) runit nice -n 15 "$PY" -m kkumteul.knockout split "$ck" --channel norm > "$out.json" 2> "$out.log" ;;
    knock-vis)  runit nice -n 15 "$PY" -m kkumteul.knockout split "$ck" --channel vision > "$out.json" 2> "$out.log" ;;
  esac
  local rc=$?
  note "판정 $what-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

pre_check || { note "바탕 확인 실패 — 체인 중단"; exit 1; }

# 1. 세계 점검(−1). 먼저 take 원천 넷(meas·inert·full·noimp)
for t in a b; do
  wbranch $t meas  $G norm_measure=true
  wbranch $t inert $G norm_enabled=true norm_mode=inert
  wbranch $t full  $G norm_enabled=true
  wbranch $t noimp $G norm_enabled=true norm_impulse=0
done
# 1b. 표지 원천 유병률 맞추기(설계 7절): take(full) 세계의 유병률을 목표로 sig 임계와 none의 잡음 확률을 정하고, 그 값으로 두 세계를 점검한다
if [ ! -s "$E/calib.json" ]; then
  runit nice -n 15 "$PY" $J calibrate --base "$BASE" --from "$E/w-full-a.json" --out "$E/calib.json" "${NS[@]}" > "$E/calib.log" 2>&1
  note "유병률 맞춤(calib.json): $(tr -d '\n ' 2>/dev/null < "$E/calib.json" | cut -c1-200)"
fi
SIG_THETA=$("$PY" -c "import json; print(json.load(open('$E/calib.json'))['sig_theta'])" 2>/dev/null || echo 0.5)
NOISE_Q=$("$PY" -c "import json; print(json.load(open('$E/calib.json'))['noise_q'])" 2>/dev/null || echo 0.0016)
for t in a b; do
  wbranch $t sig  $G norm_enabled=true norm_mark_src=sig norm_sig_theta=$SIG_THETA norm_pun_return=0
  wbranch $t none $G norm_enabled=true norm_mark_src=none norm_mark_noise=$NOISE_Q norm_pun_return=0
done
runit nice -n 15 "$PY" $J arena-check "$BASE" "${NS[@]}" > "$E/arena-check.json" 2> "$E/arena-check.log"
runit "$PY" $J gate-world --dir "$E" --tags a b > "$E/gate-world.json" 2> "$E/gate-world.log"
rc=$?
note "관문 −1 종료 코드 $rc (0 통과, 3 세계 못 섬, 4 딜레마 크기 밖): $(tr -d '\n ' 2>/dev/null < "$E/gate-world.json" | cut -c1-200)"
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 −1 못 넘음 — 판정 0을 돌리지 않고 닫는다(τ·IMP 조정은 결과 문서를 고쳐 쓰고 TAU·IMP를 바꿔 OUT을 새 폴더로 다시 부른다)"
  exit 0
fi

# 2. 판정 0: 핵심 칸과 δ 칸을 먼저(관문이 읽는 전부), 관문 판정 뒤에 나머지 칸
dose_list "${CORE[@]}" "${DELTA[@]}"
runit "$PY" $J gate-hand --dir "$E" > "$E/gate.json" 2> "$E/gate.log"
rc=$?
note "관문(α·β·γ·δ·−2) 종료 코드 $rc (0 통과, 3 가를 수 없음, 4 도구 깨짐, 5 미판정): $(tr -d '\n ' 2>/dev/null < "$E/gate.json" | cut -c1-200)"
dose_list "${REST[@]}"
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 못 넘음 — 진화 갈래를 돌리지 않고 닫는다(가를 수 없음·도구 깨짐·미판정은 결과 문서에 그대로 적는다)"
  exit 0
fi

# 3. 진화 갈래: 같은 체크포인트에서 갈라 30만 스텝. 먼저 핵심 넷(켬·불활성·끔·위반 공급 끔), 시간이 남으면 나머지 넷
ON="$G norm_enabled=true norm_pun_return=0.1"
for s in a b; do
  run_branch on    $s $ON
  run_branch inert $s $G norm_enabled=true norm_mode=inert
  run_branch off   $s $G norm_enabled=false
  run_branch noimp $s $G norm_enabled=true norm_impulse=0
done
for s in a b; do
  run_branch altr  $s $G norm_enabled=true norm_pun_return=0           # 순이타(2차 딜레마가 가장 아픈 칸)
  run_branch mix   $s $ON norm_mix_period=200
  run_branch arb   $s $G norm_enabled=true norm_mark_src=sig norm_sig_theta=$SIG_THETA norm_pun_return=0   # 임의-관습(유병률 맞춘 임계)
  run_branch blind $s $ON norm_mem_obs_mode=blind
done
if [ "$SLOPE" = 1 ]; then   # 문턱 위가 1% 아래였을 때만: 기울기 시작판(공격 편향 b0 = −0.3)
  for s in a b; do
    ATTACK_B0=-0.3 run_branch slope-on    $s $ON
    ATTACK_B0=-0.3 run_branch slope-inert $s $G norm_enabled=true norm_mode=inert
    ATTACK_B0=-0.3 run_branch slope-off   $s $G norm_enabled=false
    ATTACK_B0=-0.3 run_branch slope-noimp $s $G norm_enabled=true norm_impulse=0
  done
fi

# 4. 판정(설계 판정 나·다·라): 주 판정은 켠 갈래의 겨루기 + inert·off 끝 무리 null + noimp 음성 대조 + inert 끝 무리와의 감각 막기 비교
for s in a b; do
  for arm in on inert off noimp; do judge $arm-$s block; done     # 판정 나: 켬 / null(inert·off) / 음성 대조(noimp)
  for arm in on inert; do judge $arm-$s knock-norm; done          # 감각 막기 겨루기: 켠 갈래와 inert 끝 무리
  judge on-$s knock-vis                                           # 양성 대조(시야 막기)
  for arm in on mix arb; do judge $arm-$s bottom; done            # 판정 다(on·mix)·라(arb)
  for arm in on inert off noimp altr mix arb blind; do judge $arm-$s obs; done   # 단서
  if [ "$SLOPE" = 1 ]; then
    for arm in on inert off noimp; do judge slope-$arm-$s obs; done
    judge slope-on-$s block
  fi
done
runit "$PY" $J final --dir "$E" --tags a b > "$E/final.json" 2> "$E/final.log"
note "최종 판정 종료 코드 $?: $(tr -d '\n ' 2>/dev/null < "$E/final.json" | cut -c1-200)"
note "체인 끝"
