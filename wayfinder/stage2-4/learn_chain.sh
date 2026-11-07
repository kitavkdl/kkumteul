#!/bin/bash
# 개인 학습 실험 체인(사용자 새 방향 1번, 11/06 밤). 앞 실험(WAIT_FOR 패턴의 프로세스)이 끝난 뒤 한 번에 하나.
# 설계: docs/superpowers/specs/2026-11-07-individual-learning-design.md 둘째 판.
# 순서: wayfinder/stage2-4/results.md "개인 학습이 이득인 세계 — 예측" 끝 "순서".
#
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> [WAIT_FOR=<pgrep -f 패턴>] bash learn_chain.sh
# memcap 같은 메모리 상한 아래에서 부른다(상한은 묶음 전체). 갈래마다 done 표시가 있어 다시 불러도 이어 간다.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
E=$R/runs/learn-exp
PY=$R/.venv/bin/python
BASE=$R/runs/k4-ctrl/checkpoints/ckpt_000002300000.npz
END=ckpt_000002600000.npz
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2} PYTHONPATH=.
mkdir -p "$E"
note() { echo "$(TZ=America/New_York date '+%m-%d %H:%M ET') $*" | tee -a "$E/progress.txt"; }

if [ -n "${WAIT_FOR:-}" ]; then
  while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 30; done
  note "앞 실험 끝남 확인 — 체인 시작"
fi
cd "$C" || exit 1

run_branch() {  # 이름, 씨앗(a=같은 흐름, b=--new-stream), 설정...
  local name=$1 seed=$2; shift 2
  local dir=$R/runs/g-$name-$seed
  [ -f "$dir/done" ] && { note "$name-$seed 이미 끝남, 건너뜀"; return 0; }
  [ -f "$dir/run.json" ] && { note "$name-$seed 중간에 끊긴 흔적 — 건너뜀"; return 0; }
  local extra=(); [ "$seed" = b ] && extra=(--new-stream)
  local sets=(); for s in "$@"; do sets+=(--set "$s"); done
  note "$name-$seed 시작 ($*)"
  nice -n 15 "$PY" -m kkumteul.run branch --from "$BASE" --run "$dir" --steps 300000 --quiet \
    "${extra[@]}" "${sets[@]}" > "$dir.log" 2>&1
  local rc=$?
  touch "$dir/done"
  note "$name-$seed 끝 (종료 코드 $rc): $(tail -1 "$dir.log" | cut -c1-160)"
}

judge() {  # 이름-씨앗, 무엇(ga|gash|na|nae|obs)
  local tag=$1 what=$2
  local ck=$R/runs/g-$tag/checkpoints/$END
  local out=$E/j-$what-$tag
  [ -s "$out.json" ] && { note "판정 $what-$tag 이미 있음"; return 0; }
  [ -f "$ck" ] || { note "판정 $what-$tag: 체크포인트 없음 $ck"; return 0; }
  note "판정 $what-$tag 시작"
  local rc
  case $what in
    ga)   nice -n 15 "$PY" -m kkumteul.knockout split "$ck" --channel taste > "$out.json" 2> "$out.log" ;;
    gash) nice -n 15 "$PY" -m kkumteul.knockout split "$ck" --channel taste --mode shuffle > "$out.json" 2> "$out.log" ;;
    na)   nice -n 15 "$PY" wayfinder/stage2-4/taste_gene_judge.py rate-zero "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
    nae)  nice -n 15 "$PY" wayfinder/stage2-4/taste_gene_judge.py rate-zero "$ck" --gene eta --json "$out.json" > "$out.log" 2>&1 ;;
    obs)  nice -n 15 "$PY" wayfinder/stage2-4/taste_gene_judge.py observe "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
  esac
  rc=$?
  note "판정 $what-$tag 끝 ($rc): $(tail -1 "$out.log" | cut -c1-160)"
}

G="preset=taste taste_inherit=0 taste_gene_enabled=true"
# a 갈래 넷
run_branch p500   a $G kind_flip_period=500
run_branch fixed  a $G kind_flip_period=1000000000
run_branch p50    a $G kind_flip_period=50
run_branch p15000 a $G kind_flip_period=15000
# a 판정: 500·고정은 가·나·독, 50·15,000은 나·독
for t in p500-a fixed-a; do judge $t na; judge $t obs; judge $t ga; judge $t gash; done
for t in p50-a p15000-a; do judge $t na; judge $t obs; done
# b 갈래 둘 + 나·독
run_branch p500  b $G kind_flip_period=500
run_branch fixed b $G kind_flip_period=1000000000
for t in p500-b fixed-b; do judge $t na; judge $t obs; done
# 기본 1,500과 첫걸음(시작 속도 0.05)
run_branch p1500 a $G kind_flip_period=1500
run_branch first a $G kind_flip_period=500 taste_rate=0.05
for t in p1500-a first-a; do judge $t na; judge $t obs; done
# 헤브 갈래(설계 둘째 판 명세 2): 맛 끔, 기존 평생 학습 켬
run_branch hebb a mind_enabled=true kind_flip_period=500
judge hebb-a nae
judge hebb-a obs
note "체인 끝"
