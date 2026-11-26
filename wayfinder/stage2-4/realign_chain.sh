#!/usr/bin/env bash
# 독 순서를 이은 겨루기(kind_aligned, 2026-11-26)로 고정·긴 주기 갈래 판정을 다시 잰다.
# 옛 판정은 마당이 독을 새로 뽑아 고정 갈래가 양쪽 전멸(겨루기 깨짐)이었다.
# 쓰는 법: REPO=<원래 폴더> CODE=<코드 사본 폴더> bash realign_chain.sh   (memcap --bg로 감싸 띄운다)
set -u
REPO=${REPO:?원래 폴더}
CODE=${CODE:?코드 사본}
PY="$REPO/.venv/bin/python"
OUT="$REPO/runs/realign"
mkdir -p "$OUT"
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
log() { echo "$(TZ=America/New_York date '+%m-%d %H:%M') ET $*" >> "$OUT/progress.txt"; }
judge() {  # 갈래(예: fixed-a), 무엇(na|ga|obs)
    local t=$1 what=$2
    local ck="$REPO/runs/g-$t/checkpoints/ckpt_000002600000.npz"
    local out="$OUT/j-$what-$t"
    log "판정 $what-$t 시작"
    case $what in
        na)  nice -n 15 "$PY" wayfinder/stage2-4/taste_gene_judge.py rate-zero "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
        ga)  nice -n 15 "$PY" -m kkumteul.knockout split "$ck" --channel taste > "$out.json" 2> "$out.log" ;;
        obs) nice -n 15 "$PY" wayfinder/stage2-4/taste_gene_judge.py observe "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
    esac
    log "판정 $what-$t 끝 ($?)"
}
log "체인 시작 (코드 $CODE)"
judge fixed-a na
judge fixed-b na
judge fixed-a ga
judge p15000-a na
judge p500-a na
log "체인 끝"
