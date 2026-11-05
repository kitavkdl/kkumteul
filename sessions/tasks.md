# 진행 중 작업

대표님 지시(2026-11-05 새벽): 오전 8시(ET)까지 가능한 많은 단계를 진행한다. 이 레포에서 세션을 열어 이어 간다.

## 지금 돌고 있는 것 (2026-11-05 01:40 ET 기준, 맥)
- [ ] 단계 실험 세 갈래 — `runs/x-s2`(마음), `runs/x-s3`(+사회), `runs/x-s4`(+판)가 `runs/demo`의 80만 스텝 체크포인트에서 갈라져 110만 스텝까지 돈다. 대조는 `runs/x-ctrl`(끝남, 110만 스텝 체크포인트 있음).
  확인: `pgrep -fl "kkumteul.run branch"`, `tail -1 runs/x-s2/stats.jsonl`.
- [ ] 세 갈래 비교 — 갈래가 끝나면 앞 세션이 걸어 둔 명령이 `wayfinder/stage2-4/results/compare-s{2,3,4}.{json,txt}`를 쓴다. **파일이 없거나 비어 있으면 직접 돌린다**:
  `python3 -m kkumteul.compare --control runs/x-ctrl --treatment runs/x-s2 --at 1100000 --reps 4 --out wayfinder/stage2-4/results/compare-s2.json` (s3, s4도 같게).
  완료 기준: 단계마다 "켠 규칙에서 이김 / 양쪽에서 이김 / 가를 수 없음"과 형질 변화가 `wayfinder/stage2-4/results.md`에 적히고 README 단계 표가 고쳐짐. 추세만으로 "쓰게 됐다"고 쓰지 않는다.
- 중간에 본 것: 세 갈래 모두 평균 학습률이 0.0065 → 0.0015로 줄었다(배움이 값을 못 한다). 4단계 갈래는 거름 합이 160~196까지 올랐다(쌓기가 선택됐는지, 변이로 떠다닌 것인지는 비교로 봐야 한다).

## 다음에 할 것 (위에서부터)
- [ ] "배워야 먹는 세계" 실험 — `preset=kinds`(먹이 두 종류와 독)에서 학습이 값을 하는지. `runs/demo` 80만 스텝에서 `--set preset=kinds`(대조)와 `--set preset=stage2 --set preset=kinds`를 30만 스텝씩 가르고 `kkumteul.compare`로 견준다. 학습률이 유지되는지도 본다. 완료 기준: 결과가 `wayfinder/stage2-4/results.md`에 적힘
- [ ] 공격(`preset=conflict`) 세계가 무너지지 않는지 — `preset=stage3 preset=conflict`로 20만 스텝. 개체 수, 재주입, 공격 횟수. 무너지면 `attack_damage`·`attack_gain`을 낮춘다
- [ ] 새로 넣은 것(공격, 먹이 종류, 이주, 비교 도구) 독립 검토 — 맥락 없는 검토자에게 `kkumteul/stages.py`의 `attack`, `kernels.py`의 먹이 종류(`good_kind`, `eat`, `_see`의 부호), `pack.migrate`, `compare.py`를 맡긴다
- [ ] 서버를 최신 코드로 — `ssh ubuntu-tailscale`에서 `cd ~/git/kkumteul && systemctl --user stop kkumteul-main kkumteul-world4 && git pull` 뒤 다시 띄운다(인수인계의 서버 항목 참고). 옛 체크포인트는 채워 읽으므로 이어 돌 수 있다. 멈추고 띄우는 일이니 대표님께 한 줄 알리고 한다
- [ ] 관찰 화면: 먹이 종류(독) 색, 공격 표시. 프레임에는 아직 먹이 종류가 없다(`run.py`의 `encode_frame`)
- [ ] 씨앗 세계 공개 — 새 형식 체크포인트 하나를 GitHub 릴리스에 올리고 `CONTRIBUTING.md`에 받는 법을 적는다(서버의 `gh`로). 올리기 전에 대표님께 알린다
- [ ] `wayfinder/stage2-4/map.md` — 2~4단계의 임시 결정을 1단계처럼 지도와 티켓으로 정리(지금은 `docs/stages-2-4.md`와 검토 기록뿐)
- [ ] 마무리(07:30 ET 전): 인수인계 고쳐 쓰기, 커밋·push, 대표님께 결과 요약

## 대표님이 정하실 것
- [ ] "하루 10세대"의 뜻 — 지금 값으로는 하루 약 11만 세대(맥)가 쌓인다. 10세대를 하한으로 읽었다. `wayfinder/stage1/T09-death-reproduction-generation.md`
- [ ] 관찰 화면을 직접 보고 더 보여야 할 것 — `wayfinder/stage1/T07-observation-screen.md`
- [ ] 라이선스 확인 — MIT를 임시로 넣었다. `LICENSE`
- [ ] 서버 서비스 등록 — `deploy/kkumteul.user.service`의 머리말대로. 지금은 일회성 유닛이라 재부팅하면 안 뜬다(등록 명령은 자동 승인에서 거절됨)
- [ ] 5단계(말하는 존재)를 할지, 어느 길로 — `docs/stage5-proposal.md`
