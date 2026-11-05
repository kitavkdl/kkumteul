# 진행 중 작업

대표님 지시(2026-11-05 새벽): 오전 8시(ET)까지 가능한 많은 단계를 진행한다. 이 레포에서 세션을 열어 이어 간다.

**맡은 곳 (01:55 ET 대표님 지시)**: 서버에서 연 꿈틀 세션이 전담한다. 맥의 앞 세션은 아래 "지금 돌고 있는 것"(세 갈래 실험과 비교, 결과 기록)까지만 하고 인수인계를 이 레포에 남긴 뒤 끝낸다. "다음에 할 것"은 서버 세션이 한다 — 맥 세션은 새로 시작하지 않는다.

## 맥의 앞 세션이 끝낸 것과 남긴 것 (2026-11-05 02:05 ET, 맥 세션은 끝남)
- [x] 단계 실험 세 갈래와 대조 — 끝남. 맥 `runs/x-ctrl`, `x-s2`, `x-s3`, `x-s4`(git에 없음).
- [x] 세 갈래 비교와 기록 — `wayfinder/stage2-4/results.md`, 원자료 `wayfinder/stage2-4/results/compare-s{2,3,4}.json`. README 단계 표도 고쳤다.
  결론: 켠 세계의 생물은 켠 규칙에 적응했지만, 학습(학습률 0.01 → 0.0001~0.0006)과 주기는 진화가 꺼 버렸고, 신호·혈연·길·거름은 쓰인다는 증거가 없다.
- 맥 세션이 지시 뒤에 잘못 새로 시작했다가 멈춘 것(서버 세션이 참고만 한다):
  - 맥의 `runs/k-ctrl`, `runs/k-mind`, `runs/c-s3`은 돌다 만 것이다. 결과로 쓰지 않는다. 지워도 된다.
  - 새 코드 검토 워크플로(6갈래)는 중간에 멈췄다. 결과는 없다. 아래 "독립 검토" 항목은 그대로 남아 있다.
  - 협력 먹이(곁에 다른 개체가 있어야 먹는 큰 먹이, `preset=coop`)를 쓰다 말았다. 가지 `wip/coop-food`에 있고 시험을 끝까지 돌리지 못했다. `main`에는 없다. 쓸지 말지는 서버 세션이 정한다. 넣은 까닭은 `wayfinder/stage2-4/results.md`의 "그래서 하는 것".

## 다음에 할 것 (위에서부터)
- [ ] "배워야 먹는 세계" 실험 — `preset=kinds`(먹이 두 종류와 독)에서 학습이 값을 하는지. 서버의 `runs/main` 100만 스텝에서 `--set preset=kinds`(대조, `runs/k-ctrl`)와 `--set preset=stage2 --set preset=kinds`(`runs/k-s2`)를 30만 스텝씩 가르고(02:00 ET 서버에서 시작, 각 2스레드) `kkumteul.compare`로 견준다. 학습률이 유지되는지도 본다. 완료 기준: 결과가 `wayfinder/stage2-4/results.md`에 적힘
- [ ] 공격(`preset=conflict`) 세계가 무너지지 않는지 — `preset=stage3 preset=conflict`로 20만 스텝. 개체 수, 재주입, 공격 횟수. 무너지면 `attack_damage`·`attack_gain`을 낮춘다
- [ ] 새로 넣은 것(공격, 먹이 종류, 이주, 비교 도구) 독립 검토 — 맥락 없는 검토자에게 `kkumteul/stages.py`의 `attack`, `kernels.py`의 먹이 종류(`good_kind`, `eat`, `_see`의 부호), `pack.migrate`, `compare.py`를 맡긴다
- [ ] 서버의 세계 둘을 최신 코드로 다시 띄우기 — 서버 폴더의 코드는 01:55 ET에 최신으로 받았다. 돌고 있는 `kkumteul-main`, `kkumteul-world4`는 띄울 때의 옛 코드로 계속 돈다. 멈추고 `resume`으로 다시 띄우면 새 코드가 된다(옛 체크포인트는 채워 읽는다). 멈추고 띄우는 일이니 대표님께 한 줄 알리고 한다
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
