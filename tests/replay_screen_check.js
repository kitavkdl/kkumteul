// 다시 보기 화면(kkumteul/replay.html)의 계산 부분을 node에서 불러, 파이썬 시험이 견줄 값을 JSON으로 낸다.
//
//     node tests/replay_screen_check.js <내보낸 다시 보기 파일>
//
// 화면 스크립트는 document가 없으면 계산 함수(KKUMTEUL_REPLAY)만 내놓고 끝난다. 그리기는 여기서 보지 않는다.
"use strict";
const fs = require("fs");

const html = fs.readFileSync(process.argv[2], "utf8");
const metaText = /<script type="application\/json" id="replay-meta">([\s\S]*?)<\/script>/.exec(html)[1];
const meta = JSON.parse(metaText.split("<\\/").join("</"));
const b64 = /<script type="application\/octet-stream" id="replay-bin">([\s\S]*?)<\/script>/.exec(html)[1];
const scripts = Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/g), (m) => m[1]);
if (scripts.length !== 1) throw new Error("화면 스크립트는 하나여야 한다: " + scripts.length);
new Function(scripts[0]).call(globalThis);
const KK = globalThis.KKUMTEUL_REPLAY;
if (!KK) throw new Error("화면 스크립트가 계산 함수(KKUMTEUL_REPLAY)를 내놓지 않았다");

const A = KK.decode(meta, b64);
const D = KK.build(meta, A);

const agents = [];
for (let n = 0; n < D.nA; n++) {
  const st = KK.story(D, n);
  const sum = (t) => st.filter((e) => e.type === t).reduce((s, e) => s + (e.n || 1), 0);
  const born = st.find((e) => e.type === "born");
  const died = st.find((e) => e.type === "died");
  agents.push({
    first: D.first[n], last: D.last[n],
    good: sum("good"), poison: sum("poison"), kids: sum("child"),
    born: born ? born.step : null, died: died ? died.step : null, cause: died ? died.cause : null,
    sorted: st.every((e, i) => i === 0 || st[i - 1].f <= e.f),
    texts: st.every((e) => typeof e.text === "string" && e.text.length > 0),
    inRange: st.every((e) => e.f >= 0 && e.f < D.nF),
    // 맛 기억 자막(둘째 판): [장면, 칸, 부호]와 물려받음 줄
    mem: st.filter((e) => e.type === "mem").map((e) => [e.f, e.slot, e.sign]),
    inherit: st.some((e) => e.type === "inherit"),
  });
}

const mid = D.nF >> 1;
const windows = {};
for (const f of [0, 1, mid, D.nF - 1]) windows[f] = KK.windowStats(D, f, 100);

const found = [];
for (let n = 0; n < D.nA; n++) {
  const p = KK.find(D, mid, n);
  if (p >= 0) found.push([n, A.fr_x[p], A.fr_y[p]]);
}

const events = {};
for (let i = 0; i < D.ev.type.length; i++) events[D.ev.type[i]] = (events[D.ev.type[i]] || 0) + 1;
let evSorted = true;
for (let f = 0; f < D.nF; f++) {
  for (let i = D.ev.off[f]; i < D.ev.off[f + 1]; i++) if (D.ev.f[i] !== f) evSorted = false;
}

const verdicts = [
  KK.verdict({ eaten: 200, poisoned: 100, blind: 0.5 }, true),
  KK.verdict({ eaten: 200, poisoned: 40, blind: 0.5 }, true),
  KK.verdict({ eaten: 200, poisoned: 150, blind: 0.5 }, true),
  KK.verdict({ eaten: 5, poisoned: 1, blind: 0.5 }, true),
  KK.verdict({ eaten: 200, poisoned: 0, blind: 0 }, false),
  // 독이 바뀌기 앞 구간과 견준다: 바뀐 뒤에도 독을 덜 먹으면 가려 먹는 것, 거꾸로면 모양을 가리는 것
  KK.verdict({ eaten: 200, poisoned: 40, blind: 0.5 }, true, { eaten: 200, poisoned: 40, blind: 0.5 }),
  KK.verdict({ eaten: 200, poisoned: 100, blind: 0.37 }, true, { eaten: 200, poisoned: 100, blind: 0.63 }),
  KK.verdict({ eaten: 200, poisoned: 100, blind: 0.63 }, true, { eaten: 200, poisoned: 100, blind: 0.37 }),
  KK.verdict({ eaten: 200, poisoned: 40, blind: 0.5 }, true, { eaten: 3, poisoned: 3, blind: 0.5 }),
];
const segs = [0, 1, mid, D.nF - 1].map((f) => KK.segOf(D, f));

process.stdout.write(JSON.stringify({
  nF: D.nF, nA: D.nA, mid, agents, windows, found, events, evSorted, evTypes: KK.EV, verdicts, segs,
  frameOf: [D.start - 5, D.start, D.start + 1, D.start + D.every, D.start + D.every + 1, D.start + 10 * D.every * D.nF]
    .map((s) => KK.frameOf(D, s)),
  lifted: [KK.lift(0, 0, 0), KK.lift(10, 10, 120), KK.lift(250, 250, 250)],
  seasonRuns: KK.seasonRuns(D),
  huddlers: Array.from({ length: D.nF }, (_, f) => KK.huddlers(D, f)),
}));
