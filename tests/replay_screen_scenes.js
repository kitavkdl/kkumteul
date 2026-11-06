// 다시 보기 둘째 판의 손 점검: 장면 여럿 파일을 헤드리스 브라우저로 띄워 장면을 세 번 바꾸고,
// 옛 장면의 그리기 반복·창 리스너가 남지 않는지(해제 함수 탐침 KKUMTEUL_APP), 스크립트 오류가 없는지,
// 옆으로 넘치지 않는지, 철·뭉쳐 자기·맛 기억 글이 뜨는지 본다(pytest에 들지 않는다. 브라우저가 따로 떠 있어야 한다).
// 1) 크롬 헤드리스를 띄운다(replay_screen_probe.js 머리말과 같다).
// 2) node tests/replay_screen_scenes.js <장면 여럿 다시 보기 html> 9333 [너비 390]
"use strict";
const fs = require("fs");
const [, , file, port, width] = process.argv;
const base = `http://127.0.0.1:${port}`;

async function main() {
  const t = await (await fetch(`${base}/json/new?about:blank`, { method: "PUT" })).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0; const wait = new Map(); const errors = [];
  ws.addEventListener("message", (m) => {
    const d = JSON.parse(m.data);
    if (d.id && wait.has(d.id)) { wait.get(d.id)(d); wait.delete(d.id); }
    if (d.method === "Runtime.exceptionThrown") errors.push(d.params.exceptionDetails.exception ? d.params.exceptionDetails.exception.description : d.params.exceptionDetails.text);
    if (d.method === "Runtime.consoleAPICalled" && d.params.type === "error") errors.push(d.params.args.map((a) => a.value || a.description).join(" "));
  });
  const cmd = (method, params = {}) => new Promise((r) => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => {
    const d = await cmd("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    if (d.result.exceptionDetails) throw new Error(JSON.stringify(d.result.exceptionDetails).slice(0, 400));
    return d.result.result.value;
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const W = Number(width || 1280);
  await cmd("Page.enable"); await cmd("Runtime.enable");
  await cmd("Emulation.setDeviceMetricsOverride", { width: W, height: W < 600 ? 844 : 900, deviceScaleFactor: 1, mobile: W < 600 });
  await cmd("Page.navigate", { url: "file://" + fs.realpathSync(file) });
  for (let i = 0; i < 120; i++) {
    if (await ev(`!!(window.KKUMTEUL_APP && window.KKUMTEUL_APP.gen > 0 && document.getElementById("rp-track").getAttribute("aria-valuenow") !== null)`).catch(() => false)) break;
    await sleep(500);
  }
  const state = () => ev(`(() => {
    const $ = (i) => document.getElementById(i), txt = (i) => $(i) ? $(i).textContent : null;
    return { gen: KKUMTEUL_APP.gen, disposed: KKUMTEUL_APP.disposed, title: txt("rp-title"), guide: $("rp-guide").hidden ? null : txt("rp-guide"),
      maps: document.querySelectorAll("#rp-map").length, keys: txt("rp-keys"), trackkey: txt("rp-trackkey"),
      taste: $("rp-taste").hidden ? null : txt("rp-taste"), winter: $("rp-stage").classList.contains("winter"),
      legend: Array.from(document.querySelectorAll("#rp-legend b"), (b) => b.textContent),
      mem: Array.from(document.querySelectorAll("#rp-story li"), (l) => l.textContent).filter((s) => s.includes("기억")).slice(0, 4),
      scenes: Array.from(document.querySelectorAll("#rp-scenes button"), (b) => b.textContent + (b.getAttribute("aria-pressed") === "true" ? "*" : "")),
      overflow: document.documentElement.scrollWidth - window.innerWidth };
  })()`);
  // 세대마다의 그리기 반복을 0.6초 동안 센다. 살아 있는 반복은 지금 세대 하나뿐이어야 한다
  const loops = async () => { await ev(`(KKUMTEUL_APP.ticks = {}, 0)`); await sleep(600); return ev(`KKUMTEUL_APP.ticks`); };
  // 장면 f 비율로 가서 멈춘다
  const seekFrac = (q) => ev(`(() => { const tr = document.getElementById("rp-track"), p = document.getElementById("rp-play");
    if (p.textContent === "멈춤") p.click();
    const last = Number(tr.getAttribute("aria-valuemax")), target = Math.round(${q} * last);
    tr.dispatchEvent(new KeyboardEvent("keydown", {key: "Home", bubbles: true}));
    for (let i = 0; i < Math.floor(target / 25); i++) tr.dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowRight", shiftKey: true, bubbles: true}));
    for (let i = 0; i < target % 25; i++) tr.dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowRight", bubbles: true}));
    return target; })()`);
  const out = { file, width: W, steps: [] };
  const click = (i) => ev(`(document.querySelectorAll("#rp-scenes button")[${i}].click(), 0)`);
  out.steps.push({ at: "처음", ...(await state()), loops: await loops() });
  for (const i of [1, 0, 1]) {
    await click(i); await sleep(300);
    out.steps.push({ at: "장면 " + i, ...(await state()), loops: await loops() });
  }
  // 지금 장면(1)과 장면 0의 여러 자리에서 상태 줄과 맛 막대
  for (const sc of [1, 0]) {
    await click(sc); await sleep(300);
    for (const q of [0.05, 0.5, 0.95]) {
      await seekFrac(q); await sleep(250);
      const s = await state();
      out.steps.push({ at: `장면 ${sc} 위치 ${q}`, keys: s.keys, taste: s.taste, winter: s.winter, mem: s.mem });
    }
  }
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  ws.close();
  await fetch(`${base}/json/close/${t.id}`).catch(() => {});
}
main().catch((e) => { console.error(e); process.exit(1); });
