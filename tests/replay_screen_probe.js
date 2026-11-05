// 다시 보기 화면을 헤드리스 브라우저로 띄워 장면 몇 곳에서 지도 캔버스의 선·채움 호출을 빛깔별로 센다.
// 사건 표시(공격 #ff5f57, 건넴 #ffd36b, 먹음 #7dffa8, 죽음 #d5dce2, 태어남 #fff2b0, 운석 #ff9f43, 신호 고리 hsl(?,85%,68%))가
// 그려지는지 보는 손 점검이다(pytest에 들지 않는다. 브라우저가 따로 떠 있어야 한다). 눈에 잘 띄는지는 재지 않는다.
// 1) 크롬 헤드리스를 띄운다(메모리 상한을 건다):
//      chrome-headless-shell --remote-debugging-port=9333 --no-sandbox --allow-file-access-from-files --user-data-dir=<임시 폴더> about:blank
//    playwright를 쓰면 `npx playwright install chromium-headless-shell`이 받아 둔다.
// 2) node tests/replay_screen_probe.js <다시 보기 html> 9333 [장면 비율 0~1 ...]
//    "세계 전체"와 "가까이" 보기에서 각 장면의 사건별 호출 수와 시간 막대 설명(trackkey)을 JSON으로 찍는다.
const fs = require("fs");
const [, , file, port, ...fr] = process.argv;
const fracs = fr.length ? fr.map(Number) : [0.1, 0.5, 0.9];
const base = `http://127.0.0.1:${port}`;

const HOOK = `(() => {
  const P = CanvasRenderingContext2D.prototype, s0 = P.stroke, f0 = P.fill;
  window.__cnt = {};
  const bump = (k) => { window.__cnt[k] = (window.__cnt[k] || 0) + 1; };
  P.stroke = function () { if (this.canvas.id === "rp-map") bump("s " + this.strokeStyle); return s0.apply(this, arguments); };
  P.fill = function () { if (this.canvas.id === "rp-map") bump("f " + this.fillStyle); return f0.apply(this, arguments); };
})();`;

async function main() {
  const t = await (await fetch(`${base}/json/new?about:blank`, { method: "PUT" })).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0; const wait = new Map();
  ws.addEventListener("message", (m) => { const d = JSON.parse(m.data); if (d.id && wait.has(d.id)) { wait.get(d.id)(d); wait.delete(d.id); } });
  const cmd = (method, params = {}) => new Promise((r) => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => {
    const d = await cmd("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    if (d.result.exceptionDetails) throw new Error(JSON.stringify(d.result.exceptionDetails).slice(0, 400));
    return d.result.result.value;
  };
  await cmd("Page.enable"); await cmd("Runtime.enable");
  await cmd("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await cmd("Page.addScriptToEvaluateOnNewDocument", { source: HOOK });
  await cmd("Page.navigate", { url: "file://" + fs.realpathSync(file) });
  for (let i = 0; i < 120; i++) {
    const ok = await ev(`!!(document.getElementById("rp-track") && document.getElementById("rp-track").getAttribute("aria-valuenow") !== null)`).catch(() => false);
    if (ok) break; await new Promise((r) => setTimeout(r, 500));
  }
  const raf = `new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(r))))`;
  const key = (k) => `document.getElementById("rp-track").dispatchEvent(new KeyboardEvent("keydown", {key: ${JSON.stringify(k)}, bubbles: true}))`;
  await ev(`(document.getElementById("rp-play").textContent === "멈춤" && document.getElementById("rp-play").click(), ${raf})`);
  await ev(key("End")); await ev(raf);
  const last = Number(await ev(`document.getElementById("rp-track").getAttribute("aria-valuenow")`));
  const out = { file, frames: last + 1, views: {} };
  for (const view of ["rp-far", "rp-near"]) {
    await ev(`(document.getElementById(${JSON.stringify(view)}).click(), ${raf})`);
    for (const q of fracs) {
      const target = Math.round(q * last);
      await ev(`(() => { const tr = document.getElementById("rp-track"); tr.dispatchEvent(new KeyboardEvent("keydown", {key: "Home", bubbles: true}));
        for (let i = 0; i < ${Math.floor(target / 25)}; i++) tr.dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowRight", shiftKey: true, bubbles: true}));
        for (let i = 0; i < ${target % 25}; i++) tr.dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowRight", bubbles: true})); })()`);
      await ev(raf);
      await ev(`(window.__cnt = {}, document.getElementById("rp-track").dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowRight", bubbles: true})), document.getElementById("rp-track").dispatchEvent(new KeyboardEvent("keydown", {key: "ArrowLeft", bubbles: true})), ${raf})`);
      const cnt = await ev(`window.__cnt`);
      const step = await ev(`document.getElementById("rp-step").textContent`);
      // 신호 고리는 hsl(?, 85%, 68%) 선이다. 같은 빛깔의 채움이 없는 선만 센다.
      const hsl = (h) => { const r = parseInt(h.slice(1, 3), 16) / 255, g = parseInt(h.slice(3, 5), 16) / 255, b = parseInt(h.slice(5, 7), 16) / 255;
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, s = mx === mn ? 0 : (mx - mn) / (1 - Math.abs(2 * l - 1)); return [s, l]; };
      let ring = 0;
      for (const [k, v] of Object.entries(cnt)) {
        if (!k.startsWith("s #") || ["s #ff5f57", "s #ffd36b", "s #7dffa8", "s #d5dce2", "s #fff2b0", "s #ff9f43"].includes(k)) continue;
        const [s, l] = hsl(k.slice(2));
        if (Math.abs(s - 0.85) < 0.04 && Math.abs(l - 0.68) < 0.02 && !cnt["f " + k.slice(2)]) ring += v;
      }
      out.views[`${view} ${target}`] = { step, hit: cnt["s #ff5f57"] || 0, gave: cnt["s #ffd36b"] || 0, eat: cnt["s #7dffa8"] || 0,
        died: cnt["s #d5dce2"] || 0, born: cnt["f #fff2b0"] || 0, strike: cnt["f #ff9f43"] || 0, ring };
    }
  }
  out.trackkey = await ev(`document.getElementById("rp-trackkey").textContent`);
  console.log(JSON.stringify(out, null, 1));
  ws.close();
  await fetch(`${base}/json/close/${t.id}`).catch(() => {});
}
main().catch((e) => { console.error(e); process.exit(1); });
