// Runs index.html in headless Chrome (a visible page, so timers aren't throttled) and prints window.__result.
// Usage: node run.mjs [wasm|webgpu] (the static server from serve.mjs must be on 5199)
import { chromium } from "playwright-core";
const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-webgpu"] });
const page = await browser.newPage();
page.on("console", (m) => console.error(m.text()));
await page.goto("http://localhost:5199/index.html?device=" + (process.argv[2] || "wasm") + (process.argv[3] ? "&model=" + process.argv[3] : ""));
await page.waitForFunction(() => window.__result, null, { timeout: 600000, polling: 1000 });
console.log(JSON.stringify(await page.evaluate(() => window.__result), null, 2));
await browser.close();
