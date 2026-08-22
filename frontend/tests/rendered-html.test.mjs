import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the ChurnSignal assessment interface", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>ChurnSignal \| Local retention copilot<\/title>/i);
  assert.match(html, /Build a churn picture through conversation\./);
  assert.match(html, /Profile readiness/);
  assert.match(html, /Score CSV \/ Excel/);
  assert.match(html, /Tell me about the customer in your own words/);
  assert.match(html, /Awaiting profile/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("source connects every documented ChurnSignal workflow", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /const REQUIRED_COUNT = 19/);
  assert.match(page, /`\$\{API_URL\}\/health`/);
  assert.match(page, /`\$\{API_URL\}\/chat`/);
  assert.match(page, /`\$\{API_URL\}\/report`/);
  assert.match(page, /`\$\{API_URL\}\/bulk-predict`/);
  assert.match(page, /accept="\.csv,\.xlsx"/);
  assert.match(page, /Download professional report/);
  assert.match(layout, /title:\s*"ChurnSignal \| Local retention copilot"/);
  assert.match(packageJson, /"name": "churnsignal-frontend"/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
});
