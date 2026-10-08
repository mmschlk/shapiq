const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

test("closed coverage exposes unprepared cases through existing authenticated detail loader", async () => {
  const summary = { children: [], replaceChildren(...nodes) { this.children = nodes; }, append(node) { this.children.push(node); } };
  const elements = [], cases = [{ case: 0, preparation_status: "unclaimed", not_reached_supported: 9 }];
  let downloaded, lookup;
  const context = vm.createContext({
    data: {
      suite: { protocol: { name: "Test" } },
      catalog: { datasets: ["test"], min_players: 4, max_players: 8 },
      campaign_coverage: { closed: true, prepared_instances: 1, intended_instances: 2, unprepared_instances: 1,
        never_attempted_supported: 3, not_reached_supported: 9, details: {type: "report", id: "coverage"} },
    },
    document: {
      getElementById: () => summary,
      createTextNode: (textContent) => ({ textContent }),
      createElement: () => { const e = { style: { setProperty() {} }, children: [], append(...nodes) { this.children.push(...nodes); }, addEventListener: (name, fn) => { e[name] = fn; }, click() { this.clicked = true; } }; elements.push(e); return e; },
    },
    BenchmarkDetails: { object: async (manifest, key, options) => { lookup = key; await options.read({file: "authenticated-details"}); return cases; } },
    BenchmarkPartitions: { read: async (manifest, descriptor) => { assert.equal(descriptor.file, "authenticated-details"); } },
    Blob,
    URL: { createObjectURL: (blob) => { downloaded = blob; return "blob:test"; }, revokeObjectURL() {} },
    setTimeout: (fn) => fn(),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../../benchmark/site/query.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../../benchmark/site/protocol.js"), "utf8"), context);
  vm.runInContext("renderReportSummary(false)", context);
  const textOf = n => (n.textContent || "") + (n.children || []).map(textOf).join("");
  const text = summary.children.map(textOf).join("");
  assert.match(text, /50% reference coverage/);
  assert.match(text, /1 of 2 intended instances/);
  assert.match(text, /3 supported cells.*not attempted/);
  assert.match(text, /9 planned supported cells.*no qualified reference/);
  await elements.find(e => e.href === "#").click({preventDefault() {}});
  assert.deepEqual(JSON.parse(JSON.stringify(lookup)), {type: "report", id: "coverage"});
  assert.deepEqual(JSON.parse(await downloaded.text()).cases, cases);
  downloaded = null;
  context.BenchmarkDetails.object = async () => {
    context.data = { suite: {} }; // A different report was opened during the read.
    return cases;
  };
  await elements.find(e => e.href === "#").click({preventDefault() {}});
  assert.equal(downloaded, null);
});
