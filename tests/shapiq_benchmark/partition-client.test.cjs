const test = require("node:test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  vm = require("node:vm"),
  path = require("node:path");
const source = fs.readFileSync(
  path.resolve(__dirname, "../../benchmark/site/partition-client.js"),
  "utf8",
);
function fixture() {
  const workers = [];
  class Worker {
    constructor(url) {
      this.url = url;
      this.messages = [];
      this.terminated = false;
      workers.push(this);
    }
    postMessage(value) {
      this.messages.push(value);
    }
    terminate() {
      this.terminated = true;
    }
  }
  const context = vm.createContext({ Worker, DOMException });
  vm.runInContext(source, context);
  return { Client: context.BenchmarkPartitionClient, workers };
}
test("one worker caches identical requests and preserves local files", async () => {
  const { Client, workers } = fixture(),
    files = new Map([["data.json", { name: "data.json" }]]),
    client = new Client({ snapshot_id: "a" }, files),
    worker = workers[0];
  const first = client.query({ selection: 1 }),
    again = client.query({ selection: 1 });
  assert.equal(first, again);
  assert.equal(worker.messages.length, 1);
  assert.equal(worker.messages[0].files, files);
  worker.onmessage({ data: { id: 1, result: { table: [1] } } });
  assert.deepEqual(await first, { table: [1] });
  assert.equal(client.query({ selection: 1 }), first);
});
test("superseded and closed requests reject; stale replies cannot resolve current selection", async () => {
  const { Client, workers } = fixture(),
    client = new Client({}),
    worker = workers[0];
  const first = client.query({ q: 1 });
  const rejection = assert.rejects(first, { name: "AbortError" });
  const second = client.query({ q: 2 });
  await rejection;
  worker.onmessage({ data: { id: 1, result: "stale" } });
  assert.equal(client.pending.id, 2);
  worker.onmessage({ data: { id: 2, result: "current" } });
  assert.equal(await second, "current");
  const third = client.query({ q: 3 }),
    closed = assert.rejects(third, { name: "AbortError" });
  client.close();
  await closed;
  assert(worker.terminated);
  await assert.rejects(client.query({ q: 4 }), { name: "AbortError" });
});
test("request errors can retry; fatal worker or transport errors never leave retries pending", async () => {
  const { Client, workers } = fixture(),
    client = new Client({}),
    worker = workers[0];
  let pending = client.query({ q: 1 });
  const error = assert.rejects(pending, /bad selection/);
  worker.onmessage({ data: { id: 1, error: "bad selection" } });
  await error;
  pending = client.query({ q: 1 });
  assert.equal(worker.messages.length, 2);
  const fatal = assert.rejects(pending, /failed script/);
  worker.onerror({ message: "failed script" });
  await fatal;
  await assert.rejects(client.query({ q: 2 }), /failed script/);
  assert(worker.terminated);
  const other = new Client({}),
    otherWorker = workers[1],
    promise = other.query({ q: 1 }),
    bad = assert.rejects(promise, /could not be decoded/);
  otherWorker.onmessageerror({});
  await bad;
  await assert.rejects(other.query({ q: 2 }), /could not be decoded/);
});
