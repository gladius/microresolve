// Connected MicroResolve client for the agent-tool-router demo.
//
// Subscribes to a running Studio at http://localhost:3001 and keeps a
// local copy of the `agent` namespace in ./data-node/.
//
// Usage:
//   node agent_router_client.js status
//   node agent_router_client.js watch
//   node agent_router_client.js resolve "cancel my subscription"

const { MicroResolve } = require('microresolve');

const SERVER = 'http://localhost:3001';
const NAMESPACE = 'agent';
const DATA_DIR = './data-node';
const TICK_SECS = 5;

const mr = new MicroResolve({
  serverUrl: SERVER,
  subscribe: [NAMESPACE],
  tickIntervalSecs: TICK_SECS,
  dataDir: DATA_DIR,
});
const ns = mr.namespace(NAMESPACE);

const cmd = process.argv[2] || 'status';

if (cmd === 'status') {
  console.log(`version=${ns.version()}  intents=${ns.intentCount()}`);
  process.exit(0);
}

if (cmd === 'watch') {
  console.log(`watching '${NAMESPACE}' on ${SERVER} (tick every ${TICK_SECS}s)…`);
  let last = -1;
  setInterval(() => {
    const v = ns.version();
    if (v !== last) {
      console.log(`v${v}  intents=${ns.intentCount()}`);
      last = v;
    }
  }, 1000);
}

if (cmd === 'resolve') {
  const query = process.argv.slice(3).join(' ');
  if (!query) {
    console.error('usage: node agent_router_client.js resolve <query…>');
    process.exit(2);
  }
  const matches = ns.resolve(query);
  if (matches.length) {
    console.log(`${matches[0].id}  score=${matches[0].score.toFixed(2)}`);
  } else {
    console.log('(no match)');
  }
  process.exit(0);
}
