/**
 * connected.js — connected-mode demo for microresolve
 *
 * Sync a namespace from a running MicroResolve server, classify locally,
 * push a correction back.
 *
 * Start the server first:
 *   ../target/release/server --port 3001 --no-browser --data /tmp/mr_server_data &
 *
 * Run:
 *   node examples/connected.js
 *   # or override:
 *   MR_SERVER_URL=http://localhost:3097 MR_API_KEY=mr_xxx node examples/connected.js
 */

const { Engine } = require('..');

const SERVER_URL = process.env.MR_SERVER_URL || 'http://localhost:3001';
const API_KEY    = process.env.MR_API_KEY    || undefined;
const NS         = 'demo-node-connected';

async function httpJson(method, path, body) {
  const headers = { 'Content-Type': 'application/json' };
  if (API_KEY) headers['X-Api-Key'] = API_KEY;
  if (path.includes('/api/intents')) headers['X-Namespace-ID'] = NS;
  const res = await fetch(`${SERVER_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${path}`);
  return res;
}

async function setup() {
  // best-effort cleanup of prior runs
  try { await httpJson('DELETE', '/api/namespaces', { namespace_id: NS }); } catch {}
  await httpJson('POST', '/api/namespaces', { namespace_id: NS, description: 'node connected demo' });
  for (const [id, phrases] of [
    ['list_subscriptions',  ['list my subscriptions', 'show all subscriptions']],
    ['cancel_subscription', ['cancel subscription', 'stop my subscription']],
    ['greeting',            ['hello', 'hi there']],
  ]) {
    await httpJson('POST', '/api/intents', { id, phrases });
  }
}

async function main() {
  console.log(`─── 1. Setup namespace + intents on ${SERVER_URL} ─────────────`);
  try {
    await setup();
  } catch (err) {
    console.error(`  ✗ server unreachable: ${err.message}`);
    console.error(`  Start: ../target/release/server --port ${SERVER_URL.split(':').pop()} --no-browser`);
    process.exit(1);
  }
  console.log(`  ✓ namespace '${NS}' seeded with 3 intents`);

  console.log('\n─── 2. Connect Engine to server ──────────────────────────────');
  const engine = new Engine({
    serverUrl: SERVER_URL,
    apiKey: API_KEY,
    subscribe: [NS],
    tickIntervalSecs: 5,
  });
  const ns = engine.namespace(NS);
  console.log(`  connected. version = ${ns.version()}, intents = ${ns.intentCount()}`);

  const query = 'drop my subscription right now';

  console.log('\n─── 3. Resolve a query ───────────────────────────────────────');
  let matches = ns.resolve(query);
  const initial = matches[0]?.id ?? '(none)';
  console.log(`  query  : "${query}"`);
  if (matches.length) {
    console.log(`  routed : ${initial} (score: ${matches[0].score.toFixed(2)})`);
  }

  console.log('\n─── 4. Strict mode: library mutations refused ───────────────');
  console.log('  Connected libraries are READ-ONLY caches. ns.correct(...) throws:');
  const wrong = initial !== '(none)' ? initial : 'list_subscriptions';
  try {
    ns.correct(query, wrong, 'cancel_subscription');
    console.log('    UNEXPECTED: correct() succeeded');
  } catch (e) {
    console.log(`    ${e.message}  ← refused, as designed.`);
  }

  console.log('\n─── 5. Apply correction via the server\'s HTTP API ───────────');
  const apiUrl = SERVER_URL;
  const headers = { 'Content-Type': 'application/json', 'X-Namespace-ID': NS };
  if (apiKey) headers['X-Api-Key'] = apiKey;
  try {
    const resp = await fetch(`${apiUrl}/api/correct`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ query, wrong_intent: wrong, right_intent: 'cancel_subscription' }),
    });
    console.log(`  ✓ POST /api/correct → HTTP ${resp.status}`);
  } catch (e) {
    console.log(`  ✗ HTTP correct failed: ${e.message}`);
  }

  console.log('\n─── 6. Wait for sync tick to pull the change ────────────────');
  const vBefore = ns.version();
  for (let i = 0; i < 8; i++) {
    await new Promise(r => setTimeout(r, 1000));
    if (ns.version() > vBefore) {
      console.log(`  ✓ pulled v${ns.version()} from server (was v${vBefore})`);
      break;
    }
  }

  console.log('\nDone.');
}

main().catch(err => {
  console.error(err.message);
  process.exit(1);
});
