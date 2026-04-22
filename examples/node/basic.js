/**
 * Basic MicroResolve usage: routing, multi-intent, learning, export/import, discovery.
 *
 * Run: node basic.js
 */

const path = require('path');
// Local build — after npm publish: require('microresolve')
const { Router } = require(path.join(__dirname, '..', '..', 'node', 'microresolve.node'));

// --- Setup ---
const r = new Router();
r.beginBatch();
r.addIntent('cancel_order', ['cancel my order', 'I want to cancel', 'stop my order from shipping']);
r.addIntent('track_order', ['where is my package', 'track my order', 'shipping status update']);
r.addIntent('refund', ['I want a refund', 'get my money back', 'return and refund']);
r.endBatch();

// --- Single routing ---
console.log('=== Single routing ===');
const results = r.route('I need to cancel something');
results.forEach(res => console.log(`  ${res.id} (score: ${res.score.toFixed(2)})`));

// --- Multi-intent ---
console.log('\n=== Multi-intent ===');
const multi = r.routeMulti('cancel my order and give me a refund');
multi.confirmed.forEach(i => console.log(`  ${i.id} (score: ${i.score.toFixed(2)}, type: ${i.intentType})`));

// --- Learning ---
console.log('\n=== Learning ===');
const before = r.route('stop charging me');
console.log(`  before: ${before[0]?.id || 'no match'} (${before[0]?.score.toFixed(2) || ''})`);

r.learn('stop charging me', 'cancel_order');

const after = r.route('stop charging me');
console.log(`  after:  ${after[0].id} (${after[0].score.toFixed(2)})`);

// --- Export / Import ---
console.log('\n=== Export/Import ===');
const json = r.exportJson();
console.log(`  exported: ${json.length} bytes`);

const r2 = Router.importJson(json);
const imported = r2.route('cancel this');
console.log(`  imported route: ${imported[0].id}`);

// --- Discovery ---
console.log('\n=== Discovery ===');
const queries = [];
const seeds = [
  'cancel my order', 'I want to cancel', 'stop my order',
  'cancel the purchase', 'cancel it please', 'undo my order',
  'where is my package', 'track order', 'shipping update',
  'track my delivery', 'order tracking', 'delivery status',
];
for (let i = 0; i < 20; i++) queries.push(...seeds);

const clusters = Router.discover(queries);
console.log(`  discovered ${clusters.length} clusters from ${queries.length} queries`);
clusters.forEach(c => console.log(`    ${c.name} (size: ${c.size}, terms: ${c.topTerms.slice(0, 3)})`));

console.log('\nAll working!');
