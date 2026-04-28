/**
 * MicroResolve + Express: production intent classification endpoint.
 *
 * Setup: cd node && npm install && npx napi build --release
 * Run:   cd examples && npm install express && node node_express.js
 * Test:  curl -X POST localhost:3000/route -H 'Content-Type: application/json' -d '{"query": "cancel my order"}'
 *
 * After publishing to npm: just `npm install microresolve` and change the require to 'microresolve'.
 */

const express = require('express');

// Local build: require from the node/ build output
// After npm publish: change to require('microresolve')
const path = require('path');
const { Router } = require(path.join(__dirname, '..', '..', 'node', 'microresolve.node'));

const app = express();
app.use(express.json());

// Initialize router
const router = new Router();
router.beginBatch();
router.addIntent('cancel_order', ['cancel my order', 'stop my order', 'I want to cancel']);
router.addIntent('track_order', ['where is my package', 'track my order', 'shipping status']);
router.addIntent('refund', ['I want a refund', 'get my money back', 'return and refund']);
router.addIntent('contact_human', ['talk to a person', 'speak to agent', 'human representative']);
router.endBatch();

app.post('/route', (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).json({ error: 'query required' });
  const results = router.route(query);
  if (!results.length) return res.status(404).json({ error: 'no match' });
  res.json({ intent: results[0].id, score: results[0].score });
});

app.post('/route_multi', (req, res) => {
  const { query, threshold = 0.3 } = req.body;
  if (!query) return res.status(400).json({ error: 'query required' });
  res.json(router.routeMulti(query, threshold));
});

app.post('/learn', (req, res) => {
  const { query, intent_id } = req.body;
  router.learn(query, intent_id);
  res.json({ status: 'learned' });
});

app.get('/intents', (_req, res) => {
  res.json(router.intentIds());
});

app.listen(3000, () => console.log('MicroResolve listening on :3000'));
