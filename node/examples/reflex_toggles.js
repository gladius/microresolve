/**
 * reflex_toggles.js — smoke demo for updateNamespace / namespaceInfo.
 *
 * Shows how to read and flip individual reflex-layer toggles on a namespace.
 * No server or API key required.
 *
 * Run: node examples/reflex_toggles.js
 */

const { MicroResolve } = require('..');

const engine = new MicroResolve();
const ns = engine.namespace('tools');

// Seed a couple of intents so the namespace is non-trivial.
ns.addIntent('deploy',   ['deploy to production', 'push to prod', 'ship it']);
ns.addIntent('rollback', ['rollback release', 'revert deploy', 'undo push']);

// ── Read the current toggle state ─────────────────────────────────────────────
const before = ns.namespaceInfo();
console.log('Before:', before);
console.assert(before.l0Enabled   === true, 'l0Enabled should default to true');
console.assert(before.l1Morphology === true, 'l1Morphology should default to true');
console.assert(before.l1Synonym    === true, 'l1Synonym should default to true');
console.assert(before.l1Abbreviation === true, 'l1Abbreviation should default to true');

// ── Disable abbreviation expansion (pr → pull request not wanted here) ────────
ns.updateNamespace({ l1Abbreviation: false });

const after = ns.namespaceInfo();
console.log('After disabling l1Abbreviation:', after);
console.assert(after.l0Enabled      === true,  'l0Enabled should be unchanged');
console.assert(after.l1Abbreviation === false, 'l1Abbreviation should be off');

// ── Re-enable and verify round-trip ──────────────────────────────────────────
ns.updateNamespace({ l1Abbreviation: true, name: 'Tool Router' });
const final = ns.namespaceInfo();
console.log('After re-enable:', final);
console.assert(final.l1Abbreviation === true,        'l1Abbreviation should be back on');
console.assert(final.name           === 'Tool Router', 'name should be updated');

console.log('Done — all assertions passed.');
