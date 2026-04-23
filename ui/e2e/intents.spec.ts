import { test, expect } from '@playwright/test';

const API = 'http://127.0.0.1:3001/api';
const NS = 'ui-e2e-test';
const INTENT_ID = 'demo';

async function api(method: string, path: string, body?: unknown, ns?: string) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (ns && ns !== 'default') headers['X-Workspace-ID'] = ns;
  const opts: RequestInit = { method, headers };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(`${API}${path}`, opts);
  const text = await res.text();
  return { status: res.status, ok: res.ok, text };
}

test.beforeAll(async () => {
  // Tear down any leftover from a prior run
  await api('DELETE', '/namespaces', { namespace_id: NS });

  const r = await api('POST', '/namespaces', { namespace_id: NS, description: 'ui e2e' });
  if (!r.ok) throw new Error(`namespace create failed: ${r.status} ${r.text}`);

  const r2 = await api('POST', '/intents', { id: INTENT_ID, phrases: ['hello world'] }, NS);
  if (!r2.ok) throw new Error(`intent create failed: ${r2.status} ${r2.text}`);

  const r3 = await api('PATCH', '/settings', { selected_namespace_id: NS });
  if (!r3.ok) throw new Error(`settings patch failed: ${r3.status} ${r3.text}`);
});

test.afterAll(async () => {
  await api('PATCH', '/settings', { selected_namespace_id: 'default' });
  await api('DELETE', '/namespaces', { namespace_id: NS });
});

test('IntentsPage: edit Details and verify persistence after reload', async ({ page }) => {
  // Capture network for the PATCH we expect
  const patchRequests: { url: string; method: string; postData: string | null }[] = [];
  page.on('request', req => {
    if (req.url().includes('/api/intents/')) {
      patchRequests.push({ url: req.url(), method: req.method(), postData: req.postData() });
    }
  });

  await page.goto('/intents');

  // Wait until our seeded intent appears in the sidebar list
  const intentRow = page.locator('text=' + INTENT_ID).first();
  await expect(intentRow).toBeVisible({ timeout: 10_000 });
  await intentRow.click();

  // Details tab is the default — wait for its key inputs.
  const promptTextarea = page.locator('textarea[placeholder^="You are helping"]');
  await expect(promptTextarea).toBeVisible();
  const personaInput = page.locator('input[placeholder^="e.g. professional"]');
  await expect(personaInput).toBeVisible();
  const guardrailInput = page.locator('input[placeholder="Add a guardrail..."]');
  await expect(guardrailInput).toBeVisible();

  // Fill the form
  const NEW_PROMPT = 'always be concise and friendly to the user';
  const NEW_PERSONA = 'warm professional';
  const NEW_GUARDRAIL = 'no profanity';

  await promptTextarea.fill(NEW_PROMPT);
  await personaInput.fill(NEW_PERSONA);
  await guardrailInput.fill(NEW_GUARDRAIL);
  await page.locator('button', { hasText: /^Add$/ }).click();

  // Save (button only shows when dirty)
  const saveBtn = page.locator('button', { hasText: /^Save$/ });
  await expect(saveBtn).toBeVisible({ timeout: 3_000 });
  await saveBtn.click();

  // Save button disappears when not dirty anymore
  await expect(saveBtn).toBeHidden({ timeout: 5_000 });

  // Reload — true persistence test
  await page.reload();

  // Reselect the intent
  await page.locator('text=' + INTENT_ID).first().click();

  // Verify the form re-populated with our edits
  await expect(page.locator('textarea[placeholder^="You are helping"]'))
    .toHaveValue(NEW_PROMPT, { timeout: 5_000 });
  await expect(page.locator('input[placeholder^="e.g. professional"]'))
    .toHaveValue(NEW_PERSONA);
  // Guardrails render as editable inputs (one per item), not plain text
  const guardrailInputs = page.locator('input').filter({ hasText: '' });
  await expect(page.locator(`input[value="${NEW_GUARDRAIL}"]`)).toBeVisible();

  // Verify the network call used PATCH not the old per-field POSTs
  const patches = patchRequests.filter(r => r.method === 'PATCH' && r.url.includes(`/intents/${INTENT_ID}`));
  expect(patches.length).toBeGreaterThanOrEqual(1);
  const patchBody = JSON.parse(patches[0].postData || '{}');
  expect(patchBody).toHaveProperty('instructions');
  expect(patchBody).toHaveProperty('persona');
  expect(patchBody).toHaveProperty('guardrails');

  // Confirm zero legacy POSTs were made
  const legacyPosts = patchRequests.filter(r =>
    r.method === 'POST' && /\/intents\/(description|instructions|persona|guardrails|target|type|delete|phrase|phrase\/remove)$/.test(r.url)
  );
  expect(legacyPosts).toEqual([]);
});

test('IntentsPage: add and remove a phrase via new RESTful endpoints', async ({ page }) => {
  const reqs: { url: string; method: string }[] = [];
  page.on('request', req => {
    if (req.url().includes('/api/intents/')) reqs.push({ url: req.url(), method: req.method() });
  });

  await page.goto('/intents');
  await page.locator('text=' + INTENT_ID).first().click();

  // Switch to Phrases tab (label is "Phrases" + count number, e.g. "Phrases1")
  await page.locator('button').filter({ hasText: /^Phrases\d*$/ }).first().click();

  // Add a phrase
  const phraseInput = page.locator('input[placeholder^="Type a phrase"]');
  await expect(phraseInput).toBeVisible();
  const NEW_PHRASE = 'goodbye world';
  await phraseInput.fill(NEW_PHRASE);
  await phraseInput.press('Enter');

  // Wait for it to appear in the phrase list
  await expect(page.locator('text=' + NEW_PHRASE)).toBeVisible({ timeout: 5_000 });

  // Verify the request hit the new RESTful URL, not legacy /intents/phrase
  const phraseAdd = reqs.find(r => r.method === 'POST' && r.url.includes(`/intents/${INTENT_ID}/phrases`));
  expect(phraseAdd, 'POST /intents/{id}/phrases should be called').toBeTruthy();
  const legacyPhrase = reqs.find(r => r.method === 'POST' && r.url.endsWith('/api/intents/phrase'));
  expect(legacyPhrase, 'legacy POST /intents/phrase must NOT be called').toBeFalsy();
});
