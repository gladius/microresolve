"""Smoke demo: reflex-layer toggle API (update_namespace / namespace_info).

Shows how to read the current toggle state and disable individual L1 layers
on a namespace. No server or API key required.
"""

from microresolve import MicroResolve

engine = MicroResolve()
ns = engine.namespace("tools")

# Seed a couple of intents so the namespace is non-trivial.
ns.add_intent("deploy", ["deploy to production", "push to prod", "ship it"])
ns.add_intent("rollback", ["rollback release", "revert deploy", "undo push"])

# ── Read the current toggle state ──────────────────────────────────────────────
info = ns.namespace_info()
print("Before:", info)
assert info.l0_enabled is True
assert info.l1_morphology is True
assert info.l1_synonym is True
assert info.l1_abbreviation is True

# ── Disable abbreviation expansion (pr → pull request not wanted here) ─────────
ns.update_namespace({"l1_abbreviation": False})

info2 = ns.namespace_info()
print("After disabling l1_abbreviation:", info2)
assert info2.l0_enabled is True          # unchanged
assert info2.l1_abbreviation is False    # toggled off

# ── Re-enable and verify round-trip ───────────────────────────────────────────
ns.update_namespace({"l1_abbreviation": True, "name": "Tool Router"})
info3 = ns.namespace_info()
print("After re-enable:", info3)
assert info3.l1_abbreviation is True
assert info3.name == "Tool Router"

print("Done — all assertions passed.")
