# Security Policy

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

To report a suspected security issue in MicroResolve, please use GitHub's
private vulnerability reporting:

👉 https://github.com/gladius/microresolve/security/advisories/new

Alternatively, email the maintainer directly at: **gladius.thayalarajan@gmail.com**

Please include:
- A description of the issue and the component affected (Rust library,
  Python / Node bindings, HTTP server, etc.)
- Steps to reproduce or a proof-of-concept
- The version / commit SHA you tested against
- Your assessment of the impact (data exposure, DoS, etc.)

---

## What to expect

- **Acknowledgement:** within 72 hours.
- **Triage and initial assessment:** within 7 days.
- **Fix and disclosure:** coordinated with you. For critical issues, a patched
  release typically lands within 14 days.

You will be credited in the release notes and the advisory (unless you prefer
to remain anonymous).

---

## Supported versions

MicroResolve is pre-1.0. Only the latest published version of each package
(`microresolve` on crates.io, PyPI, npm) receives security fixes.

| Version  | Supported          |
| -------- | ------------------ |
| latest   | :white_check_mark: |
| previous | :x:                |

---

## Scope

In scope:
- The `microresolve` Rust crate, Python package, Node package, and HTTP
  server binary.
- Bundled dependencies with known vulnerabilities (we will update).

Out of scope:
- Vulnerabilities in third-party integrations unless caused by MicroResolve.
- Social engineering, phishing, or physical attacks.
- DoS via resource exhaustion from adversarial inputs — MicroResolve is a
  lexical engine; callers are responsible for input size limits.
