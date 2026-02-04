# Flower Federated AI Glossary

This directory contains the source for Flower’s glossary on **Federated AI** (federated learning, federated analytics, federated evaluation) and key Flower concepts.

## Goals

- **Accurate**: definitions are correct under realistic threat models and deployment constraints.
- **Concept-first**: prefer core federated AI concepts; keep “how-to” explainers in the docs.
- **Consistent**: every entry follows the same structure and tone.
- **Cross-linked**: related terms connect to each other.
- **Maintainable**: lightweight checks prevent drift.

## Flower terminology

This glossary uses **Flower terminology** as the reference implementation for deployed federated systems (SuperLink/SuperNode/SuperExec; ServerApp/ClientApp; Strategy). We keep the main definition general; Flower-specific mapping belongs in `## In Flower`.

## Entry template

Create a new file as `glossary/<kebab-case-slug>.mdx`:

```mdx
---
title: "Term Name"
description: "One-sentence description used in listings."
date: "YYYY-MM-DD"
author:
  name: "Your Name"
  position: "Your Role"
  website: "https://example.com" # optional
  github: "github.com/handle"    # optional
related:
  - text: "Related Term"
    link: "/glossary/related-term"
---

One-paragraph definition in plain language.

## Why it matters
Explain why this term matters for practitioners.

## In federated settings
Explain how the concept changes in federated learning/analytics/evaluation.

## Common pitfalls
List the most common ways this concept is misunderstood or misapplied.

## In Flower
Explain how Flower uses/implements the concept (omit if not relevant).
```

## Writing rules (high signal)

- Don’t imply **privacy is automatic**. If you mention privacy, clarify conditions (e.g., secure aggregation/DP, threat model).
- Prefer **model update(s)** over “weights” unless you specifically mean full model parameters.
- Keep entries roughly **200–500 words**. If it needs more, link to a docs page instead.
- Avoid marketing language and unverifiable claims (“best”, “magic”, “guaranteed”).
- In `## In Flower`, make the **first mention** of any Flower term an inline link to its glossary entry.
