
"""Lightweight regex-based guardrail checks."""
import re
import time

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|safety|your)\s+(instructions|rules|guidelines)",
    r"you\s+are\s+(now\s+)?(DAN|jailbreak|unfiltered|unrestricted)",
    r"pretend\s+you\s+(have\s+no|are\s+an?\s+unfiltered)",
    r"developer\s+mode",
    r"(forget|disregard|override)\s+(all\s+)?(rules|restrictions|guidelines|safety)",
]

PII_PATTERNS = [
    r"\b(SSN|social\s+security\s+number)\b",
    r"\b(credit\s+card\s+number|account\s+number)\b",
    r"\b(home\s+address(es)?|phone\s+number)\b",
    r"\bpatient\s+\w+.*?(SSN|diagnosis|record)\b",
    r"\b(reveal|expose|leak|show)\s+(the\s+)?(privileged|confidential|private)\b",
    r"\baccount\s+ID\s+\d+",
]

TOPIC_PATTERNS = [
    r"\b(launder\s+money|insider\s+trading|ponzi\s+scheme)\b",
    r"\b(forge|forging|fabricat)\w*\s+(a\s+)?(prescription|document|report|citation)",
    r"\b(destroy\s+evidence|tamper\s+with|obstruct)\b",
    r"\b(bribe|threaten\s+a\s+witness|harass)\b",
    r"\b(lethal|controlled\s+substances?\s+without)\b",
    r"\b(phishing|fake\s+(insurance|audit|clinical|notarized))\b",
    r"\b(manipulate\s+stock|wash\s+trading|front-?run)\b",
    r"\b(shell\s+company|evade\s+sanctions)\b",
]


def _match(text, patterns):
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def run_guardrails(prompt):
    start = time.perf_counter()
    inj = _match(prompt, INJECTION_PATTERNS)
    pii = _match(prompt, PII_PATTERNS)
    top = _match(prompt, TOPIC_PATTERNS)
    is_safe = not any([inj, pii, top])
    latency = (time.perf_counter() - start) * 1000
    return {
        "is_safe": is_safe,
        "injection": inj,
        "pii": pii,
        "topic": top,
        "latency_ms": latency,
    }
