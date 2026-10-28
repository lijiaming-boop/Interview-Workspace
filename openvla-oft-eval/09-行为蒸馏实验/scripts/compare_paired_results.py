import argparse
import json
import math
import re
from pathlib import Path


def results(paths):
    values = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        values.extend(value == "True" for value in re.findall(r"Success:\s+(True|False)", text))
    return values


parser = argparse.ArgumentParser()
parser.add_argument("--base", nargs="+", required=True)
parser.add_argument("--candidate", nargs="+", required=True)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--expected", type=int, required=True)
parser.add_argument("--min-rate", type=float, default=0.96)
parser.add_argument("--max-regressions", type=int, default=1)
parser.add_argument("--require-p", type=float)
args = parser.parse_args()

base, candidate = results(args.base), results(args.candidate)
if len(base) != args.expected or len(candidate) != args.expected:
    raise SystemExit(
        f"FAIL: expected {args.expected} results, got base={len(base)}, candidate={len(candidate)}"
    )

wins = [args.offset + i for i, (b, c) in enumerate(zip(base, candidate)) if not b and c]
regressions = [args.offset + i for i, (b, c) in enumerate(zip(base, candidate)) if b and not c]
base_failures = [args.offset + i for i, ok in enumerate(base) if not ok]
candidate_failures = [args.offset + i for i, ok in enumerate(candidate) if not ok]
report = {
    "base": f"{sum(base)}/{len(base)}",
    "candidate": f"{sum(candidate)}/{len(candidate)}",
    "base_rate": sum(base) / len(base),
    "candidate_rate": sum(candidate) / len(candidate),
    "fixed_states": wins,
    "regressed_states": regressions,
    "base_failures": base_failures,
    "candidate_failures": candidate_failures,
}
discordant = len(wins) + len(regressions)
if discordant:
    tail = sum(math.comb(discordant, i) for i in range(min(len(wins), len(regressions)) + 1))
    report["mcnemar_exact_two_sided_p"] = min(1.0, 2.0 * tail / (2**discordant))
else:
    report["mcnemar_exact_two_sided_p"] = 1.0
print(json.dumps(report, indent=2, ensure_ascii=False))

passed = (
    report["candidate_rate"] >= args.min_rate
    and report["candidate_rate"] >= report["base_rate"]
    and len(regressions) <= args.max_regressions
    and len(wins) > len(regressions)
    and (args.require_p is None or report["mcnemar_exact_two_sided_p"] < args.require_p)
)
print("PASS" if passed else "FAIL")
raise SystemExit(0 if passed else 2)
