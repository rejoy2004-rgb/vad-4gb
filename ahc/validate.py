"""Check a submission against the arena's stated field rules before uploading.

A rejected file doesn't cost a run, but it does cost time on the day, and every
rule below is one the brief explicitly calls out as a thing that catches people
out.
"""
from __future__ import annotations

import argparse
import json
import os

from .config import CLASSES

VALID = set(CLASSES) - {"normal"}
MAX_BYTES = 5 * 1024 * 1024


def validate(sub: dict, levels: dict[str, int], path=None) -> list[str]:
    errs: list[str] = []

    if "predictions" not in sub:
        return ["missing required top-level field 'predictions'"]

    if path and os.path.getsize(path) > MAX_BYTES:
        errs.append(f"file is {os.path.getsize(path)/1e6:.1f} MB, limit is 5 MB")

    seen = set()
    for p in sub["predictions"]:
        vid = p.get("video_id")
        if not vid:
            errs.append("a prediction has no video_id")
            continue
        if vid in seen:
            errs.append(f"{vid}: appears more than once")
        seen.add(vid)
        if levels and vid not in levels:
            errs.append(f"{vid}: not in the manifest")
        if "events" not in p:
            errs.append(f"{vid}: missing 'events'")
        if "runtime_metadata" not in p:
            errs.append(f"{vid}: missing 'runtime_metadata' (required on every video)")
        else:
            rm = p["runtime_metadata"]
            for mr in rm.get("model_runtimes", []):
                tot, calls = mr.get("total_time_ms"), mr.get("call_count")
                avg = mr.get("average_time_ms")
                if tot is not None and calls and avg is not None:
                    exp = tot / calls
                    if exp > 0 and abs(avg - exp) / exp > 0.02:
                        errs.append(f"{vid}: model_runtimes average {avg} != total/calls {exp:.1f} "
                                    f"(must agree within 2%)")
                ct = mr.get("call_times_ms")
                if ct is not None and calls is not None and len(ct) != calls:
                    errs.append(f"{vid}: call_times_ms has {len(ct)} entries, call_count is {calls}")

        lvl = levels.get(vid, 1)
        for i, e in enumerate(p.get("events", [])):
            tag = f"{vid} event[{i}]"
            cn = e.get("class_name")
            if cn == "normal":
                errs.append(f"{tag}: class_name 'normal' is rejected - use \"events\": []")
            elif cn not in VALID:
                errs.append(f"{tag}: unknown class_name {cn!r}")
            s, t = e.get("start_time_sec"), e.get("end_time_sec")
            if lvl == 1:
                if s is not None or t is not None:
                    errs.append(f"{tag}: level 1 timestamps must be null")
            else:
                if s is None or t is None:
                    errs.append(f"{tag}: level {lvl} requires start and end")
                else:
                    if s < 0:
                        errs.append(f"{tag}: start_time_sec must be >= 0")
                    if t <= s:
                        errs.append(f"{tag}: end ({t}) must be greater than start ({s})")
            ex = e.get("explanation")
            if ex is not None and not 20 <= len(ex) <= 500:
                errs.append(f"{tag}: explanation is {len(ex)} chars, must be 20-500")

    if levels:
        missing = sorted(set(levels) - seen)
        if missing:
            errs.append(f"note: {len(missing)} manifest videos unanswered "
                        f"(scored as normal): {', '.join(missing[:8])}"
                        + (" ..." if len(missing) > 8 else ""))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    from .infer import load_levels

    sub = json.load(open(args.submission, encoding="utf-8"))
    errs = validate(sub, load_levels(args.manifest), args.submission)
    if not errs:
        print(f"OK - {len(sub['predictions'])} videos, "
              f"{os.path.getsize(args.submission)/1024:.0f} KB")
        return
    hard = [e for e in errs if not e.startswith("note:")]
    for e in errs:
        print(("  " if e.startswith("note:") else "ERROR ") + e)
    raise SystemExit(1 if hard else 0)


if __name__ == "__main__":
    main()
