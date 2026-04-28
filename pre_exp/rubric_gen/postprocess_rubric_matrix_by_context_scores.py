import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

try:
    from .api_kimi import call_kimi
    from .rubric_context_consistency_analysis import normalize_rubric_score
except ImportError:
    from api_kimi import call_kimi
    from rubric_context_consistency_analysis import normalize_rubric_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
SYSTEM_PROMPT = "You are a meticulous rubric deduplication judge. Return only valid JSON."


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(x, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_rubrics(s):
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        str(x.get("criterion", "")).strip()
        for x in data
        if isinstance(x, dict) and str(x.get("criterion", "")).strip()
    ]


def trunc(s, n):
    s = str(s).strip()
    if n <= 0 or len(s) <= n:
        return s
    k = n // 2
    return s[:k].rstrip() + "\n\n...[truncated]...\n\n" + s[-(n - k):].lstrip()


def fmt_scores(xs):
    return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


def parse_groups(s, n):
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    groups = json.loads(s).get("merge_groups")
    if not isinstance(groups, list):
        raise ValueError("missing merge_groups")

    seen, out = set(), []
    for g in groups:
        cur = []
        for cid in g:
            if not isinstance(cid, int) or cid < 1 or cid > n or cid in seen:
                raise ValueError(f"bad cluster id: {cid}")
            seen.add(cid)
            cur.append(cid)
        out.append(sorted(cur))

    if seen != set(range(1, n + 1)):
        raise ValueError("merge_groups must cover every cluster once")
    return out


def flagged(metrics, threshold, inclusive):
    ans = defaultdict(lambda: defaultdict(set))
    for r in metrics.get("records", []):
        t = r.get("t_obs")
        if isinstance(t, (int, float)) and (t > threshold or (inclusive and t == threshold)):
            q = int(r["question_index"])
            u = int(r["unique_rubric_index"])
            ans[q][u].add(str(r["gen_model"]))
    return ans


def collect(q, u, models, match, rubric_item, result):
    rubric_lists = [parse_rubrics(x) for x in rubric_item.get("rubric_responses", [])]
    occs = []

    for si, mapping in enumerate(match.get("sample_match_indices", []), 1):
        local_rubrics = rubric_lists[si - 1] if si <= len(rubric_lists) else []
        for li, mapped in enumerate(mapping):
            if int(mapped) != u:
                continue

            occ = {
                "sample": si,
                "local": li,
                "text": local_rubrics[li] if li < len(local_rubrics) else "",
                "scores": {},
                "evidence": {},
            }
            for m in models:
                scores, evs = [], []
                for trial in result.get(str(q), {}).get(m, {}).get(str(si), []):
                    if li >= len(trial):
                        continue
                    sc = normalize_rubric_score(trial[li])
                    if isinstance(sc, float) and math.isnan(sc):
                        continue
                    scores.append(float(sc))
                    evs.append(str(trial[li].get("evidence", "")))
                occ["scores"][m] = scores
                occ["evidence"][m] = evs
            occs.append(occ)

    return occs


def clusters(occs, models, n_trials):
    buckets = defaultdict(list)
    for o in occs:
        sig = []
        for m in models:
            s = [round(x, 6) for x in o["scores"].get(m, [])[:n_trials]]
            sig += s + [None] * (n_trials - len(s))
        buckets[tuple(sig)].append(o)

    ordered = sorted(buckets.values(), key=lambda g: min((o["sample"], o["local"]) for o in g))
    return [
        {"id": i, "occs": sorted(g, key=lambda o: (o["sample"], o["local"]))}
        for i, g in enumerate(ordered, 1)
    ]


def build_prompt(question, cs, models, responses, n_trials, max_resp, max_ev):
    lines = [
        "## Background Information",
        "I am running an experiment on generated rubrics for LLM-as-a-judge evaluation.",
        "For each question, multiple rubric lists are sampled, used for judging several gen_model responses, deduplicated into unique rubrics, and stored in a sample-by-unique-rubric matrix.",
        "The first deduplication pass may falsely merge related but non-equivalent rubrics. High T_obs flags such possible false merges.",
        "Initial clusters below are formed by concatenating each local rubric's judge scores across selected gen_models; identical score signatures share a cluster.",
        f"Missing judge scores are missing values in the signature. Expected samples per gen_model: {n_trials}.",
        "",
        "## Task",
        "Merge clusters only when their rubrics evaluate the same aspect of the response at the same requirement level under the original question, even if standalone wording differs in breadth or specificity.",
        "Keep clusters split when score differences and evidence indicate a real semantic difference rather than judge noise.",
        "",
        "## Output Format",
        'Return only valid JSON: {"merge_groups": [[1], [2, 3]]}. Every cluster id must appear exactly once.',
        "",
        "## Question",
        question,
        "",
        "## Candidate Clusters",
    ]

    for c in cs:
        lines += ["", f"### Cluster {c['id']}"]
        lines += [f"+ rubric{i}: {o['text']}" for i, o in enumerate(c["occs"], 1)]

    for ri, m in enumerate(models, 1):
        lines += ["", f"## Response {ri}: gen_model = {m}", "```text", trunc(responses.get(m, ""), max_resp), "```"]
        for c in cs:
            lines += ["", f"### Cluster {c['id']}"]
            for i, o in enumerate(c["occs"], 1):
                s = o["scores"].get(m, [])
                lines += [
                    f"#### cluster{c['id']}-rubric{i}",
                    f"+ rubric: {o['text']}",
                    f"+ score_list: {fmt_scores(s) if s else 'missing'} ({len(s)}/{n_trials} valid judge samples)",
                ]
                for j, ev in enumerate(o["evidence"].get(m, [])[:n_trials], 1):
                    score = f"{s[j - 1]:.3f}" if j - 1 < len(s) else "missing"
                    lines.append(f"+ judge{j}: result:{score}; evidence:{trunc(ev, max_ev)}")

    return "\n".join(lines)


def choose_criterion(group, old):
    exact = next((o["text"] for o in group if o["text"].strip() == old.strip()), None)
    return exact or next((o["text"] for o in group if o["text"].strip()), old)


def decide(prompt, cs, dry_run, on_error):
    if len(cs) <= 1:
        return [cs[0]["occs"]] if cs else [], {"mode": "single_cluster"}

    try:
        if dry_run:
            mgs = [[c["id"]] for c in cs]
            mode = "dry_run"
        else:
            mgs = parse_groups(call_kimi(prompt, system_prompt=SYSTEM_PROMPT), len(cs))
            mode = "llm"
    except Exception as e:
        if on_error == "fail":
            raise
        mgs, mode = [[c["id"]] for c in cs], f"llm_error_keep_split: {e}"

    by_id = {c["id"]: c for c in cs}
    groups = [
        sorted(
            [o for cid in mg for o in by_id[cid]["occs"]],
            key=lambda o: (o["sample"], o["local"]),
        )
        for mg in mgs
    ]
    return groups, {"mode": mode, "merge_groups": mgs}


def apply_split(match, u, groups):
    if len(groups) <= 1:
        return None

    uniq = match["unique_rubrics"]
    old = str(match["unique_rubrics"][u].get("criterion", ""))
    ordered = sorted(groups, key=lambda g: min((o["sample"], o["local"]) for o in g))
    new_ids = [u]

    for g in ordered[1:]:
        new_id = len(uniq)
        new_ids.append(new_id)
        uniq.append({"rubric_index": new_id + 1, "criterion": choose_criterion(g, old)})
        for o in g:
            match["sample_match_indices"][o["sample"] - 1][o["local"]] = new_id

    match["num_unique_rubrics"] = len(uniq)
    return {"kept_original_index": u, "new_indices": new_ids}


def rebuild_matrix(match):
    n = len(match["unique_rubrics"])
    match["num_unique_rubrics"] = n
    match["matrix"] = [
        [1 if i in set(row) else 0 for i in range(n)]
        for row in match.get("sample_match_indices", [])
    ]


def validate(matches):
    for item in matches:
        n = item["num_unique_rubrics"]
        for row, mapping in zip(item["matrix"], item["sample_match_indices"]):
            active = {i for i, v in enumerate(row) if v}
            mapping_ok = not mapping or (min(mapping) >= 0 and max(mapping) < n)
            if len(row) != n or not mapping_ok or active != set(mapping):
                raise ValueError(f"bad matrix for q={item['question_index']}")


def postprocess(args):
    metrics = load(args.metrics_path)
    matches = load(args.match_path)
    rubric_data = load(args.rubric_path)
    result = load(args.result_path)
    responses = load(args.response_path)

    by_q = {int(x["question_index"]): x for x in matches}
    rub_by_q = {int(x["question_index"]): x for x in rubric_data}
    flags = flagged(metrics, args.t_obs_threshold, args.threshold_inclusive)
    locks = {q: Lock() for q in by_q}
    old_total = sum(x["num_unique_rubrics"] for x in matches)
    decisions = {}
    rebuild = defaultdict(lambda: {"split_unique_indices": {}})

    def work(q, u):
        models = sorted(result.get(str(q), {})) if args.score_gen_models == "all" else sorted(flags[q][u])
        occs = collect(q, u, models, by_q[q], rub_by_q[q], result)
        cs = clusters(occs, models, args.n_trials)
        response_item = responses[q] if q < len(responses) else {}
        prompt = build_prompt(
            str(by_q[q].get("question", "")),
            cs,
            models,
            response_item,
            args.n_trials,
            args.max_response_chars,
            args.max_evidence_chars,
        )
        groups, info = decide(prompt, cs, args.dry_run, args.on_llm_error)

        with locks[q]:
            split_info = apply_split(by_q[q], u, groups)

        decision = {
            "question_index": q,
            "old_unique_index": u,
            "flagged_gen_models": sorted(flags[q][u]),
            "score_gen_models_used": models,
            "num_occurrences": len(occs),
            "num_initial_clusters": len(cs),
            "will_split": split_info is not None,
            **info,
        }
        return f"{q}:{u}", decision, split_info

    tasks = [
        (q, u)
        for q in sorted(flags)
        for u in sorted(flags[q])
        if q in by_q and q in rub_by_q
    ]
    total_tasks = len(tasks)
    print(f"[postprocess] queued {total_tasks} (question_index, unique_rubric_index) pairs", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, q, u) for q, u in tasks]
        for completed, fut in enumerate(as_completed(futures), 1):
            key, dec, split = fut.result()
            decisions[key] = dec
            print(f"[postprocess] completed {completed}/{total_tasks}: {key}", flush=True)
            if split:
                q_key = str(dec["question_index"])
                u_key = str(dec["old_unique_index"])
                rebuild[q_key]["split_unique_indices"][u_key] = split

    for m in matches:
        rebuild_matrix(m)
    validate(matches)

    summary = {
        "num_flagged_question_unique": len(tasks),
        "num_split_unique": sum(len(x["split_unique_indices"]) for x in rebuild.values()),
        "old_total_unique_rubrics": old_total,
        "new_total_unique_rubrics": sum(x["num_unique_rubrics"] for x in matches),
    }
    audit = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "summary": summary,
        "decisions": decisions,
        "rebuild": dict(rebuild),
    }
    return matches, audit


def parse_args():
    p = argparse.ArgumentParser()
    default_paths = [
        ("metrics_path", BASE / "rubric_context_consistency_metrics.json"),
        ("match_path", BASE / "rubric_matrix_list_match.json"),
        ("rubric_path", BASE / "rubric_with_dedup_oriented_prompt.json"),
        ("result_path", BASE / "generated_rubric_judge_result.json"),
        ("response_path", ROOT / "model_res.json"),
        ("output_path", BASE / "rubric_matrix_list_match_context_postprocessed.json"),
        ("audit_path", BASE / "rubric_matrix_list_match_context_postprocess_audit.json"),
    ]
    for name, default in default_paths:
        p.add_argument("--" + name.replace("_", "-"), type=Path, default=default)

    p.add_argument("--t-obs-threshold", type=float, default=float(os.environ.get("RUBRIC_POSTPROCESS_T_OBS_THRESHOLD", 0.1)))
    p.add_argument("--threshold-inclusive", action="store_true")
    p.add_argument("--score-gen-models", choices=["all", "flagged"], default="all")
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-response-chars", type=int, default=2500)
    p.add_argument("--max-evidence-chars", type=int, default=500)
    p.add_argument("--on-llm-error", choices=["keep_split", "fail"], default="keep_split")
    p.add_argument("--workers", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output, audit = postprocess(args)
    save(output, args.output_path)
    save(audit, args.audit_path)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
