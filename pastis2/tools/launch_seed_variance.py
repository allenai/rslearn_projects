"""Option A: 3 seeds on the headline probe configs, same 443/LOIO protocol.

Varies finetune_seed only -- the frozen encoder is deterministic and the test split
is fixed, so this isolates probe-training variance (init + data order) on exactly
the evaluation the published numbers use. That is what makes a gap like
"probe 128d 34.22 vs Tessera v2 34.24" falsifiable.

Deliberately NOT the balanced-trial protocol: that measures training-draw variance
on the remainder split, a different evaluation, and cannot produce error bars for
the published 443 number.

Grid: 2 dims x 2 splits x 8 LRs x 2 extra seeds = 64 jobs. Seed 42 is the existing
run, so only seeds 43 and 44 are new.
"""
import json, re, subprocess, sys, yaml

SC = "/tmp/claude-0/-/bc3913f2-90e2-4f94-86a8-5293802f1535/scratchpad"
P2 = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
LRS = ["0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5"]
SEEDS = [43, 44]
DRY = "--dry" in sys.argv

def existing():
    """Names already created -- a foreground timeout can truncate the launch, and
    beaker rejects duplicate names, so a rerun must skip rather than fail."""
    have = set()
    r = subprocess.run(["beaker", "workspace", "experiments", "ai2/earth-systems",
                        "--text", "sv-p", "--format", "json"],
                       capture_output=True, text=True, timeout=300)
    try:
        for x in json.loads(r.stdout or "[]"):
            n = x.get("name") or ""
            if n.startswith("sv-p"):
                have.add(n)
    except Exception:
        pass
    return have


HAVE = existing()
print(f"  {len(HAVE)} sv- jobs already exist; skipping those", flush=True)
made = 0
for dim in (64, 128):
    for split in ("443", "loio"):
        for lr in LRS:
            src = f"{SC}/spec_p{dim}-{split}-lr{lr}-r5.yml"
            try:
                base = yaml.safe_load(open(src))
            except FileNotFoundError:
                print(f"  SKIP missing template {src.split('/')[-1]}", flush=True)
                continue
            for seed in SEEDS:
                spec = yaml.safe_load(open(src))
                t = spec["tasks"][0]
                args = [str(a) for a in t["arguments"]]
                tasks = sorted(set(re.findall(
                    r"tasks\.([a-z0-9_]+)\.probe_lr", " ".join(args))))
                # add finetune_seed for every task this job runs
                for x in tasks:
                    args.append("--trainer.callbacks.downstream_evaluator.tasks."
                                f"{x}.finetune_seed={seed}")
                tag = f"sv-p{dim}-{split}-lr{lr}-s{seed}"
                args[2] = f"{args[2]}_seed{seed}"
                t["arguments"] = args
                for e in t.get("envVars", []):
                    if e["name"] == "OE_PRED_DIR":
                        e["value"] = f"{P2}/oe_preddump_{tag}"
                    elif e["name"] == "OE_DUMP_DIR":
                        e["value"] = f"{P2}/oe_embdump_{tag}"
                t["context"] = dict(t.get("context", {}), priority="urgent",
                                    minRuntime="4h", autoResume=True)
                t.setdefault("constraints", {})["cluster"] = ["ai2/jupiter"]

                joined = " ".join(args)
                ev = {e["name"]: str(e.get("value")) for e in t["envVars"]}
                assert joined.count(f"finetune_seed={seed}") == len(tasks), \
                    "seed must be set for every task"
                assert not re.search(r"finetune_seed=(?!%d\b)\d+" % seed, joined), \
                    "a different seed leaked in"
                assert joined.count(f"probe_lr={lr}") == len(tasks)
                assert ev["OE_PRED_DIR"].endswith(tag)
                assert f"eval_projection_dim={dim}" in joined

                if tag in HAVE:
                    continue
                p = f"{SC}/spec_{tag}.yml"
                open(p, "w").write(yaml.safe_dump(spec))
                if DRY:
                    made += 1
                    continue
                r = subprocess.run(["beaker", "experiment", "create", p, "--name", tag,
                                    "--workspace", "ai2/earth-systems"],
                                   capture_output=True, text=True, timeout=300)
                if r.returncode:
                    print(f"  FAIL {tag}: {(r.stderr or r.stdout).strip()[-100:]}", flush=True)
                else:
                    made += 1
print(f"  {'validated' if DRY else 'launched'} {made} jobs "
      f"(2 dims x 2 splits x {len(LRS)} LRs x {len(SEEDS)} seeds)")
