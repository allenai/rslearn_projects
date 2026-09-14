"""Re-run the few-shot arms on the d768_proj128lin checkpoint (option B).

The published few-shot / arms probe numbers came from spec_x*-probe-old.yml, which
loads the RETIRED d128_wideread checkpoint -- a different model from the one behind
the 443 and LOIO probe tables. They were also a single LR at a single projection
dim, while 443/LOIO report best-of-8-LRs at 64d and 128d. Both differences are
fixed here so Table 5 becomes comparable to Table 1.

Grid: 2 dims x 3 arms x 8 LRs = 48 jobs, 4 tasks each (X = 10/25/100/1000).
Arms map to task families -- unbalanced has NO arm token (pxi), balanced is bal,
planteur-only is plo (SUPARM in build_v3_data.py).
"""
import re, subprocess, sys, yaml

SC = "/tmp/claude-0/-/bc3913f2-90e2-4f94-86a8-5293802f1535/scratchpad"
P2 = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
TEMPLATE = f"{SC}/spec_p128-443-lr0.1-r5.yml"
CKPT = ("/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/"
        "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1/"
        "step667200")
XS = [10, 25, 100, 1000]
ARMS = {"unbal": "pxi", "bal": "bal", "plo": "plo"}
LRS = ["0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5"]
DIMS = [64, 128]
DRY = "--dry" in sys.argv

base = yaml.safe_load(open(TEMPLATE))
btask = base["tasks"][0]
keep = [a for a in (str(x) for x in btask["arguments"]) if "tasks." not in a]

made = 0
for dim in DIMS:
    for arm, tok in ARMS.items():
        for lr in LRS:
            spec = yaml.safe_load(open(TEMPLATE))
            t = spec["tasks"][0]
            tag = f"fsb-p{dim}-{arm}-lr{lr}"
            tasks = [f"pastis_planteur_{tok}{X}_ws16_ps1_sentinel2" for X in XS]

            args = list(keep)
            args[2] = (f"regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform"
                       f"_stunorm_mlpgram1_step667200_emb_{arm}_p{dim}_lr{lr}")
            args = [f"--trainer.load_path={CKPT}" if a.startswith("--trainer.load_path=")
                    else a for a in args]
            args = [f'--trainer.callbacks.downstream_evaluator.tasks_to_run='
                    f'[{", ".join(chr(34)+x+chr(34) for x in tasks)}]'
                    if a.startswith("--trainer.callbacks.downstream_evaluator.tasks_to_run=")
                    else a for a in args]
            for x in tasks:
                p = f"--trainer.callbacks.downstream_evaluator.tasks.{x}"
                args += [f"{p}.norm_stats_from_pretrained=True",
                         f"{p}.quantize_embeddings=True",
                         f"{p}.probe_lr={lr}",
                         # both flags are required: eval_projection_dim alone raises
                         f"{p}.eval_on_projected_registers=True",
                         f"{p}.eval_projection_dim={dim}"]
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
            assert CKPT in joined, "must load the NEW checkpoint"
            assert "d128_wideread" not in joined, "retired checkpoint leaked in"
            assert joined.count(f"eval_projection_dim={dim}") == len(XS)
            assert joined.count("eval_on_projected_registers=True") == len(XS)
            assert joined.count(f"probe_lr={lr}") == len(XS)
            assert "wandb.enabled=False" in joined
            assert ev["OE_PRED_DIR"].endswith(tag) and ev["OE_DUMP_DIR"].endswith(tag)
            for x in tasks:
                assert x in joined

            p = f"{SC}/spec_{tag}.yml"
            open(p, "w").write(yaml.safe_dump(spec))
            if DRY:
                made += 1
                continue
            r = subprocess.run(["beaker", "experiment", "create", p, "--name", tag,
                                "--workspace", "ai2/earth-systems"],
                               capture_output=True, text=True, timeout=300)
            if r.returncode:
                print(f"  FAIL {tag}: {(r.stderr or r.stdout).strip()[-110:]}", flush=True)
            else:
                made += 1
                print(f"  OK   {tag}", flush=True)
print(f"  {'validated' if DRY else 'launched'} {made} of "
      f"{len(DIMS)*len(ARMS)*len(LRS)} jobs ({len(XS)} tasks each)")
