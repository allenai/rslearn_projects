"""Convert a fine-tune's last.ckpt into a checkpoint directory a probe can load.

Why this is needed: the fine-tune writes ONE file --
  {epoch, model_state, optimizer_state, scheduler_state, best_state, ...}
with the encoder under a `backbone.` prefix and the task head under `_head.`.
A probe's --trainer.load_path instead wants a directory of
  config.json + model_and_optim/   (olmo-core distributed-checkpoint format)
and loads weights through olmo-core, which cannot read the single file. No PASTIS
fine-tune has ever produced the directory form, so the transfer probe could never
have loaded one.

Design: start from the FULL pretrained checkpoint and overwrite only the encoder.
The fine-tune checkpoint holds just backbone + head, so a model rebuilt from it
alone would be missing decoder / target_encoder / supervision_head. Patching keeps
a complete, loadable model whose encoder is the fine-tuned one; the discarded
`_head.` is the 19-class French head, which the PLANTEUR probe replaces anyway.

Runs inside the container: olmo-core is not installed on the host.

  python convert_ft_ckpt.py --ft <last.ckpt> --base <pretrained step dir> --out <dir>
"""
import argparse, json, os, shutil, sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True, help="fine-tune last.ckpt")
    ap.add_argument("--base", required=True, help="pretrained step<N> dir")
    ap.add_argument("--out", required=True, help="checkpoint dir to write")
    ap.add_argument("--state", default="best_state", choices=["best_state", "model_state"],
                    help="best_state is the epoch selected on PASTIS val")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the encoder is unchanged. ONLY for smoke-"
                         "testing the write path -- the output is not a fine-tuned "
                         "encoder and must not be used for a reported result.")
    a = ap.parse_args()

    # The repo only ever IMPORTS load_model_and_optim_state, never a save helper, so
    # the save API name is not something to assume. Discover it and report what the
    # installed olmo-core actually offers.
    import olmo_core.distributed.checkpoint as occ
    saver = next((getattr(occ, n) for n in
                  ("save_model_and_optim_state", "save_state_dict", "save_model_state")
                  if hasattr(occ, n)), None)
    print(f"[0/5] olmo_core.distributed.checkpoint exports: "
          f"{[n for n in dir(occ) if 'save' in n.lower()]}", flush=True)
    print(f"      using saver: {getattr(saver, '__name__', None)}", flush=True)
    from olmoearth_pretrain.model_loader import load_pretrain_checkpoint

    print(f"[1/5] building the pretrained model from {a.base}", flush=True)
    model = load_pretrain_checkpoint(a.base)
    tgt = model.state_dict()
    n_enc = sum(1 for k in tgt if k.startswith("encoder."))
    print(f"      model has {len(tgt)} tensors, {n_enc} under encoder.", flush=True)

    print(f"[2/5] reading {a.ft}", flush=True)
    ck = torch.load(a.ft, map_location="cpu", weights_only=True)
    sd = ck[a.state]
    print(f"      epoch={ck.get('epoch')} backbone_unfrozen={ck.get('backbone_unfrozen')} "
          f"best_val={ck.get('best_val_metric')}", flush=True)
    if ck.get("backbone_unfrozen") is False:
        print("      WARNING: backbone still frozen -- encoder equals pretrained.",
              flush=True)
        if not a.force:
            print("      REFUSING: this conversion would be a no-op dressed up as a "
                  "fine-tune. Re-run once the backbone has unfrozen.", flush=True)
            return 2
        print("      --force given: continuing as a WRITE-PATH TEST ONLY.", flush=True)

    print("[3/5] mapping backbone.* -> encoder.*", flush=True)
    mapped, skipped = {}, []
    for k, v in sd.items():
        if k.startswith("backbone."):
            mapped["encoder." + k[len("backbone."):]] = v
        else:
            skipped.append(k)
    print(f"      {len(mapped)} encoder tensors, dropped {len(skipped)} "
          f"(the task head: {skipped[:3]})", flush=True)

    # every mapped key must exist in the target with the same shape, and the
    # fine-tune must cover the whole encoder -- a partial overwrite would silently
    # leave some layers pretrained
    missing = [k for k in mapped if k not in tgt]
    badshape = [k for k in mapped if k in tgt and tuple(tgt[k].shape) != tuple(mapped[k].shape)]
    uncovered = [k for k in tgt if k.startswith("encoder.") and k not in mapped]
    print(f"      unmatched={len(missing)} shape-mismatched={len(badshape)} "
          f"encoder-keys-not-overwritten={len(uncovered)}", flush=True)
    if missing or badshape:
        print(f"      ABORT missing={missing[:4]} badshape={badshape[:4]}", flush=True)
        return 1
    if uncovered:
        print(f"      ABORT: fine-tune did not cover {uncovered[:4]}", flush=True)
        return 1

    changed = sum(1 for k in mapped if not torch.equal(tgt[k], mapped[k]))
    print(f"      {changed}/{len(mapped)} encoder tensors differ from pretrained",
          flush=True)
    if changed == 0:
        print("      encoder identical to pretrained; nothing was learned.", flush=True)
        if not a.force:
            print("      REFUSING.", flush=True)
            return 2

    print("[4/5] loading the fine-tuned encoder into the model", flush=True)
    incompat = model.load_state_dict(mapped, strict=False)
    assert not incompat.unexpected_keys, incompat.unexpected_keys[:4]

    print(f"[5/5] writing {a.out}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(os.path.join(a.base, "config.json"),
                    os.path.join(a.out, "config.json"))
    # weights.pth always: load_pretrain_checkpoint reads it, and it costs nothing.
    # model_and_optim/ additionally, since the TRAINER's load_path wants the
    # distributed layout -- written only if this olmo-core exposes a saver.
    torch.save(model.state_dict(), os.path.join(a.out, "weights.pth"))
    print("      wrote weights.pth", flush=True)
    if saver is not None:
        saver(os.path.join(a.out, "model_and_optim"), model)
        print("      wrote model_and_optim/", flush=True)
    else:
        print("      NO SAVER FOUND -- model_and_optim/ not written; a probe's "
              "--trainer.load_path will not be able to load this yet.", flush=True)

    # prove it: reload from the directory we just wrote and compare the encoder
    print("      verifying by reloading the written directory", flush=True)
    rt = load_pretrain_checkpoint(a.out).state_dict()
    bad = [k for k in mapped if not torch.equal(rt[k].cpu(), mapped[k].cpu())]
    if bad:
        print(f"      VERIFY FAILED on {len(bad)} tensors, e.g. {bad[:3]}", flush=True)
        return 1
    print(f"      OK: all {len(mapped)} encoder tensors round-tripped", flush=True)
    print("CONVERT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
