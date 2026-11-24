"""Check Sentinel-1 vessel labels with Gemini via Vertex AI batch prediction.

For each label, Gemini sees VV and VH chips (2.56 km, 256 x 256 px) of the target image and
of up to four same-track historical images, with a red ring at the label. It answers
wrong / low_confidence / correct. See README.md for how the answers were turned into
label edits (apply_labels.py).

Steps (each a subcommand; all state lives in --work_dir):

    items    list every label in windows that have a target image -> items.json
    render   draw the chips -> <work_dir>/chips, dates per item -> render.json
    upload   copy the chips to GCS (--gcs/chips)
    batch    write requests.jsonl, upload it and submit a Vertex batch job -> batches.json
    status   print the state of every submitted job
    collect  download finished outputs -> answers.jsonl and results.json
    retry    re-run items whose batch output was not valid JSON as live requests

Example:

    python gemini_label_check.py items --ds_path /weka/.../dataset_v1/20261103 --work_dir /tmp/check
    python gemini_label_check.py render --ds_path ... --work_dir /tmp/check
    python gemini_label_check.py upload --work_dir /tmp/check --gcs gs://bucket/prefix
    python gemini_label_check.py batch --work_dir /tmp/check --gcs gs://bucket/prefix
    python gemini_label_check.py status --work_dir /tmp/check
    python gemini_label_check.py collect --work_dir /tmp/check
"""

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
import rasterio
from PIL import Image, ImageDraw

MODEL = "gemini-4-argon"

PROMPT = """You are checking training labels for a ship detector on Sentinel-1 SAR imagery. Each label claims that there
is a vessel at one position in the "current" image. For each label you get Sentinel-1 IW GRD image chips centred on the
labelled position: 2.56 km x 2.56 km, 10 m pixels (256 x 256 px), north up, brightness in dB with the same stretch on
every date. You get VV and VH for the current image and for up to four earlier images of the same place from the same
satellite track (historical 1 is the most recent). A thin red ring marks the labelled position (radius about 70 m) on
every chip. Black areas are outside the satellite image (no data).

Decide whether the label is:
- wrong: there is no vessel at the ring. For example fixed infrastructure (an offshore oil/gas platform, a pier or
  jetty, a beacon), a wind turbine, an anchored platform, a single-point mooring buoy, an island or rock, a SAR glint,
  sidelobe or other artefact, or just nothing there (sea clutter, empty water).
- low_confidence: cannot be judged reliably, e.g. the position is on or right next to the border of the image (no-data
  edge), or the object is too faint to tell.
- correct: an actual vessel. Include vessels at sea. For moored vessels, only include if it is high-confidence and the
  vessel is not visible in most of the historical images. For anchored vessels, only include if it is high-confidence and
  most of the historical images do not contain what appears to be the same vessel.
Useful cues: fixed structures and islands are bright in exactly the same spot on every date; a vessel is usually absent or
at a different position on other dates, but a vessel can be moored or anchored near the same spot for a while.
Answer as JSON: {"answer": "wrong" | "low_confidence" | "correct", "category": "<short description, e.g. wind turbine,
vessel at sea, moored vessel, border>", "reason": "<one sentence>"}."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING", "enum": ["wrong", "low_confidence", "correct"]},
        "category": {"type": "STRING"},
        "reason": {"type": "STRING"},
    },
    "required": ["answer", "category", "reason"],
}

# Chip key -> dataset layer; h1 is the most recent historical.
LAYERS = {
    "t": "sentinel1",
    "h1": "sentinel1_historical",
    "h2": "sentinel1_historical.1",
    "h3": "sentinel1_historical.2",
    "h4": "sentinel1_historical.3",
}
NAMES = {
    "t": "current",
    "h1": "historical 1",
    "h2": "historical 2",
    "h3": "historical 3",
    "h4": "historical 4",
}
# The stretch is computed over the central 96 x 96 px; chips are 256 x 256 px.
R, RZ = 48, 128
RING_RADIUS = 7


def gsutil(*args: str) -> None:
    """Run gsutil, with a service account key if GCS_SERVICE_KEY_FILE is set."""
    cmd = ["gsutil"]
    if os.environ.get("GCS_SERVICE_KEY_FILE"):
        cmd += ["-o", f"Credentials:gs_service_key_file={os.environ['GCS_SERVICE_KEY_FILE']}"]
    subprocess.run(cmd + list(args), check=True)


def genai_client():  # type: ignore[no-untyped-def]
    """Vertex AI client; project from GOOGLE_CLOUD_PROJECT."""
    from google import genai

    return genai.Client(
        vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global"
    )


def load_json(path: str, default):  # type: ignore[no-untyped-def]
    """Load a JSON file, or return default if it doesn't exist."""
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: str) -> None:  # type: ignore[no-untyped-def]
    """Write JSON atomically."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=0)
    os.replace(path + ".tmp", path)


def labels_of(windows_dir: str, win: str) -> list[tuple[str, int, int]]:
    """(window, row, col) of every label in a window that has a target image."""
    if not os.path.exists(f"{windows_dir}/{win}/layers/sentinel1/vv_vh/geotiff.tif"):
        return []
    with open(f"{windows_dir}/{win}/layers/label/data.geojson") as f:
        fc = json.load(f)
    with open(f"{windows_dir}/{win}/metadata.json") as f:
        x0, y0 = json.load(f)["bounds"][:2]
    return [
        (
            win,
            int(round(feat["geometry"]["coordinates"][1] - y0)),
            int(round(feat["geometry"]["coordinates"][0] - x0)),
        )
        for feat in fc["features"]
    ]


def items(args: argparse.Namespace) -> None:
    """List every label as an item {id, win, row, col, split}."""
    windows_dir = f"{args.ds_path}/windows"
    wins = sorted(
        f"{g}/{w}" for g in os.listdir(windows_dir) for w in os.listdir(f"{windows_dir}/{g}")
    )
    with Pool(args.workers) as p:
        labs = sorted(
            {
                x
                for xs in p.map(partial(labels_of, windows_dir), wins, chunksize=64)
                for x in xs
            }
        )
    out = [
        {
            "id": f"label_{i:05d}",
            "win": win,
            "row": row,
            "col": col,
            "split": "val" if "-val_" in win else "train",
        }
        for i, (win, row, col) in enumerate(labs)
    ]
    save_json(out, f"{args.work_dir}/items.json")
    print(len(wins), "windows,", len(out), "labels")


def crop_db(a: np.ndarray, r0: int, c0: int, rad: int) -> np.ndarray:
    """Crop (2 rad)^2 px around (r0, c0), zero-padded, in dB."""
    pad = np.pad(a, ((0, 0), (rad, rad), (rad, rad)))
    return 20 * np.log10(np.maximum(pad[:, r0 : r0 + 2 * rad, c0 : c0 + 2 * rad], 1))


def render_one(windows_dir: str, chips_dir: str, it: dict) -> tuple[str, dict]:
    """Render the chips of one label.

    The stretch per band is shared across dates: 2nd percentile of valid pixels in the
    central 96 px to the max of the 5 x 5 px at the label.

    Returns:
        (item id, {chip key: acquisition date}) or (item id, {"error": ...}).
    """
    win, r0, c0 = it["win"], it["row"], it["col"]
    with open(f"{windows_dir}/{win}/items.json") as f:
        layer_items = {x["layer_name"]: x for x in json.load(f)}
    scenes = {"t": layer_items["sentinel1"]["serialized_item_groups"][0][0]}
    hist = layer_items.get("sentinel1_historical", {}).get("serialized_item_groups", [])
    for i, group in enumerate(hist):
        scenes[f"h{i + 1}"] = group[0]
    imgs = {}
    for k, layer in LAYERS.items():
        path = f"{windows_dir}/{win}/layers/{layer}/vv_vh/geotiff.tif"
        completed = os.path.exists(f"{windows_dir}/{win}/layers/{layer}/completed")
        if k in scenes and os.path.exists(path) and (k == "t" or completed):
            with rasterio.open(path) as r:
                imgs[k] = r.read().astype(np.float32)
    big = {k: crop_db(a, r0, c0, RZ) for k, a in imgs.items()}
    try:
        for b, band in enumerate(["vv", "vh"]):
            centres = [c[b][RZ - R : RZ + R, RZ - R : RZ + R] for c in big.values()]
            vals = np.concatenate([c[c > 0] for c in centres])
            lo = np.percentile(vals, 2) if vals.size else 0
            hi = max(c[R - 2 : R + 3, R - 2 : R + 3].max() for c in centres)
            for k, c in big.items():
                arr = np.clip((c[b] - lo) / max(hi - lo, 1e-3) * 255, 0, 255)
                im = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                ImageDraw.Draw(im).ellipse(
                    [RZ - RING_RADIUS, RZ - RING_RADIUS, RZ + RING_RADIUS, RZ + RING_RADIUS],
                    outline=(255, 40, 40),
                    width=1,
                )
                im.save(f"{chips_dir}/{it['id']}_{k}_{band}.png")
    except ValueError as e:
        # The label is outside its window (no pixels in the central 96 px).
        return it["id"], {"error": repr(e)[:120]}
    return it["id"], {k: scenes[k]["geometry"]["time_range"][0][:10] for k in big}


def render(args: argparse.Namespace) -> None:
    """Render chips for items not rendered yet (all, or the first --n)."""
    chips_dir = f"{args.work_dir}/chips"
    os.makedirs(chips_dir, exist_ok=True)
    its = load_json(f"{args.work_dir}/items.json", [])
    done = load_json(f"{args.work_dir}/render.json", {})
    errors = load_json(f"{args.work_dir}/render_errors.json", {})
    todo = [x for x in its if x["id"] not in done and x["id"] not in errors]
    if args.n:
        todo = todo[: args.n]
    fn = partial(render_one, f"{args.ds_path}/windows", chips_dir)
    with Pool(args.workers) as p:
        new = dict(p.map(fn, todo, chunksize=8))
    errors |= {k: v for k, v in new.items() if "error" in v}
    done |= {k: v for k, v in new.items() if "error" not in v}
    save_json(errors, f"{args.work_dir}/render_errors.json")
    save_json(done, f"{args.work_dir}/render.json")
    print(len(new), "rendered now,", len(done), "total,", len(errors), "errors")


def upload(args: argparse.Namespace) -> None:
    """Copy the chips to GCS."""
    gsutil("-m", "-q", "rsync", f"{args.work_dir}/chips", f"{args.gcs}/chips")


def request_for(item_id: str, dates: dict[str, str], gcs: str) -> dict:
    """The Vertex batch request for one label."""
    parts: list[dict] = [{"text": PROMPT}, {"text": "Now the label to check:"}]
    for k in ["t", "h1", "h2", "h3", "h4"]:
        if k not in dates:
            continue
        for band in ("vv", "vh"):
            parts.append({"text": f"{band.upper()}, {NAMES[k]} image, {dates[k]}:"})
            uri = f"{gcs}/chips/{item_id}_{k}_{band}.png"
            parts.append({"fileData": {"fileUri": uri, "mimeType": "image/png"}})
    return {
        "request": {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": SCHEMA,
                "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
            },
        }
    }


def batch(args: argparse.Namespace) -> None:
    """Submit a batch job for every rendered item without an answer."""
    from google.genai import types

    dates = load_json(f"{args.work_dir}/render.json", {})
    answered = {r["id"] for r in read_answers(args.work_dir)}
    todo = [i for i in sorted(dates) if i not in answered]
    if args.n:
        todo = todo[: args.n]
    tag = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    local = f"{args.work_dir}/requests_{tag}.jsonl"
    with open(local, "w") as f:
        for i in todo:
            f.write(json.dumps(request_for(i, dates[i], args.gcs)) + "\n")
    gsutil("-q", "cp", local, f"{args.gcs}/requests/requests_{tag}.jsonl")
    job = genai_client().batches.create(
        model=MODEL,
        src=f"{args.gcs}/requests/requests_{tag}.jsonl",
        config=types.CreateBatchJobConfig(
            dest=f"{args.gcs}/output/{tag}", display_name=f"s1-label-check-{tag}"
        ),
    )
    batches = load_json(f"{args.work_dir}/batches.json", [])
    batches.append(
        {"tag": tag, "job": job.name, "n": len(todo), "output": f"{args.gcs}/output/{tag}"}
    )
    save_json(batches, f"{args.work_dir}/batches.json")
    print("submitted", job.name, job.state, len(todo), "requests")


def status(args: argparse.Namespace) -> None:
    """Print the state of every submitted job."""
    client = genai_client()
    for b in load_json(f"{args.work_dir}/batches.json", []):
        job = client.batches.get(name=b["job"])
        print(b["tag"], b["n"], job.state, getattr(job, "error", None) or "")


def read_answers(work_dir: str) -> list[dict]:
    """Answers collected so far."""
    path = f"{work_dir}/answers.jsonl"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def item_id_from_uri(uri: str) -> str:
    """'.../chips/label_00012_t_vv.png' -> 'label_00012'."""
    return "_".join(os.path.basename(uri).split("_")[:2])


def collect(args: argparse.Namespace) -> None:
    """Download finished batch outputs and merge the answers into results.json."""
    client = genai_client()
    have = {r["id"]: r for r in read_answers(args.work_dir)}
    usage: Counter = Counter()
    bad = []
    for b in load_json(f"{args.work_dir}/batches.json", []):
        job = client.batches.get(name=b["job"])
        if "SUCCEEDED" not in str(job.state):
            print("skip", b["tag"], job.state)
            continue
        local = f"{args.work_dir}/out_{b['tag']}"
        os.makedirs(local, exist_ok=True)
        gsutil("-m", "-q", "rsync", "-r", b["output"], local)
        for root, _, files in os.walk(local):
            for fn in files:
                if not fn.endswith(".jsonl"):
                    continue
                with open(os.path.join(root, fn)) as f:
                    for line in f:
                        r = json.loads(line)
                        parts = r["request"]["contents"][0]["parts"]
                        uri = next(p["fileData"]["fileUri"] for p in parts if p.get("fileData"))
                        iid = item_id_from_uri(uri)
                        resp = r.get("response") or {}
                        try:
                            text = resp["candidates"][0]["content"]["parts"][0]["text"]
                            answer = json.loads(text)
                        except (KeyError, IndexError, ValueError):
                            if iid not in have:
                                bad.append(iid)
                            continue
                        u = resp.get("usageMetadata", {})
                        usage["in"] += u.get("promptTokenCount", 0)
                        usage["out"] += u.get("candidatesTokenCount", 0)
                        usage["out"] += u.get("thoughtsTokenCount", 0)
                        have[iid] = {"id": iid, "answer": answer, "batch": b["tag"]}
    write_answers(args.work_dir, have)
    print("batch tokens:", dict(usage))
    if bad:
        print(len(bad), "outputs were not valid JSON; rerun with: retry --ids", ",".join(bad))


def write_answers(work_dir: str, have: dict[str, dict]) -> None:
    """Write answers.jsonl and results.json (items + answers)."""
    with open(f"{work_dir}/answers.jsonl.tmp", "w") as f:
        for v in have.values():
            f.write(json.dumps(v) + "\n")
    os.replace(f"{work_dir}/answers.jsonl.tmp", f"{work_dir}/answers.jsonl")
    results = {}
    for it in load_json(f"{work_dir}/items.json", []):
        r = dict(it)
        if it["id"] in have:
            r["answer"] = have[it["id"]]["answer"]
        results[it["id"]] = r
    save_json(results, f"{work_dir}/results.json")
    counts = Counter(r["answer"]["answer"] if "answer" in r else "none" for r in results.values())
    print(len(results), "items:", dict(counts))


def retry(args: argparse.Namespace) -> None:
    """Re-run the given items as live (non-batch) requests."""
    client = genai_client()
    dates = load_json(f"{args.work_dir}/render.json", {})
    have = {r["id"]: r for r in read_answers(args.work_dir)}
    for iid in args.ids.split(","):
        req = request_for(iid, dates[iid], args.gcs)["request"]
        for _ in range(3):
            r = client.models.generate_content(
                model=MODEL, contents=req["contents"], config=req["generationConfig"]
            )
            try:
                have[iid] = {"id": iid, "answer": json.loads(r.text), "batch": "live_retry"}
                break
            except (TypeError, ValueError):
                continue
        print(iid, have.get(iid, {}).get("answer"))
    write_answers(args.work_dir, have)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command", choices=["items", "render", "upload", "batch", "status", "collect", "retry"]
    )
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--ds_path", help="rslearn dataset root (items, render)")
    parser.add_argument("--gcs", help="GCS prefix for chips, requests and outputs")
    parser.add_argument("--n", type=int, help="only the first N items (render, batch)")
    parser.add_argument("--ids", help="comma-separated item ids (retry)")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    globals()[args.command](args)
