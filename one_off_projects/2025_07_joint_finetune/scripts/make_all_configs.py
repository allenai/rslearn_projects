import os
import sys

base_dir = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs"
d = sys.argv[1]

if not os.path.exists(os.path.join(base_dir, d)):
    d = [dd for dd in os.listdir(base_dir) if dd.endswith(d)]
    if len(d) == 1:
        d = d[0]
    else:
        i = int(input(f"No such directory. Choose from: {d} (enter index)"))
        d = d[i]

done = []
for f in os.listdir(os.path.join(base_dir, d)):
    if not f.startswith("OUT_") and not f.endswith("base.yaml") and f.endswith(".yaml"):
        print(os.path.join(base_dir, d, f))
        os.system(f"python3 make_multidataset_config.py --cfg {os.path.join(base_dir, d, f)}")
        done.append(os.path.join(base_dir, d, f))

print()
print("========== FINISHED ===========")
for f in done:
    print(f" - {f}")
