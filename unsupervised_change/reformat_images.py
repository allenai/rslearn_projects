import glob
import os
import shutil
import tqdm

fnames = glob.glob('windows/default/*/layers/sentinel2/*/image.png')

needed = []
for fname in tqdm.tqdm(fnames):
    slash_parts = fname.split('/')
    date_parts = slash_parts[2].split('_')
    out_fname = f"{date_parts[0]}_{date_parts[1]}.png"
    label = f"{date_parts[2]}-{date_parts[3]}"
    band = slash_parts[5].lower()
    if band == 'r_g_b':
        band = 'tci'
    dst_dir = os.path.join('reformatted', label, band)
    #if os.path.exists(os.path.join(dst_dir, out_fname)):
    #    continue
    needed.append((fname, dst_dir, out_fname))

for fname, dst_dir, out_fname in tqdm.tqdm(needed):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(fname, os.path.join(dst_dir, out_fname))
