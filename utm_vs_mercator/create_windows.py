"""Outputs commands that can be used to create the windows."""

with open("coords.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        lat, lon = line.split(", ")
        print(
            f"python -m rslearn.main dataset add_windows --root /data/favyenb/rslearn_utm_vs_webmercator/ --group webmercator --name='{lon}_{lat}_webmercator' --box='{lon},{lat},{lon},{lat}' --crs EPSG:3857 --resolution 9.555 --src_crs EPSG:4326 --window_size 512 --start '2020-05-01T00:00:00' --end '2020-10-01T00:00:00'"
        )
        print(
            f"python -m rslearn.main dataset add_windows --root /data/favyenb/rslearn_utm_vs_webmercator/ --group utm --name='{lon}_{lat}_utm' --box='{lon},{lat},{lon},{lat}' --utm --resolution 10 --src_crs EPSG:4326 --window_size 512 --start '2020-05-01T00:00:00' --end '2020-10-01T00:00:00'"
        )
