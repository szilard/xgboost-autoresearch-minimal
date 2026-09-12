import urllib.request
from pathlib import Path

base_url = "https://xgboost-autoresearch--airline-dataset.s3.us-west-2.amazonaws.com"

files = [
    "2005-slice1-100k.csv",
    "2005-slice2-1m.csv",
    "2006-slice2-1m.csv",
]

data_dir = Path(__file__).parent / "data-cache"
data_dir.mkdir(exist_ok=True)

for name in files:
    dest = data_dir / name
    if dest.exists():
        print(f"{name}: already downloaded ({dest.stat().st_size:,} bytes)")
        continue
    print(f"Downloading {name}...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(f"{base_url}/{name}", tmp)
    tmp.rename(dest)
    print(f"  done ({dest.stat().st_size:,} bytes)")
