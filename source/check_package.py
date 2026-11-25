"""Verify all distributed files against the delivered SHA-256 manifest."""
import csv, hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
with (root/"package_sha256.csv").open(encoding="utf-8",newline="") as stream:
    rows=list(csv.DictReader(stream))
for row in rows:
    path=root/row["file"]
    with path.open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==row["sha256"], row["file"]
    assert path.stat().st_size==int(row["bytes"]),row["file"]
print("Verified",len(rows),"files.")

