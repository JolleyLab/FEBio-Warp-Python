"""Create a root-level, browser-uploadable source ZIP using an explicit allowlist."""

from pathlib import Path
import argparse
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path, default=root / "dist/warp_febio_upload.zip"
)
args = parser.parse_args()
files = [
    root / name
    for name in ("README.md", "MERGE_NOTES.md", "pyproject.toml", ".gitignore")
]
for folder in ("docs", "src", "examples", "tests", "tools", "validation"):
    for path in (root / folder).rglob("*"):
        if not path.is_file() or path.is_symlink() or path.name == "UPLOAD_MANIFEST.json":
            continue
        if any(
            part == "__pycache__" or part.endswith(".egg-info") for part in path.parts
        ):
            continue
        if path.suffix not in (".py", ".md", ".json", ".csv", ".txt", ".feb", ".toml"):
            continue
        files.append(path)
files = sorted(set(files))
if len(files) + 1 > 100:
    print("Upload the extracted files in batches of at most 100.")
assert all(
    p.stat().st_size < 25 * 1024 * 1024 for p in files
), "Browser file size limit exceeded"
manifest = [
    dict(
        path=str(p.relative_to(root)),
        bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    for p in files
]
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(
    args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9
) as archive:
    for path in files:
        archive.write(path, str(path.relative_to(root)))
    archive.writestr(
        "validation/UPLOAD_MANIFEST.json", json.dumps(manifest, indent=2) + "\n"
    )
with zipfile.ZipFile(args.output) as archive:
    assert archive.testzip() is None
    for entry in manifest:
        assert (
            hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
        )
print(
    json.dumps(
        dict(
            zip=str(args.output),
            files=len(files) + 1,
            bytes=args.output.stat().st_size,
            largest_file_bytes=max(p.stat().st_size for p in files),
            integrity="PASS",
        ),
        indent=2,
    )
)
