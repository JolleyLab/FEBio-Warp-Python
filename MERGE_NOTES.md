# Upload to a separate GitHub repository

Extract the generated ZIP and upload its CONTENTS through Add file / Upload files.
It opens directly to docs/, src/, examples/, tests/, tools/, validation/ and packaging
files. Do not upload the ZIP as source. .gitignore may need a manual browser edit.

If files already exist, manually merge README.md, pyproject.toml, .gitignore and
any matching paths. Preserve the destination's package metadata/build backend,
license and CI configuration. No .git directory or remote is created here.
No vendored Warp, anatomical mesh, environment or historical trajectory is included.

If the extracted ZIP contains more than 100 files, use two uploads and preserve
the relative folder paths in each batch.
