"""Relative links in the Markdown files point at files and headings that exist.

The README links to diagrams and decision tables in the design notes by
heading, so a renamed heading would break a link without this check. Links to
other sites are not followed. Pilot material is kept as contributed.
"""

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", re.S)


def markdown_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / line for line in out.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in ROOT.rglob("*.md") if ".venv" not in p.parts]
    return [p for p in files if p.exists() and "pilots" not in p.relative_to(ROOT).parts]


def anchors(path):
    """Heading anchors as GitHub makes them: lower case, punctuation dropped, spaces to hyphens."""
    found, seen = set(), {}
    for line in FENCE.sub("", path.read_text(encoding="utf-8")).splitlines():
        m = re.match(r"^#+\s+(.*)", line)
        if not m:
            continue
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(1).strip())
        anchor = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        n = seen.get(anchor, 0)
        seen[anchor] = n + 1
        found.add(anchor if n == 0 else f"{anchor}-{n}")
    return found


class MarkdownLinks(unittest.TestCase):
    def test_relative_links_and_anchors_resolve(self):
        broken = []
        for path in markdown_files():
            text = FENCE.sub("", path.read_text(encoding="utf-8"))
            for target in LINK.findall(text):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                file_part, _, anchor = target.partition("#")
                dest = (path.parent / file_part).resolve() if file_part else path
                where = f"{path.relative_to(ROOT)}: {target}"
                if not dest.exists():
                    broken.append(f"{where} (no such file)")
                elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                    broken.append(f"{where} (no such heading)")
        self.assertEqual(broken, [], "broken links:\n" + "\n".join(broken))


if __name__ == "__main__":
    unittest.main()
