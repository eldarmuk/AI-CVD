"""Offline candidate checks; technical checks never grant disclosure permission."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

MANIFEST = "public-release-manifest.json"
ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".yml", ".cff", ".txt", ".json"}
FORBIDDEN_DIRS = {"data", "reports", "notebooks", "outputs", "runs", "logs", "checkpoints"}


def path_problem(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        return "unsafe path"
    if path.parts[0] in FORBIDDEN_DIRS:
        return "generated/private directory"
    if name == ".gitignore":
        return None
    if path.suffix not in ALLOWED_SUFFIXES:
        return "unreviewed file type"
    return None


def link_problems(root, name, text):
    errors = []
    # Inline Markdown links/images, with optional title. Also validate fragments.
    for match in re.finditer(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", text):
        target = unquote(match.group(1).strip("<>"))
        parts = urlsplit(target)
        if parts.scheme or parts.netloc:
            continue
        destination = (root / name).parent / parts.path if parts.path else root / name
        destination = destination.resolve()
        if not destination.is_relative_to(root.resolve()) or not destination.exists():
            errors.append(f"{name}: broken or external local link")
            continue
        if parts.fragment and destination.suffix == ".md":
            headings = re.findall(r"^#{1,6}\s+(.+)$", destination.read_text(encoding="utf-8"), re.M)
            anchors = [re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings]
            if parts.fragment not in anchors:
                errors.append(f"{name}: missing heading anchor")
    return errors


def inspect(root, tracked):
    errors = []
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    entries = manifest["files"]
    expected = {entry["path"] for entry in entries} | {MANIFEST}
    if len(entries) != len(expected) - 1:
        errors.append("duplicate manifest entry")
    for name in sorted(expected ^ set(tracked)):
        errors.append(f"{name}: tracked/manifest mismatch")
    digests = {entry["path"]: entry["sha256"] for entry in entries}
    for name in sorted(set(tracked)):
        problem = path_problem(name)
        if problem:
            errors.append(f"{name}: {problem}")
            continue
        path = root / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"{name}: missing or symlink file")
            continue
        raw = path.read_bytes()
        # Git may convert LF/CRLF on Windows; manifest hashes normalized UTF-8 text.
        try:
            text = raw.decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            errors.append(f"{name}: binary content")
            continue
        if "\x00" in text:
            errors.append(f"{name}: binary content")
        if name != MANIFEST and hashlib.sha256(text.encode()).hexdigest() != digests.get(name):
            errors.append(f"{name}: unreviewed content digest")
        patterns = [
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"\b(?:ghp|gho|ghs)_[A-Za-z0-9]{30,}\b",
            r"\bAKIA[A-Z0-9]{16}\b",
            r"[A-Z]:\\Users\\",
            "/" + r"home/[^/\s]+/",
        ]
        if any(re.search(pattern, text) for pattern in patterns):
            errors.append(f"{name}: possible secret or personal path (value withheld)")
        if name.endswith(".md"):
            errors.extend(link_problems(root, name, text))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    tracked = (
        subprocess.check_output(["git", "-C", str(args.root), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    tracked = [name for name in tracked if name]
    errors = inspect(args.root, tracked)
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)
    print(
        f"PASS: {len(tracked)} tracked text files; manifest, artifact policy, links and targeted patterns."
    )
    print("Custodian/rights approval and independent secret scanning are separate requirements.")


if __name__ == "__main__":
    main()
