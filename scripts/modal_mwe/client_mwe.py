from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from author_name_disambiguation.public_cli import main


def _translate_args(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    command, *rest = argv
    if command == "run":
        translated = ["infer", "--backend", "modal"]
        index = 0
        while index < len(rest):
            token = rest[index]
            if token == "--publications":
                translated.extend(["--publications-path", rest[index + 1]])
                index += 2
                continue
            if token == "--references":
                translated.extend(["--references-path", rest[index + 1]])
                index += 2
                continue
            translated.append(token)
            index += 1
        return translated
    if command == "cost":
        return ["cost", *rest]
    return argv


if __name__ == "__main__":
    main(_translate_args(sys.argv[1:]))
