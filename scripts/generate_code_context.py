from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    REPO_ROOT
    / "oran_project_code_context.txt"
)


# Text/source file extensions that are useful for
# understanding the project.
INCLUDED_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".sh",
}


# Useful root files that may not have one of the
# suffixes above.
INCLUDED_NAMES = {
    "README",
    "README.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "pytest.ini",
}


# Directories whose contents should not be included,
# even if a tracked file happens to exist there.
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "experiments",
    "checkpoints",
    "wandb",
    "runs",
}


def run_git_command(
    *args: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.stdout.strip()


def get_tracked_files() -> list[Path]:
    output = run_git_command(
        "ls-files"
    )

    paths: list[Path] = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        path = Path(line)

        if any(
            part in EXCLUDED_PARTS
            for part in path.parts
        ):
            continue

        if (
            path.name not in INCLUDED_NAMES
            and path.suffix.lower()
            not in INCLUDED_SUFFIXES
        ):
            continue

        paths.append(
            path
        )

    return sorted(
        paths,
        key=lambda item: str(item),
    )


def get_git_metadata() -> dict[str, str]:
    metadata: dict[str, str] = {}

    try:
        metadata["branch"] = run_git_command(
            "branch",
            "--show-current",
        )
    except subprocess.CalledProcessError:
        metadata["branch"] = "UNKNOWN"

    try:
        metadata["commit"] = run_git_command(
            "rev-parse",
            "HEAD",
        )
    except subprocess.CalledProcessError:
        metadata["commit"] = "UNKNOWN"

    try:
        metadata["status"] = run_git_command(
            "status",
            "--short",
        )
    except subprocess.CalledProcessError:
        metadata["status"] = "UNKNOWN"

    return metadata


def read_text_file(
    path: Path,
) -> str:
    absolute_path = (
        REPO_ROOT
        / path
    )

    try:
        return absolute_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        return (
            "[FILE SKIPPED: "
            "not valid UTF-8 text]"
        )


def main() -> None:
    tracked_files = (
        get_tracked_files()
    )

    metadata = get_git_metadata()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:
        output.write(
            "=" * 80
        )
        output.write(
            "\nO-RAN DEEP SCHEDULER "
            "REPOSITORY CONTEXT\n"
        )
        output.write(
            "=" * 80
        )
        output.write(
            "\n\n"
        )

        output.write(
            f"Repository: {REPO_ROOT}\n"
        )

        output.write(
            "Git branch: "
            f"{metadata['branch']}\n"
        )

        output.write(
            "Git commit: "
            f"{metadata['commit']}\n"
        )

        output.write(
            "\nGit status --short:\n"
        )

        status = (
            metadata["status"]
            or "(clean)"
        )

        output.write(
            status
        )

        output.write(
            "\n\n"
        )

        output.write(
            "Tracked text/source files "
            "included:\n"
        )

        for path in tracked_files:
            output.write(
                f"  - {path}\n"
            )

        output.write(
            "\n"
        )

        for path in tracked_files:
            output.write(
                "\n"
            )

            output.write(
                "=" * 80
            )

            output.write(
                "\nFILE: "
                f"{path}\n"
            )

            output.write(
                "=" * 80
            )

            output.write(
                "\n\n"
            )

            content = read_text_file(
                path
            )

            output.write(
                content
            )

            if (
                content
                and not content.endswith(
                    "\n"
                )
            ):
                output.write(
                    "\n"
                )

    size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024 * 1024)
    )

    print(
        "Created:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print(
        "Included tracked files:"
    )

    print(
        f"  {len(tracked_files)}"
    )

    print(
        "Output size:"
    )

    print(
        f"  {size_mb:.2f} MiB"
    )


if __name__ == "__main__":
    main()


