"""Regenerate the reference sections of docs/tutorial.md from the code.

Each section sits between <!-- AUTO-GENERATED:<name> --> and <!-- /AUTO-GENERATED:<name> -->
and is rebuilt from its source of truth, so the tutorial can't drift from it:

    cli-commands   the sdgf command line's subcommands (sdgf.cli.build_parser)
    mcp-options    python -m sdgf.mcp's options (sdgf.mcp.server.build_parser)
    mcp-tools      the MCP server's tools, with authoring on (sdgf.mcp.server.build_server)
    extras         pyproject.toml's optional dependencies
    environment    API key variables the shipped task specs name, and those sdgf reads

    python docs/tutorial_reference.py            rewrite the sections
    python docs/tutorial_reference.py --check    exit 1 if any section is stale

Needs the mcp extra (pip install -e '.[mcp]'). Text outside the markers is never touched.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import tomllib
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
TUTORIAL = DOCS / "tutorial.md"
MARKER = re.compile(
    r"(<!-- AUTO-GENERATED:(?P<name>[\w-]+) -->\n).*?(<!-- /AUTO-GENERATED:(?P=name) -->)",
    re.DOTALL,
)


def _cell(text: str | None) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def _subcommands(parser: argparse.ArgumentParser) -> list[tuple[str, str, argparse.ArgumentParser]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {a.dest: a.help for a in action._choices_actions}
            return [(name, helps.get(name) or "", sub) for name, sub in action.choices.items()]
    return []


def cli_commands() -> str:
    from sdgf.cli import build_parser

    rows = ["| Command | What it does |", "|---|---|"]
    for name, help_, sub in _subcommands(build_parser()):
        rows.append(f"| `sdgf {name}` | {_cell(help_)} |")
        for child, child_help, _ in _subcommands(sub):
            rows.append(f"| `sdgf {name} … {child}` | {_cell(child_help)} |")
    return "\n".join(rows)


def mcp_options() -> str:
    from sdgf.mcp.server import build_parser

    rows = ["| Option | What it sets |", "|---|---|"]
    for action in build_parser()._actions:
        if not action.option_strings or action.dest == "help":
            continue
        flag = max(action.option_strings, key=len)
        if action.nargs == 0:
            value = ""
        elif action.choices:
            value = " {" + ",".join(map(str, action.choices)) + "}"
        else:
            value = f" {action.metavar or action.dest.upper()}"
        default = action.default
        shown = "" if default in (None, [], False) else f" (default {default})"
        rows.append(f"| `{flag}{value}` | {_cell(action.help)}{shown} |")
    return "\n".join(rows)


def mcp_tools() -> str:
    import anyio
    from mcp import Client

    from sdgf.mcp.server import ServerConfig, build_server

    with tempfile.TemporaryDirectory() as tmp:
        server = build_server(
            ServerConfig(store=Path(tmp) / "store", task_roots=[], drafts=Path(tmp) / "drafts")
        )

        async def listed():
            async with Client(server) as client:
                return (await client.list_tools()).tools

        tools = anyio.run(listed)
    rows = ["| Tool | What it does |", "|---|---|"]
    for tool in tools:
        params = ", ".join(tool.input_schema.get("properties", {}))
        rows.append(f"| `{tool.name}({params})` | {_cell(tool.description)} |")
    return "\n".join(rows)


def extras() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    rows = ["| Extra | Installs |", "|---|---|"]
    for name, packages in project["optional-dependencies"].items():
        rows.append(f"| `{name}` | {', '.join(f'`{p}`' for p in packages)} |")
    return "\n".join(rows)


def environment() -> str:
    keys: dict[str, set[str]] = {}
    for spec in sorted((ROOT / "tasks").glob("*/task*.yaml")):
        for name in re.findall(r"api_key_env:\s*([A-Z_][A-Z0-9_]*)", spec.read_text()):
            keys.setdefault(name, set()).add(str(spec.relative_to(ROOT)))
    read: dict[str, set[str]] = {}
    for source in sorted((ROOT / "src" / "sdgf").rglob("*.py")):
        pattern = r'environ(?:\.get)?[\[(]\s*"([A-Z_][A-Z0-9_]*)"'
        for name in re.findall(pattern, source.read_text()):
            read.setdefault(name, set()).add(str(source.relative_to(ROOT)))
    rows = ["| Variable | Needed for |", "|---|---|"]
    for name in sorted(keys):
        rows.append(
            f"| `{name}` | the API key of models in {', '.join(f'`{s}`' for s in sorted(keys[name]))} |"
        )
    for name in sorted(set(read) - set(keys)):
        files = ", ".join(f"`{s}`" for s in sorted(read[name]))
        rows.append(f"| `{name}` | read by {files} |")
    return "\n".join(rows)


SECTIONS = {
    "cli-commands": cli_commands,
    "mcp-options": mcp_options,
    "mcp-tools": mcp_tools,
    "extras": extras,
    "environment": environment,
}


def render(text: str) -> str:
    found = {m.group("name") for m in MARKER.finditer(text)}
    unknown = sorted(found - set(SECTIONS))
    missing = sorted(set(SECTIONS) - found)
    if unknown or missing:
        raise SystemExit(f"tutorial.md markers: unknown {unknown}, missing {missing}")
    return MARKER.sub(lambda m: f"{m.group(1)}{SECTIONS[m.group('name')]()}\n{m.group(3)}", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a section is stale")
    args = parser.parse_args(argv)
    current = TUTORIAL.read_text(encoding="utf-8")
    fresh = render(current)
    if args.check:
        if fresh != current:
            print(
                "docs/tutorial.md is stale: run python docs/tutorial_reference.py", file=sys.stderr
            )
            return 1
        return 0
    TUTORIAL.write_text(fresh, encoding="utf-8")
    print(f"wrote {TUTORIAL.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
