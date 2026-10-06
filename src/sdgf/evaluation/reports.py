"""Stage 5 outputs (FRAMEWORK_DESIGN.md §6.6): a versioned release on pass, a shortfall
report on fail.

A release is one immutable directory:

    <release_root>/<task>/<version>/
        dataset.jsonl            the accepted records, without provenance
        provenance.jsonl         one line per record: index, sha256 of the record, provenance
        dataset_card.md          what the dataset is, how it was made, how it scored
        governance_report.json   governance profile, violations, and every model endpoint
                                 that received data, external ones listed apart (D12)
        metrics.json             the metrics report and the gate result
        manifest.json            version, spec_version, run id, counts, sha256 of each file

version defaults to <task.version>-<spec_version[:12]>-<run_id>. The directory is built
under a temporary name and renamed into place, so a release either exists complete or
not at all, and an existing release is never overwritten.

Endpoints come from the run's stage 0 record (spec.json "models": every stage the run
built, so every model that was sent data, dropped candidates included) plus any model
named in a record's provenance.

A failed gate writes shortfall.json and shortfall.md into the run directory: the failing
metrics by name, reason and cell, and the short cells with the records each still needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sdgf.evaluation.gate import GateResult, threshold_values
from sdgf.evaluation.metrics import MetricsReport
from sdgf.governance.profile import profile_for
from sdgf.store.provenance import PROVENANCE_KEY

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec
    from sdgf.store.artefacts import RunDir

DATASET = "dataset.jsonl"
PROVENANCE = "provenance.jsonl"
CARD = "dataset_card.md"
GOVERNANCE = "governance_report.json"
METRICS = "metrics.json"
MANIFEST = "manifest.json"
SHORTFALL = "shortfall"

RELEASE_FORMAT = 1


class ReportError(RuntimeError):
    """A release was asked for that must not be written."""


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _line(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n"


def record_sha256(record: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_line(record).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _split(record: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    bare = {k: v for k, v in record.items() if k != PROVENANCE_KEY}
    prov = record.get(PROVENANCE_KEY)
    return bare, prov if isinstance(prov, Mapping) else None


def default_version(compiled: CompiledSpec, run_id: str) -> str:
    return f"{compiled.spec.task.version}-{compiled.spec_version[:12]}-{run_id}"


# ── governance ───────────────────────────────────────────────────


def _endpoint_key(e: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (str(e["stage"]), str(e["backend"]), str(e["model"]), str(e["hosting"]))


def data_endpoints(
    run_models: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Every model endpoint that received data: the run's built stages plus any model a
    record's provenance names, deduplicated and sorted."""
    seen = {_endpoint_key(e) for e in run_models}
    for r in records:
        _, prov = _split(r)
        for m in (prov or {}).get("models", ()):
            seen.add(_endpoint_key(m))
    return [dict(zip(("stage", "backend", "model", "hosting"), k)) for k in sorted(seen)]


def governance_report(
    compiled: CompiledSpec,
    metrics: MetricsReport,
    gate: GateResult,
    endpoints: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    held_out_check: bool | None = None,
    trace_sinks: Sequence[Mapping[str, Any]] = (),
    seeds_sent_to: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    tools: dict[str, dict[str, Any]] = {}
    for r in records:
        _, prov = _split(r)
        for t in (prov or {}).get("tool_trace", ()):
            entry = tools.setdefault(t["tool"], {"calls": 0, "sensitivity": set()})
            entry["calls"] += 1
            if t.get("sensitivity"):
                entry["sensitivity"].add(t["sensitivity"])
    overall = metrics.overall
    return {
        "spec_version": compiled.spec_version,
        "profile": profile_for(compiled).to_dict(),
        "violations": overall.governance_violations,
        "violation_codes": dict(overall.governance_codes),
        "governance_gate": "fail" if gate.hard_fail else "pass",
        "overlap": dict(overall.overlap),
        "held_out_check": held_out_check,
        "endpoints": [dict(e) for e in endpoints],
        "external_endpoints": [dict(e) for e in endpoints if e["hosting"] == "provider_api"],
        # services that received run data for tracing (sdgf run --trace), not models
        "trace_sinks": [dict(t) for t in trace_sinks],
        # external models that were shown seed text as few-shot examples
        "seeds_sent_to": [dict(s) for s in seeds_sent_to],
        "tools": {
            name: {"calls": t["calls"], "sensitivity": sorted(t["sensitivity"])}
            for name, t in sorted(tools.items())
        },
    }


# ── dataset card ─────────────────────────────────────────────────


def _fmt(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def dataset_card(
    compiled: CompiledSpec,
    version: str,
    run_id: str,
    metrics: MetricsReport,
    gate: GateResult,
    governance: Mapping[str, Any],
    created_at: str,
) -> str:
    spec = compiled.spec
    task = spec.task
    overall = metrics.overall
    failed = {f.metric for f in gate.failures if f.cell is None}
    out = [
        f"# {task.name} {version}",
        "",
        "Synthetic dataset generated by sdgf. Every label and code-owned field comes from "
        "the coverage plan, not from the generator model.",
        "",
        "## Task",
        "",
        f"- Type: `{task.type}` ({task.generation_mode})",
        f"- Task version: {task.version}",
        f"- spec_version: `{compiled.spec_version}`",
        f"- Run: `{run_id}`",
        f"- Released: {created_at}",
        f"- Records: {overall.accepted}",
        "",
        task.description.strip(),
        "",
        "## Composition",
        "",
    ]
    for axis, bal in sorted(overall.balance.items()):
        out.append(f"**{axis}** (target vs released share)")
        out.append("")
        out.append("| value | target | released |")
        out.append("|---|---|---|")
        for value in sorted(set(bal.target) | set(bal.actual)):
            out.append(
                f"| {value} | {_fmt(bal.target.get(value))} | {_fmt(bal.actual.get(value))} |"
            )
        out.append("")
    out.append("| cell | quota | accepted |")
    out.append("|---|---|---|")
    for cell_id, m in sorted(metrics.per_cell.items()):
        out.append(f"| `{cell_id}` | {_fmt(m.quota)} | {m.accepted} |")
    out += ["", "## Models", "", "| stage | backend | model | hosting |", "|---|---|---|---|"]
    for e in governance["endpoints"]:
        out.append(f"| {e['stage']} | {e['backend']} | {e['model']} | {e['hosting']} |")
    external = governance["external_endpoints"]
    out += [
        "",
        f"External endpoints that received data: {len(external)}"
        + (" — " + ", ".join(f"{e['backend']}:{e['model']}" for e in external) if external else ""),
    ]
    for e in governance.get("seeds_sent_to", ()):
        out.append(
            f"Seed text sent to an external model: {e['seeds']} few-shot seed(s) to "
            f"{e['backend']}:{e['model']}."
        )
    for t in governance.get("trace_sinks", ()):
        out.append(
            f"Traced to {t['sink']} ({t['endpoint']}, project `{t['project']}`, "
            f"{t['content']} content)."
        )
    out += [
        "",
        "## Release gate",
        "",
        "| threshold | value | limit | result |",
        "|---|---|---|---|",
    ]
    values = threshold_values(overall)
    for name, limit in sorted(dict(spec.thresholds).items()):
        if limit is None:
            continue
        result = "waived" if name in gate.waived else ("fail" if name in failed else "pass")
        out.append(f"| {name} | {_fmt(values.get(name))} | {_fmt(limit)} | {result} |")
    if gate.waived:
        out += ["", "Waived (unmeasured, not failed): " + ", ".join(gate.waived)]
    out += [
        "",
        "## Cost",
        "",
        f"- Tokens per released record: {_fmt(overall.tokens_per_record)}",
        f"- Estimated cost per released record (USD): {_fmt(overall.cost_per_record)}",
        f"- Seconds per released record: {_fmt(overall.seconds_per_record)}",
    ]
    if overall.usage_by_stage:
        out += [
            "",
            "| stage | calls | tokens | estimated calls | cost (USD) | cost per record |",
            "|---|---|---|---|---|---|",
        ]
        for stage, u in sorted(overall.usage_by_stage.items()):
            out.append(
                f"| {stage} | {u.get('calls')} | {u.get('tokens')} | "
                f"{u.get('estimated_calls')} | {_fmt(u.get('cost_usd'))} | "
                f"{_fmt(u.get('cost_per_record'))} |"
            )
    out += [
        "",
        "## Governance",
        "",
        f"- Violations in the released set: {_fmt(governance['violations'])}",
        f"- Held-out overlap check: {_fmt(governance['held_out_check'])}",
    ]
    for exc in governance["profile"].get("exceptions", ()):
        out.append(f"- Documented exception `{exc['rule']}`: {exc['reason']}")
    out += [
        "",
        "## Files",
        "",
        f"- `{DATASET}`: records; `{PROVENANCE}`: per-record provenance, same order",
        f"- `{GOVERNANCE}`, `{METRICS}`, `{MANIFEST}`",
        "",
    ]
    return "\n".join(out)


# ── release ──────────────────────────────────────────────────────


def write_release(
    release_root: str | Path,
    compiled: CompiledSpec,
    run: RunDir,
    metrics: MetricsReport,
    gate: GateResult,
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    version: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Write the release directory for a passed gate and return its path.

    records defaults to the run's accepted.jsonl."""
    if not gate.passed or gate.hard_fail:
        raise ReportError(f"gate did not pass (failing: {gate.failing_metrics}); not releasing")
    if metrics.spec_version not in (None, compiled.spec_version) or (
        gate.spec_version not in (None, compiled.spec_version)
    ):
        raise ReportError("metrics or gate belong to a different spec_version")
    if run.spec_version != compiled.spec_version:
        raise ReportError("run belongs to a different spec_version")
    records = list(run.read_jsonl("accepted") if records is None else records)
    version = version or default_version(compiled, run.run_id)
    if "/" in version or version.startswith(".") or not version.strip():
        raise ReportError(f"invalid release version {version!r}")
    created_at = (now or datetime.now(timezone.utc)).isoformat()

    final = Path(release_root) / compiled.spec.task.name / version
    if final.exists():
        raise ReportError(f"release {final} already exists; releases are never overwritten")
    tmp = final.with_name(f".{version}.tmp-{secrets.token_hex(3)}")
    tmp.mkdir(parents=True)
    try:
        intake = run.read_stage("spec") if run.has_stage("spec") else {}
        endpoints = data_endpoints(intake.get("models", ()), records)
        governance = governance_report(
            compiled,
            metrics,
            gate,
            endpoints,
            records,
            held_out_check=intake.get("held_out_check"),
            trace_sinks=run.read_stage("trace_sinks") if run.has_stage("trace_sinks") else (),
            seeds_sent_to=intake.get("seeds_sent_to", ()),
        )
        with (
            (tmp / DATASET).open("w", encoding="utf-8") as data,
            (tmp / PROVENANCE).open("w", encoding="utf-8") as prov_out,
        ):
            for i, record in enumerate(records):
                bare, prov = _split(record)
                data.write(_line(bare))
                prov_out.write(
                    _line({"index": i, "record_sha256": record_sha256(bare), "provenance": prov})
                )
        (tmp / GOVERNANCE).write_text(_dumps(governance), encoding="utf-8")
        (tmp / METRICS).write_text(
            _dumps({"metrics": metrics.to_dict(), "gate": gate.to_dict()}), encoding="utf-8"
        )
        (tmp / CARD).write_text(
            dataset_card(compiled, version, run.run_id, metrics, gate, governance, created_at),
            encoding="utf-8",
        )
        files = {p.name: _file_sha256(p) for p in sorted(tmp.iterdir())}
        manifest = {
            "format": RELEASE_FORMAT,
            "task": compiled.spec.task.name,
            "version": version,
            "spec_version": compiled.spec_version,
            "run_id": run.run_id,
            "created_at": created_at,
            "records": len(records),
            "files": files,
        }
        (tmp / MANIFEST).write_text(_dumps(manifest), encoding="utf-8")
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def verify_release(path: str | Path) -> dict[str, Any]:
    """Re-hash a release's files against its manifest; raise ReportError on any mismatch."""
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if not (path / name).is_file() or _file_sha256(path / name) != digest:
            raise ReportError(f"{path / name}: missing or altered since release")
    return manifest


# ── shortfall ────────────────────────────────────────────────────


def shortfall_report(metrics: MetricsReport, gate: GateResult, run_id: str) -> dict[str, Any]:
    return {
        "spec_version": gate.spec_version or metrics.spec_version,
        "run_id": run_id,
        "passed": gate.passed,
        "hard_fail": gate.hard_fail,
        "failing_metrics": gate.failing_metrics,
        "failures": gate.to_dict()["failures"],
        "short_cells": dict(gate.short_cells),
        "records_missing": sum(gate.short_cells.values()),
        "waived": list(gate.waived),
        "accepted": metrics.overall.accepted,
        "quota": metrics.overall.quota,
    }


def _shortfall_markdown(report: Mapping[str, Any]) -> str:
    out = [
        f"# Shortfall — run {report['run_id']}",
        "",
        f"spec_version `{report['spec_version']}`: {report['accepted']} of "
        f"{_fmt(report['quota'])} records accepted.",
        "",
    ]
    if report["hard_fail"]:
        out += ["**Hard fail: governance violations in the accepted set.**", ""]
    out += ["| metric | reason | cell | value | limit |", "|---|---|---|---|---|"]
    for f in report["failures"]:
        cell = f"`{f['cell']}`" if f["cell"] else "overall"
        out.append(
            f"| {f['metric']} | {f['reason']} | {cell} | {_fmt(f['value'])} | "
            f"{_fmt(f['threshold'])} |"
        )
    out += ["", f"Short cells ({report['records_missing']} records missing):", ""]
    if report["short_cells"]:
        out += ["| cell | missing |", "|---|---|"]
        out += [f"| `{c}` | {n} |" for c, n in sorted(report["short_cells"].items())]
    else:
        out.append("none")
    out.append("")
    return "\n".join(out)


def write_shortfall(run: RunDir, metrics: MetricsReport, gate: GateResult) -> Path:
    """Write shortfall.json (a run stage artefact) and shortfall.md into the run directory
    for a failed gate; return the JSON path."""
    if gate.passed:
        raise ReportError("gate passed; there is no shortfall to report")
    report = shortfall_report(metrics, gate, run.run_id)
    path = run.write_stage(SHORTFALL, report)
    (run.path / f"{SHORTFALL}.md").write_text(_shortfall_markdown(report), encoding="utf-8")
    return path


def publish(
    release_root: str | Path,
    compiled: CompiledSpec,
    run: RunDir,
    metrics: MetricsReport,
    gate: GateResult,
    **kwargs: Any,
) -> Path:
    """The stage 5 outcome: a release directory on pass, a shortfall report on fail."""
    if gate.passed:
        return write_release(release_root, compiled, run, metrics, gate, **kwargs)
    return write_shortfall(run, metrics, gate)


__all__ = [
    "ReportError",
    "data_endpoints",
    "dataset_card",
    "default_version",
    "governance_report",
    "publish",
    "record_sha256",
    "shortfall_report",
    "verify_release",
    "write_release",
    "write_shortfall",
]
