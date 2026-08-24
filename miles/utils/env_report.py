import base64
import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

logger = logging.getLogger(__name__)


@dataclass
class EditablePackageInfo:
    name: str
    version: str
    location: str


@dataclass
class GitRepoInfo:
    package_name: str
    location: str
    commit: str
    dirty: bool
    diff_stat: str
    remote_url: str | None = None


@dataclass(frozen=True)
class PackageProvenance:
    version: str
    source_type: str
    location: str | None
    source_url: str | None
    commit: str | None
    dirty: bool | None
    diff_stat: str | None


@dataclass
class NodeEnvReport:
    role: str
    rank: int
    launcher_env_report: dict[str, Any] | None
    editable_packages: list[EditablePackageInfo]
    git_repos: list[GitRepoInfo]
    full_pip_list: list[dict[str, str]]


_CORE_PACKAGE_NAMES = frozenset({"megatron-bridge", "megatron-core", "miles", "sglang"})
_LOCAL_SOURCE_TYPES = frozenset({"editable", "editable_git", "local", "local_git", "source_path_git", "vcs"})
_REPOSITORY_PACKAGE_NAMES = {"megatron-lm": "megatron-core"}


def decode_env_report(raw: str) -> dict[str, Any] | None:
    """Decode an env report string (base64-encoded JSON or raw JSON)."""
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw).decode()
        return json.loads(decoded)
    except Exception:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse env report", exc_info=True)
            return None


def collect_and_print_node_env_report(
    *,
    role: str,
    rank: int,
    partial_env_report: str,
) -> NodeEnvReport:
    """Collect environment info for this node, print to stdout, return structured report.

    Called during actor init. Only performs collection when partial_env_report is non-empty.

    Args:
        role: Actor role, e.g. "training" or "rollout"
        rank: Actor rank
        partial_env_report: JSON string from launcher (may contain launch config info)
    """
    launcher_report = decode_env_report(partial_env_report)

    editable_packages, full_pip_list = _collect_pip_info()

    git_repos = [
        info for pkg in editable_packages if (info := _collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    report = NodeEnvReport(
        role=role,
        rank=rank,
        launcher_env_report=launcher_report,
        editable_packages=editable_packages,
        git_repos=git_repos,
        full_pip_list=full_pip_list,
    )

    _print_report(report)
    return report


def _collect_pip_info() -> tuple[list[EditablePackageInfo], list[dict[str, str]]]:
    """Collect all pip info in a single `pip inspect` call.

    Returns (editable_packages, full_pip_list).
    """
    installed = _inspect_installed_packages()
    full_pip_list = [_parse_pip_entry(pkg) for pkg in installed]
    editable_packages = [
        EditablePackageInfo(
            name=entry["name"],
            version=entry["version"],
            location=_file_url_to_path(pkg["direct_url"]["url"]),
        )
        for pkg, entry in zip(installed, full_pip_list, strict=True)
        if _is_editable(pkg)
    ]
    return editable_packages, full_pip_list


def _inspect_installed_packages() -> list[dict[str, Any]]:
    try:
        # TODO: remove this workaround and still make Megatron detected
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(
            ["pip", "inspect"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if result.returncode != 0:
            logger.warning("pip inspect failed: %s", result.stderr)
            return []
        data = json.loads(result.stdout)
        return data.get("installed", [])
    except Exception:
        logger.warning("Failed to collect pip info", exc_info=True)
        return []


def _parse_pip_entry(pkg: dict[str, Any]) -> dict[str, str]:
    metadata = pkg.get("metadata", {})
    return {"name": metadata.get("name", ""), "version": metadata.get("version", "")}


def _is_editable(pkg: dict[str, Any]) -> bool:
    direct_url = pkg.get("direct_url")
    return bool(direct_url and direct_url.get("dir_info", {}).get("editable"))


def collect_code_provenance() -> dict[str, dict[str, Any]]:
    """Collect reproducibility metadata for source-backed and core runtime packages."""
    provenance = {}
    for package in _inspect_installed_packages():
        package_name = _normalize_package_name(package.get("metadata", {}).get("name", ""))
        if not package_name:
            continue
        try:
            package_provenance = _collect_package_provenance(package_name=package_name, package=package)
        except Exception:
            logger.warning("Failed to collect code provenance for %s", package_name, exc_info=True)
            continue
        if package_name not in _CORE_PACKAGE_NAMES and package_provenance.source_type not in _LOCAL_SOURCE_TYPES:
            continue
        provenance[package_name] = asdict(package_provenance)
    return dict(sorted(_add_source_path_repositories(provenance).items()))


def _collect_package_provenance(*, package_name: str, package: dict[str, Any]) -> PackageProvenance:
    version = package.get("metadata", {}).get("version", "")
    direct_url = package.get("direct_url") or {}
    source_url = direct_url.get("url")
    vcs_info = direct_url.get("vcs_info")

    if vcs_info:
        return PackageProvenance(
            version=version,
            source_type="vcs",
            location=None,
            source_url=_sanitize_source_url(source_url),
            commit=vcs_info.get("commit_id"),
            dirty=None,
            diff_stat=None,
        )

    if source_url and "dir_info" in direct_url and urlsplit(source_url).scheme == "file":
        location = _file_url_to_path(source_url)
        git_info = _collect_git_info(package_name=package_name, location=location)
        editable = _is_editable(package)
        if git_info:
            source_type = "editable_git" if editable else "local_git"
        else:
            source_type = "editable" if editable else "local"
        return PackageProvenance(
            version=version,
            source_type=source_type,
            location=location,
            source_url=git_info.remote_url if git_info else None,
            commit=git_info.commit if git_info else None,
            dirty=git_info.dirty if git_info else None,
            diff_stat=git_info.diff_stat if git_info else None,
        )

    return PackageProvenance(
        version=version,
        source_type="archive" if direct_url else "installed",
        location=None,
        source_url=_sanitize_source_url(source_url),
        commit=None,
        dirty=None,
        diff_stat=None,
    )


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _add_source_path_repositories(provenance: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = dict(provenance)
    seen_locations = {
        os.path.realpath(package["location"])
        for package in output.values()
        if package.get("location")
    }
    source_package_names = set()
    for location in dict.fromkeys(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        git_info = _collect_git_info(package_name="", location=location)
        if not git_info or git_info.location in seen_locations:
            continue
        package_name = _repository_package_name(git_info)
        if not package_name or package_name in source_package_names:
            continue
        existing = output.get(package_name, {})
        output[package_name] = asdict(
            PackageProvenance(
                version=existing.get("version", ""),
                source_type="source_path_git",
                location=git_info.location,
                source_url=git_info.remote_url,
                commit=git_info.commit,
                dirty=git_info.dirty,
                diff_stat=git_info.diff_stat,
            )
        )
        seen_locations.add(git_info.location)
        source_package_names.add(package_name)
    return output


def _repository_package_name(git_info: GitRepoInfo) -> str:
    source = git_info.remote_url or git_info.location
    repository_name = source.rstrip("/").rsplit("/", maxsplit=1)[-1].removesuffix(".git")
    normalized_name = _normalize_package_name(repository_name)
    return _REPOSITORY_PACKAGE_NAMES.get(normalized_name, normalized_name)


def _file_url_to_path(url: str) -> str:
    return unquote(urlsplit(url).path)


def _sanitize_source_url(url: str | None) -> str | None:
    if not url:
        return None
    if "://" not in url:
        return url.split("@", maxsplit=1)[-1]
    try:
        parsed = urlsplit(url)
        if not parsed.hostname:
            return None
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return None


def _collect_git_info(*, package_name: str, location: str) -> GitRepoInfo | None:
    if not location or not os.path.isdir(location):
        return None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=location,
        )
        if commit_result.returncode != 0:
            return None
        commit = commit_result.stdout.strip()

        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=location,
        )
        if root_result.returncode != 0:
            return None
        repo_location = os.path.realpath(root_result.stdout.strip())

        diff_result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=repo_location,
        )
        diff_stat = diff_result.stdout.strip()

        dirty = bool(diff_stat)

        remote_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=repo_location,
        )
        remote_url = _sanitize_source_url(remote_result.stdout.strip()) if remote_result.returncode == 0 else None

        return GitRepoInfo(
            package_name=package_name,
            location=repo_location,
            commit=commit,
            dirty=dirty,
            diff_stat=diff_stat,
            remote_url=remote_url,
        )
    except Exception:
        logger.warning("Failed to collect git info for %s at %s", package_name, location, exc_info=True)
        return None


ENV_REPORT_PREFIX = "ENV_REPORT_JSON="


def _print_report(report: NodeEnvReport) -> None:
    print(f"{ENV_REPORT_PREFIX}{json.dumps(asdict(report), separators=(',', ':'), sort_keys=True, default=str)}")
