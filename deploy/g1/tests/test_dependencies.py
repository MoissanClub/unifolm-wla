import json
from pathlib import Path

import pytest

from deploy.g1 import dependencies as deps


def report(*names):
    return {"install": [{
        "metadata": {"name": name, "version": "1.0"},
        "download_info": {"url": f"https://example.invalid/{name}.whl",
                          "archive_info": {"hashes": {"sha256": "a"*64}}},
    } for name in names]}


def test_versions_reused_or_reported_without_replacement():
    requirements = [deps.Requirement(s) for s in ("numpy>=1.24,<2", "msgpack>=1,<2", "Pillow>=10", "diffusers==0.35.2")]
    missing, conflicts = deps.classify(requirements, {"numpy": "1.26.4", "msgpack": "1.1.1", "diffusers": "0.36.0"})
    assert missing == ["Pillow>=10"]
    assert len(conflicts) == 1 and "diffusers==0.36.0" in conflicts[0]


def test_export_reuses_deployed_onnx_but_rejects_numpy2():
    requirements = deps.read_requirements(Path(deps.__file__).with_name("requirements-export.txt"))
    _, conflicts = deps.classify(requirements, {"onnx": "1.23.0", "numpy": "2.2.6"})
    assert len(conflicts) == 1 and "numpy==2.2.6" in conflicts[0]


@pytest.mark.parametrize("name", ["torch", "torchvision", "numpy", "scipy", "nvidia-cudss-cu12",
                                 "cuda-toolkit", "tensorrt-cu12", "opencv-python", "pin", "pyrealsense2"])
def test_new_platform_packages_blocked(name):
    with pytest.raises(ValueError, match="platform"):
        deps.validate_plan(report(name), {})


def test_even_same_version_reinstall_is_blocked():
    with pytest.raises(ValueError, match="replace installed"):
        deps.validate_plan(report("msgpack"), {"msgpack": "1.0"})


def test_unhashed_archive_is_blocked():
    planned = report("msgpack")
    planned["install"][0]["download_info"]["archive_info"] = {}
    with pytest.raises(ValueError, match="hash"):
        deps.validate_plan(planned, {})


def mock_main(monkeypatch, versions, args=()):
    monkeypatch.setattr(deps.sys, "argv", ["dependencies", "client", *args])
    monkeypatch.setattr(deps, "verify_conda", lambda: None)
    monkeypatch.setattr(deps, "verify_runtime", lambda kind: None)
    monkeypatch.setattr(deps, "verify_application", lambda kind: None)
    monkeypatch.setattr(deps, "installed_versions", lambda: versions)
    monkeypatch.setattr(deps, "read_requirements", lambda path: [deps.Requirement("msgpack>=1,<2")])
    def unexpected_install(*args, **kwargs):
        pytest.fail("pip must not run")
    monkeypatch.setattr(deps, "install_missing", unexpected_install)


def test_satisfied_environment_never_runs_pip(monkeypatch):
    mock_main(monkeypatch, {"msgpack": "1.1.1"})
    deps.main()


@pytest.mark.parametrize("versions,args", [({"msgpack": "2.0"}, ()), ({}, ("--check",))])
def test_conflict_and_offline_check_never_install(monkeypatch, versions, args):
    mock_main(monkeypatch, versions, args)
    with pytest.raises(SystemExit) as error:
        deps.main()
    assert error.value.code == 1


@pytest.mark.parametrize("dry_run", [False, True])
def test_transaction_freezes_all_packages_and_applies_only_plan(monkeypatch, dry_run):
    installed = {"numpy": "1.26.4", "torch": "2.11.0", "urllib3": "2.6.0"}
    calls = []
    current = installed.copy()
    def run(command, check):
        calls.append(command)
        assert check
        if "--dry-run" in command:
            constraints = Path(command[command.index("-c")+1]).read_text()
            assert all(f"{name}=={version}\n" in constraints for name, version in installed.items())
            assert Path(command[command.index("-r")+1]).read_text().strip() == "msgpack>=1,<2"
            Path(command[command.index("--report")+1]).write_text(json.dumps(report("msgpack")))
        else:
            assert "--no-deps" in command and "--no-index" in command
            assert command[-1].endswith("#sha256="+"a"*64)
            current["msgpack"] = "1.0"
    monkeypatch.setattr(deps.subprocess, "run", run)
    monkeypatch.setattr(deps, "installed_versions", lambda: current.copy())
    deps.install_missing(["msgpack>=1,<2"], installed, dry_run)
    assert len(calls) == (1 if dry_run else 2)
    assert all(current[name] == version for name, version in installed.items())


def test_environment_change_aborts_before_install(monkeypatch):
    def run(command, check):
        assert "--dry-run" in command
        Path(command[command.index("--report")+1]).write_text(json.dumps(report("msgpack")))
    monkeypatch.setattr(deps.subprocess, "run", run)
    monkeypatch.setattr(deps, "installed_versions", lambda: {"numpy": "2.2.6"})
    with pytest.raises(RuntimeError, match="changed during planning"):
        deps.install_missing(["msgpack>=1,<2"], {"numpy": "1.26.4"})
