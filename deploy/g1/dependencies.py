"""Add missing dependencies to an existing conda env without replacing installed packages."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urldefrag

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version


# These are provisioned by the working robot environment, never by PyPI resolution here.
PLATFORM_PACKAGES = {
    "numpy", "scipy", "torch", "torchvision", "torchaudio", "triton", "pin", "pinocchio",
    "casadi", "cyclonedds", "unitree-sdk2py", "pyrealsense2", "cuda", "pytorch-triton",
}
PLATFORM_PREFIXES = ("nvidia-", "cuda-", "tensorrt", "opencv-", "cupy", "triton-", "pytorch-triton-")
LEARNINGS = "../g1-fridge-fetch/docs/06_dependency_learnings.md"
PIP = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check"]


def installed_versions():
    result = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            # Match importlib's first match when a system .pth exposes another distribution.
            result.setdefault(canonicalize_name(name), dist.version)
    return result


def read_requirements(path):
    requirements = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            requirements.extend(read_requirements(Path(path).parent / line[3:].strip()))
        else:
            req = Requirement(line)
            if req.marker is None or req.marker.evaluate():
                requirements.append(req)
    return requirements


def classify(requirements, installed):
    missing, conflicts = [], []
    for req in requirements:
        name = canonicalize_name(req.name)
        if name not in installed:
            missing.append(str(req))
        elif not req.specifier.contains(installed[name], prereleases=True):
            conflicts.append(f"{req.name}=={installed[name]} is installed; requires {req}")
    return missing, conflicts


def platform_package(name):
    name = canonicalize_name(name)
    return name in PLATFORM_PACKAGES or name.startswith(PLATFORM_PREFIXES)


def validate_plan(report, installed):
    """Reject replacements and any resolver attempt to provision the native/GPU stack."""
    urls = []
    for item in report.get("install", []):
        name = canonicalize_name(item["metadata"]["name"])
        if name in installed:
            raise ValueError(f"Refusing to replace installed {name}=={installed[name]}")
        if platform_package(name):
            raise ValueError(f"Refusing automatic platform package installation: {name}; see {LEARNINGS}")
        download = item["download_info"]
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        if not digest:
            raise ValueError(f"No immutable archive hash for planned package {name}")
        url, _ = urldefrag(download["url"])
        urls.append(f"{url}#sha256={digest}")
    return urls


def verify_conda():
    if not (Path(sys.prefix) / "conda-meta").is_dir():
        raise RuntimeError("Activate an existing conda environment; setup does not create virtualenvs")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >=3.10 is required")


def verify_runtime(kind):
    """Import libraries and run a GPU kernel; never open cameras or initialize DDS."""
    jetson = Path("/etc/nv_tegra_release").exists()
    if jetson and sys.version_info[:2] != (3, 10):
        raise RuntimeError("This JP6 setup reuses Python 3.10 system TensorRT bindings; keep conda on Python 3.10")
    # Load conda libstdc++ before torch when the robotics library is present.
    if kind == "client" or importlib.util.find_spec("pinocchio") is not None:
        importlib.import_module("pinocchio")
    if kind == "client":
        for module in ("cv2", "pyrealsense2", "unitree_sdk2py", "cyclonedds"):
            importlib.import_module(module)
        print("Runtime: Pinocchio, OpenCV, RealSense and DDS imports passed; no devices opened")
        return
    import torch
    import torchvision
    if Version(torch.__version__) < Version("2.8"):
        raise RuntimeError("PyTorch >=2.8 must already be provisioned; no generic torch wheel will be installed")
    print(f"Runtime: torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA {torch.version.cuda}")
    if kind == "model":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Model serving needs a working BF16 CUDA build, not CPU/SBSA torch")
        value = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        if (value @ value)[0, 0].item() != 32:
            raise RuntimeError("CUDA BF16 matrix multiplication failed")
        torch.cuda.synchronize()
        print("Runtime: CUDA BF16 matrix multiplication passed")
    if jetson:
        import tensorrt
        print(f"Runtime: TensorRT {tensorrt.__version__} from {tensorrt.__file__}")


def verify_application(kind):
    modules = ["deploy.g1.contract", "model_server.tools.msgpack_numpy"]
    if kind != "client":
        modules.append("unifolm_wla.model.framework.base_framework")
    if kind == "export":
        modules.extend(("onnx", "onnxscript", "ml_dtypes"))
    for module in modules:
        importlib.import_module(module)


def install_missing(requirements, installed, dry_run=False):
    with tempfile.TemporaryDirectory(prefix="unifolm-dependencies-") as directory:
        root = Path(directory)
        constraints = root / "installed.txt"
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(installed.items())))
        requested = root / "missing.txt"
        requested.write_text("\n".join(requirements)+"\n")
        report_path = root / "plan.json"
        # Freeze ALL visible installed distributions, including transitive dependencies.
        subprocess.run(PIP + ["install", "--dry-run", "--no-user", "--no-build-isolation", "--prefer-binary",
                              "--report", str(report_path), "-c", str(constraints), "-r", str(requested)], check=True)
        report = json.loads(report_path.read_text())
        urls = validate_plan(report, installed)
        for item in report.get("install", []):
            print(f"ADD {item['metadata']['name']}=={item['metadata']['version']}")
        if dry_run:
            print("Dry run complete; no packages installed")
            return
        if installed_versions() != installed:
            raise RuntimeError("Environment changed during planning; rerun setup")
        if urls:
            # Install exactly the audited archives. A second dependency resolution could pick new builds.
            subprocess.run(PIP + ["install", "--no-deps", "--no-index", "--no-user", "--no-build-isolation", *urls],
                           check=True)
        after = installed_versions()
        if any(after.get(name) != version for name, version in installed.items()):
            raise RuntimeError("Installed package versions changed unexpectedly; inspect this environment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("client", "model", "export"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline inventory/import checks; no installs or resolver")
    mode.add_argument("--dry-run", action="store_true", help="Resolve additions without installing; may access the index")
    args = parser.parse_args()
    try:
        verify_conda()
        print(f"Environment: {sys.prefix} ({sys.executable})", flush=True)
        requirements = read_requirements(Path(__file__).with_name(f"requirements-{args.kind}.txt"))
        installed = installed_versions()
        missing, conflicts = classify(requirements, installed)
        for req in requirements:
            name = canonicalize_name(req.name)
            status = "MISSING" if name not in installed else (
                "KEEP" if req.specifier.contains(installed[name], prereleases=True) else "CONFLICT")
            print(f"{status} {req.name}: {installed.get(name, str(req))}")
        if conflicts:
            raise RuntimeError("Version conflicts; no packages changed:\n  " + "\n  ".join(conflicts))
        protected_missing = [req for req in missing if platform_package(Requirement(req).name)]
        if protected_missing:
            raise RuntimeError("Provision these in the base conda environment first: " + ", ".join(protected_missing))
        verify_runtime(args.kind)
        if args.check and missing:
            raise RuntimeError("Missing dependencies listed above; --check made no changes")
        if missing:
            install_missing(missing, installed, args.dry_run)
            if args.dry_run:
                return
        else:
            print("All requested dependencies already satisfied; skipping pip")
        verify_application(args.kind)
        print("Dependency and import checks passed")
    except (RuntimeError, ValueError, ImportError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Setup stopped: {exc}\nNative/GPU stack troubleshooting: {LEARNINGS}\n")


if __name__ == "__main__":
    main()
