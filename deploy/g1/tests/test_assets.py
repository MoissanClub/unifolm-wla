import json
import struct

import pytest

from deploy.g1.assets import inspect_checkpoint


def test_checkpoint_header_is_bounded_by_file_not_deployment_ceiling(tmp_path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header = header.ljust(11 * 1024 * 1024, b" ")
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1))
    assert inspect_checkpoint(path) == {"weight": {"parameters": 1, "bytes": 4, "dtypes": ["F32"]}}


@pytest.mark.parametrize("content", [b"", struct.pack("<Q", 100) + b"{}"])
def test_checkpoint_truncation_is_still_rejected(tmp_path, content):
    path = tmp_path / "model.safetensors"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="header"):
        inspect_checkpoint(path)
