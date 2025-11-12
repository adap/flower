# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for Flower command line interface `app publish` command."""


import json
from contextlib import ExitStack
from pathlib import Path

import pytest

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    FLWR_DIR,
    REFRESH_TOKEN_KEY,
)
from flwr.supercore.constant import ALLOWED_EXTS

from .publish import (
    _build_multipart_files_param,
    _compile_gitignore,
    _depth_of,
    _detect_mime,
    _load_gitignore,
    _validate_credentials_content,
)


def write(tmp: Path, rel: str, data: bytes) -> Path:
    """Write data to given path."""
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def pick_text_ext():
    """Pick a texty extension from the REAL allowed set, or skip if none."""
    preferred = [
        ".txt",
        ".py",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
    ]
    for ext in preferred:
        if ext in ALLOWED_EXTS:
            return ext
    # Fall back to any allowed extension; if none exist, skip the test
    allowed = list(ALLOWED_EXTS)
    if not allowed:
        pytest.skip("No allowed extensions configured")
    return allowed[0]


def test_load_gitignore(tmp_path: Path) -> None:
    """Test '.gitignore' loading."""
    (tmp_path / ".gitignore").write_text("*.log\nsecret/\n", encoding="utf-8")
    lines = _load_gitignore(tmp_path)
    assert list(lines) == ["*.log", "secret/"]


def test_compile_gitignore_ignores_flwr_dir(tmp_path: Path) -> None:
    """Test if the specified files are ignored."""
    flwr_dir = FLWR_DIR
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")

    ignored = _compile_gitignore(tmp_path)

    file_tmp = write(tmp_path, "a.tmp", b"ok")
    assert ignored(file_tmp) is True

    write(tmp_path, f"{flwr_dir}/creds.json", b"{}")
    assert ignored(tmp_path / flwr_dir / "creds.json") is True

    normal = write(tmp_path, "good.py", b"print('x')")
    assert ignored(normal) is False


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        (Path("a.py"), 0),
        (Path("d1/file.txt"), 1),
        (Path("d1/d2/d3/f.txt"), 3),
        (Path("d1/d2/d3/d4/d5/x"), 5),
    ],
)
def test_depth_of(rel, expected):
    """Test the directory depth detection."""
    assert _depth_of(rel) == expected


def test_detect_mime_has_string(tmp_path: Path):
    """Test the MIME detection."""
    ext = pick_text_ext()
    f = write(tmp_path, f"a{ext}", b"content")
    # We don't assert exact mapping—only that a non-empty MIME string is returned.
    mime = _detect_mime(f)
    assert isinstance(mime, str) and len(mime) > 0


def test_build_multipart_files_param(tmp_path: Path):
    """Test multipart files building."""
    ext = pick_text_ext()
    f1 = write(tmp_path, f"a{ext}", b"hello")
    files = [(f1, Path(f"a{ext}"))]

    with ExitStack() as stack:
        parts = _build_multipart_files_param(files, stack)
        assert len(parts) == 1
        key, (fname, fobj, mime) = parts[0]
        assert key == "files"
        assert fname == f"a{ext}"
        assert hasattr(fobj, "read")
        assert isinstance(mime, str) and mime

    # ExitStack closes the opened file object
    with pytest.raises(ValueError):
        fobj.read(1)  # closed file


def test_validate_credentials_content_success(tmp_path: Path):
    """Test the credentials content loading."""
    creds = {
        AUTHN_TYPE_JSON_KEY: "userpass",
        ACCESS_TOKEN_KEY: "abc",
        REFRESH_TOKEN_KEY: "def",
    }
    p = tmp_path / "creds.json"
    p.write_text(json.dumps(creds), encoding="utf-8")
    token = _validate_credentials_content(p)
    assert token == "abc"
