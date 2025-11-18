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


from contextlib import ExitStack
from pathlib import Path

import pytest
import typer

from flwr.common.constant import FLWR_DIR
from flwr.supercore.constant import (
    MAX_DIR_DEPTH,
    MAX_FILE_BYTES,
    MAX_FILE_COUNT,
    MAX_TOTAL_BYTES,
)

from .publish import (
    _build_multipart_files_param,
    _collect_files,
    _compile_gitignore,
    _depth_of,
    _detect_mime,
    _load_gitignore,
    _validate_files,
)

TEXT_EXT = ".py"
ALT_TEXT_EXT = ".md"


def write(tmp: Path, file_name: str, data: bytes) -> Path:
    """Write data to given path."""
    path = tmp / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_load_gitignore(tmp_path: Path) -> None:
    """Test '.gitignore' loading."""
    (tmp_path / ".gitignore").write_text("*.log\nsecret/\n", encoding="utf-8")
    raw = _load_gitignore(tmp_path)

    assert raw is not None
    assert isinstance(raw, bytes)

    lines = raw.decode("utf-8").splitlines()
    assert lines == ["*.log", "secret/"]


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
def test_depth_of(rel: Path, expected: int) -> None:
    """Test the directory depth detection."""
    assert _depth_of(rel) == expected


def test_detect_mime_has_string(tmp_path: Path) -> None:
    """Test the MIME detection."""
    f = write(tmp_path, f"a{TEXT_EXT}", b"print('x')")
    mime = _detect_mime(f)
    assert isinstance(mime, str) and len(mime) > 0


def test_collect_files_depth_limit(tmp_path: Path) -> None:
    """Test collect files depth limit."""
    # Create a path deeper than allowed
    parts = [f"d{i}" for i in range(MAX_DIR_DEPTH + 1)]
    deep = Path(*parts) / f"too_deep{TEXT_EXT}"
    write(tmp_path, deep.as_posix(), b"x")

    with pytest.raises(typer.Exit) as exc:
        _collect_files(tmp_path)
    assert exc.value.exit_code == 2


def test_collect_files_count_limit(tmp_path: Path) -> None:
    """Test collect files count limit."""
    # Create (max_count + 1) tiny files
    for i in range(MAX_FILE_COUNT + 1):
        write(tmp_path, f"f{i}{TEXT_EXT}", b"x")

    with pytest.raises(typer.Exit) as exc:
        _validate_files(_collect_files(tmp_path))
    assert exc.value.exit_code == 2


def test_collect_files_total_bytes_limit(tmp_path: Path) -> None:
    """Test collect files total bytes limit."""
    # One file exceeding the total limit by 1 byte
    data = b"x" * (MAX_TOTAL_BYTES + 1)
    write(tmp_path, f"big{TEXT_EXT}", data)

    with pytest.raises(typer.Exit) as exc:
        _validate_files(_collect_files(tmp_path))
    assert exc.value.exit_code == 2


def test_collect_files_per_file_size_limit(tmp_path: Path) -> None:
    """Test collect files per file size limit."""
    data = b"x" * (MAX_FILE_BYTES + 1)
    write(tmp_path, f"too_big{ALT_TEXT_EXT}", data)

    with pytest.raises(typer.Exit) as exc:
        _validate_files(_collect_files(tmp_path))
    assert exc.value.exit_code == 2


def test_collect_files_non_utf8_raises_for_text(tmp_path: Path) -> None:
    """Test collect files UTF8 format."""
    # Invalid UTF-8 payload in a text extension
    write(tmp_path, f"bad{TEXT_EXT}", b"\xff\xfe\xfa")

    with pytest.raises(typer.Exit) as exc:
        _validate_files(_collect_files(tmp_path))
    assert exc.value.exit_code == 2


def test_build_multipart_files_param(tmp_path: Path) -> None:
    """Test multipart files building."""
    f1 = write(tmp_path, f"a{TEXT_EXT}", b"hello")
    files = [(f1, Path(f"a{TEXT_EXT}"))]

    with ExitStack() as stack:
        parts = _build_multipart_files_param(files, stack)
        assert len(parts) == 1
        key, (fname, fobj, mime) = parts[0]
        assert key == "files"
        assert fname == f"a{TEXT_EXT}"
        assert hasattr(fobj, "read")
        assert isinstance(mime, str) and mime

    # ExitStack closes the opened file object
    with pytest.raises(ValueError):
        fobj.read(1)  # closed file
