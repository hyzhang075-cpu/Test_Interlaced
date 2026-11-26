"""Result writes tolerate brief Windows locks without hiding persistent failures."""
import pytest
from shape_sensing.artifacts import file_io


def test_atomic_replace_retries_transient_windows_lock(tmp_path, monkeypatch):
    source = tmp_path / 'new.json'
    target = tmp_path / 'result.json'
    source.write_text('new', encoding='utf-8')
    target.write_text('old', encoding='utf-8')
    replace = file_io.os.replace
    calls = []
    def locked_once(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            error = PermissionError('sharing violation')
            error.winerror = 32
            raise error
        replace(src, dst)
    monkeypatch.setattr(file_io.os, 'replace', locked_once)
    monkeypatch.setattr(file_io.time, 'sleep', lambda seconds: None)
    file_io.atomic_replace(source, target)
    assert len(calls) == 2
    assert target.read_text(encoding='utf-8') == 'new'
    assert not source.exists()


def test_atomic_replace_preserves_persistent_error(monkeypatch):
    calls = []
    def locked(src, dst):
        calls.append(1)
        error = PermissionError('access denied')
        error.winerror = 5
        raise error
    monkeypatch.setattr(file_io.os, 'replace', locked)
    monkeypatch.setattr(file_io.time, 'sleep', lambda seconds: None)
    with pytest.raises(PermissionError, match='access denied'):
        file_io.atomic_replace('new', 'old')
    assert len(calls) == 6
