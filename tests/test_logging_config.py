from napari_raman_widget.logging_config import configure_pymmcore_process_log


def test_default_pymmcore_log_is_unique_to_process():
    environ = {"LOCALAPPDATA": r"C:\Users\test\AppData\Local"}

    result = configure_pymmcore_process_log(environ, pid=12345)

    assert result is not None
    assert result.name == "cns-raman-12345.log"
    assert environ["PYMM_LOG_FILE"] == str(result)


def test_explicit_pymmcore_log_is_preserved():
    environ = {"PYMM_LOG_FILE": r"C:\logs\chosen.log"}

    result = configure_pymmcore_process_log(environ, pid=12345)

    assert result is not None
    assert str(result) == environ["PYMM_LOG_FILE"]


def test_explicit_disabled_file_logging_is_preserved():
    environ = {"PYMM_LOG_FILE": "none"}

    result = configure_pymmcore_process_log(environ, pid=12345)

    assert result is None
    assert environ["PYMM_LOG_FILE"] == "none"
