"""Tests for the command line interface."""

from odin_parser.cli import main


def test_cli_info(tmp_path, make_origin, capsys):
    origin = tmp_path / "origin"
    make_origin(origin)
    assert main(["info", str(origin)]) == 0
    out = capsys.readouterr().out
    assert "cloud frames: 5" in out
    assert "image frames: 14" in out
    assert "leads cloud" in out


def test_cli_process_and_preview(tmp_path, make_origin):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    assert (
        main(["process", str(origin), "--out", str(out), "--overwrite", "--no-masks"])
        == 0
    )
    assert (out / "dataset_info.json").exists()

    overlay = tmp_path / "overlay.png"
    assert main(["preview", str(out), "--frame", "0", "--out", str(overlay)]) == 0
    assert overlay.exists()


def test_cli_preview_frame_out_of_range(tmp_path, make_origin):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    main(["process", str(origin), "--out", str(out), "--overwrite", "--no-masks"])
    rc = main(["preview", str(out), "--frame", "999", "--out", str(tmp_path / "x.png")])
    assert rc == 1


def test_cli_info_handles_empty_streams(tmp_path, capsys):
    origin = tmp_path / "origin"
    origin.mkdir()
    for name in ("OdinImage.bin", "OdinPose.bin", "MT20260424_231347.olx"):
        (origin / name).write_bytes(b"")
    assert main(["info", str(origin)]) == 0
    out = capsys.readouterr().out
    assert "cloud frames: 0" in out
    assert "image frames: 0" in out


def test_cli_info_missing_dir(tmp_path, capsys):
    assert main(["info", str(tmp_path / "nope")]) == 1
    assert "no Odin streams found" in capsys.readouterr().out
