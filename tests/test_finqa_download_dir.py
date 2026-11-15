"""load_finqa_datasets(data_dir=...) puts the downloaded files in data_dir, not in ./data."""

import tarfile

from agenttune.agentic.tools.builtin import finqa_tool

CSV = "user_query,answer,company,id,question_type\nWhat?,1,ACME,1,basic\n"


def fake_archive(tmp_path):
    src = tmp_path / "src" / "data"
    (src / "company_tables").mkdir(parents=True)
    for name in ("train_finqa.csv", "val_finqa.csv"):
        (src / name).write_text(CSV)
    (src / "company_tables" / "acme.json").write_text("{}")
    tar_path = tmp_path / "data.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src, arcname="data")
    return str(tar_path)


def test_download_lands_in_data_dir(tmp_path, monkeypatch):
    archive = fake_archive(tmp_path)
    monkeypatch.setattr(finqa_tool, "hf_hub_download", lambda **kw: archive)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    data_dir = tmp_path / "elsewhere" / "finqa"

    train, val = finqa_tool.load_finqa_datasets(data_dir=data_dir)

    assert len(train) == 1 and len(val) == 1
    assert (data_dir / "company_tables" / "acme.json").exists()
    assert list(cwd.iterdir()) == []
    # A second call finds the files and does not download again.
    monkeypatch.setattr(finqa_tool, "hf_hub_download", None)
    assert len(finqa_tool.load_finqa_datasets(data_dir=data_dir)[0]) == 1
