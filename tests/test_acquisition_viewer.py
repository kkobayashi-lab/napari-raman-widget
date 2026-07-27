import json

from napari_raman_widget.acquisition_viewer import vandermonde_objectives


def test_vandermonde_objectives_are_read_without_hardware(tmp_path):
    model = tmp_path / "vandermonde.json"
    model.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "objectives": {
                    "1": {"degree": 1, "C": []},
                    "3": {"degree": 1, "C": []},
                },
            }
        ),
        encoding="utf-8",
    )

    assert vandermonde_objectives(model) == ["1", "3"]
