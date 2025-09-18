from pipeline.pipeline import _api_settings


def test_api_settings_includes_custom_base_url():
    config = {
        "data": {
            "api": {
                "base_url": " https://alt.example/v2 ",
                "params": {"query.term": "glioblastoma"},
            }
        }
    }

    _, _, _, client_settings = _api_settings(config)

    assert client_settings["base_url"] == "https://alt.example/v2"
