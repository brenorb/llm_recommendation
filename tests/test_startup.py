import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class StartupTest(unittest.TestCase):
    def test_data_routes_work_without_an_openai_key(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            (data / "rating.csv").write_text(
                "user_id,anime_id,rating\n1,10,9\n1,20,6\n2,10,7\n2,20,8\n"
            )
            (data / "anime.csv").write_text(
                "anime_id,name,genre,type,episodes,rating,members\n"
                "10,Fixture A,Drama,TV,12,8.0,100\n"
                "20,Fixture B,Comedy,Movie,1,7.0,200\n"
            )
            environment = {**os.environ, "PYTHONPATH": str(repository)}
            environment.pop("OPENAI_API_KEY", None)
            result = subprocess.run(
                [sys.executable, "-c", """
import json
import os
from types import SimpleNamespace
from unittest.mock import patch
from main import app

client = app.test_client()
assert client.get("/").status_code == 200
anime = client.get("/anime?genre=Drama")
assert anime.status_code == 200
assert [row["name"] for row in json.loads(anime.data)] == ["Fixture A"]
talk = client.get("/talk/1/synthetic-request")
assert talk.status_code == 503
assert "OPENAI_API_KEY" in talk.get_json()["error"]

completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Fixture B"))])
with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key"}), patch("main.askgpt", return_value=completion) as askgpt:
    talk = client.get("/talk/1/synthetic-request")
    assert talk.status_code == 200
    assert talk.get_json() == {"rec": "Fixture B"}
    assert "synthetic-request" in askgpt.call_args.args[0]
"""],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
