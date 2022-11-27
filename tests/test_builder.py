import unittest
import os
import json
import glob

import context  # noqa: F401
import builder


class Tests(unittest.TestCase):
    output_path = f'{os.path.dirname(os.path.abspath(__file__))}/../output/'

    @classmethod
    def setUpClass(cls):

        files = glob.glob(f"{cls.output_path}/*")
        for f in files:
            os.remove(f)
        print("Starting tests for builder")

    @classmethod
    def tearDownClass(cls):
        print("\nFinishing tests for builder")

    def test_all_shapes_generated(self):

        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/shapes.ndjson', 'r') as f:
            for ndjson_line in f.readlines():
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui']:
                    core = builder.Builder().factory(data)
                    core.get_piece(data)
                    filename = f"{data['name']}_piece".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.step"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.obj"))

    def test_all_technical_drawings_generated(self):

        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/shapes.ndjson', 'r') as f:
            for ndjson_line in f.readlines():
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui']:
                    core = builder.Builder().factory(data)
                    core.get_piece_technical_drawing(data, save_files=True)
                    filename = f"{data['name']}_piece".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    print(f"{self.output_path}/{filename}_TopView.svg")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_scaled_TopView.svg"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_scaled_FrontView.svg"))

    def test_get_families(self):
        
        families = builder.Builder().get_families()
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/shapes.ndjson', 'r') as f:
            for ndjson_line in f.readlines():
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui']:
                    self.assertTrue(data["family"] in list(families.keys()))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

