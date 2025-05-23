import unittest
import os
import json
import glob

import context  # noqa: F401
import builder
import copy
import PyMKF


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

    def test_all_magnetic_generated(self):
        dummyGapping = [
            {
                'length': 0.0001,
                'type': 'additive'
            },
            {
                'length': 0.0001,
                'type': 'additive'
            },
            {
                'length': 0.0001,
                'type': 'additive'
            }
        ]

        dummyCore = {
            "functionalDescription": {
                "name": "dummy",
                "type": "two-piece set",
                "material": "N97",
                "shape": None,
                "gapping": dummyGapping,
                "numberStacks": 1
            }
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['pq', 'rm', 'e']:
                    continue 
                core = copy.deepcopy(dummyCore)
                if data['family'] in ['t']:
                    core['functionalDescription']['type'] = "toroidal"
                if data['family'] in ['ut']:
                    core['functionalDescription']['type'] = "closed shape"
                core['functionalDescription']['shape'] = data

                if data['family'] in ['t']:
                    core['functionalDescription']['gapping'] = []
                else:
                    gapping = []
                    core_datum = PyMKF.calculate_core_data(core, False)
                    core_datum['processedDescription'] = PyMKF.calculate_core_processed_description(core)
                    for column_index, column in enumerate(core_datum['processedDescription']['columns']):
                        aux = copy.deepcopy(dummyGapping[column_index])
                        aux['coordinates'] = column['coordinates']
                        gapping.append(aux)
                    core['functionalDescription']['gapping'] = gapping

                core_datum = PyMKF.calculate_core_data(core, False)
                # import pprint
                # pprint.pprint(core_datum['processedDescription'])
                bobbin_name_aliases = core_datum['functionalDescription']['shape']['aliases']
                bobbin_name = None
                if bobbin_name_aliases:  # Check if there are any aliases
                    for alias in bobbin_name_aliases:
                        if alias.startswith("E"):
                            bobbin_name = "Bobbin " + alias.replace(" ", "")  #TODO: This is only the case for E cores?
                        else:
                            bobbin_name = "Bobbin " + alias
                if bobbin_name is None:
                    continue
                bobbin_datanum = PyMKF.find_bobbin_by_name(bobbin_name)
                if bobbin_datanum is None or (isinstance(bobbin_datanum, dict) and any(isinstance(value, str) and 'Exception' in value for value in bobbin_datanum.values())):
                    print(f"Bobbin '{bobbin_name}' not found in the database.")
                    continue   
                
                core = builder.Builder("FreeCAD").get_magnetic(data['name'], core_datum['geometricalDescription'])                
                print(core)
                filename = f"{data['name']}_magnetic".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.step"))
                self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.obj") or os.path.exists(f"{self.output_path}/{filename}.stl"))

  

    def test_all_bobbins_generated(self):

        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/bobbins.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["functionalDescription"]["family"] in ['pq', 'rm', 'e']:
                    # if data['family'] != "p":
                    # if data['name'] != "Bobbin E25/7":
                        # continue

                    print(data["name"])
                    core = builder.Builder("FreeCAD").factory(data["functionalDescription"])
                    core.get_bobbin(data["functionalDescription"], save_files=True, export_files=True)
                    filename = f"{data['functionalDescription']['shape']}_bobbin".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.step"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.obj") or os.path.exists(f"{self.output_path}/{filename}.stl"))

    def test_all_shapes_generated(self):

        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'pqi', 'ut']:
                    # if data['name'] != "T 22/14/13":
                    # if data['family'] != "c":
                    # if data['family'] != "planar el":
                        # continue

                    print(data["name"])
                    core = builder.Builder().factory(data)
                    core.get_piece(data, save_files=True, export_files=True)
                    filename = f"{data['name']}_piece".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.step"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.obj") or os.path.exists(f"{self.output_path}/{filename}.stl"))

    def test_all_technical_drawings_generated(self):
        colors = {
            "projection_color": "#d4d4d4",
            "dimension_color": "#d4d4d4"
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'pqi']:
                    # if data['family'] != "ut":
                    #     continue
                    core = builder.Builder().factory(data)
                    print(data["name"])
                    core.get_piece_technical_drawing(data, colors=colors, save_files=True)
                    filename = f"{data['name']}_piece".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_scaled_TopView.svg"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_scaled_FrontView.svg"))

    def test_get_families(self):
        
        families = builder.Builder().get_families()
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'pqi']:
                    self.assertTrue(data["family"] in list(families.keys()))

    def test_all_subtractive_gapped_cores_generated(self):
        dummyGapping = [
            {
                'length': 0.001,
                'type': 'subtractive'
            },
            {
                'length': 0.002,
                'type': 'subtractive'
            },
            {
                'length': 0,
                'type': 'subtractive'
            }
        ]

        dummyCore = {
            "functionalDescription": {
                "name": "dummy",
                "type": "two-piece set",
                "material": "N97",
                "shape": None,
                "gapping": dummyGapping,
                "numberStacks": 3
            }
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'ut', 'pqi']:
                    if data['family'] != "c":
                    # if data['name'] != "T 22/14/13":
                        continue

                    core = copy.deepcopy(dummyCore)
                    if data['family'] in ['t']:
                        core['functionalDescription']['type'] = "toroidal"
                    if data['family'] in ['ut']:
                        core['functionalDescription']['type'] = "closed shape"
                    core['functionalDescription']['shape'] = data

                    if data['family'] in ['t']:
                        core['functionalDescription']['gapping'] = []
                    else:
                        gapping = []
                        core_datum = PyMKF.calculate_core_data(core, False)
                        for column_index, column in enumerate(core_datum['processedDescription']['columns']):
                            aux = copy.deepcopy(dummyGapping[column_index])
                            aux['coordinates'] = column['coordinates']
                            gapping.append(aux)

                        core['functionalDescription']['gapping'] = gapping
                    core_datum = PyMKF.calculate_core_data(core, False)
                    core = builder.Builder().get_core(data['name'], core_datum['geometricalDescription'])
                    print(core)
                    filename = f"{data['name']}_core".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.step"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.obj") or os.path.exists(f"{self.output_path}/{filename}.stl"))

    def test_all_subtractive_distributed_gapped_cores_generated(self):
        dummyGapping = [
            {
                'length': 0.001,
                'type': 'subtractive'
            },
            {
                'length': 0.0005,
                'type': 'subtractive'
            },
            {
                'length': 0.002,
                'type': 'subtractive'
            },
            {
                'length': 0.00005,
                'type': 'residual'
            },
            {
                'length': 0.00005,
                'type': 'residual'
            }
        ]

        dummyCore = {
            "functionalDescription": {
                "name": "dummy",
                "type": "two-piece set",
                "material": "N97",
                "shape": None,
                "gapping": dummyGapping,
                "numberStacks": 1
            }
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'ut', 'pqi']:

                    core = copy.deepcopy(dummyCore)
                    if data['family'] in ['t']:
                        core['functionalDescription']['type'] = "toroidal"
                    if data['family'] in ['ut']:
                        core['functionalDescription']['type'] = "closed shape"
                    core['functionalDescription']['shape'] = data

                    if data['family'] in ['t']:
                        core['functionalDescription']['gapping'] = []

                    core_datum = PyMKF.calculate_core_data(core, False)
                    core = builder.Builder().get_core(data['name'], core_datum['geometricalDescription'])
                    print(core)
                    filename = f"{data['name']}_core".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.step"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.obj") or os.path.exists(f"{self.output_path}/{filename}.stl"))

    def test_all_additive_gapped_cores_generated(self):
        dummyGapping = [
            {
                'length': 0.0001,
                'type': 'additive'
            },
            {
                'length': 0.0001,
                'type': 'additive'
            },
            {
                'length': 0.0001,
                'type': 'additive'
            }
        ]

        dummyCore = {
            "functionalDescription": {
                "name": "dummy",
                "type": "two-piece set",
                "material": "N97",
                "shape": None,
                "gapping": dummyGapping,
                "numberStacks": 1
            }
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'ut', 'pqi']:

                    core = copy.deepcopy(dummyCore)
                    if data['family'] in ['t']:
                        core['functionalDescription']['type'] = "toroidal"
                    if data['family'] in ['ut']:
                        core['functionalDescription']['type'] = "closed shape"
                    core['functionalDescription']['shape'] = data

                    if data['family'] in ['t']:
                        core['functionalDescription']['gapping'] = []
                    else:
                        gapping = []
                        core_datum = PyMKF.calculate_core_data(core, False)
                        core_datum['processedDescription'] = PyMKF.calculate_core_processed_description(core)
                        for column_index, column in enumerate(core_datum['processedDescription']['columns']):
                            aux = copy.deepcopy(dummyGapping[column_index])
                            aux['coordinates'] = column['coordinates']
                            gapping.append(aux)
                        core['functionalDescription']['gapping'] = gapping

                    core_datum = PyMKF.calculate_core_data(core, False)
                    # import pprint
                    # pprint.pprint(core_datum['processedDescription'])
                    # pprint.pprint(core_datum['geometricalDescription'])
                    core = builder.Builder().get_core(data['name'], core_datum['geometricalDescription'])
                    print(core)
                    filename = f"{data['name']}_core".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.step"))
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}.obj") or os.path.exists(f"{self.output_path}/{filename}.stl"))

    def test_all_additive_technical_drawing_cores_generated(self):
        dummyGapping = [
            {
                'length': 0.001,
                'type': 'additive'
            },
            {
                'length': 0.001,
                'type': 'additive'
            },
            {
                'length': 0.001,
                'type': 'additive'
            }
        ]

        dummyCore = {
            "functionalDescription": {
                "name": "dummy",
                "type": "two-piece set",
                "material": "N97",
                "shape": None,
                "gapping": dummyGapping,
                "numberStacks": 1
            }
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'ut', 'pqi', 't']:
                    core = copy.deepcopy(dummyCore)
                    if data['family'] in ['t']:
                        core['functionalDescription']['type'] = "toroidal"
                    if data['family'] in ['ut']:
                        core['functionalDescription']['type'] = "closed shape"
                    core['functionalDescription']['shape'] = data

                    if data['family'] in ['t']:
                        core['functionalDescription']['gapping'] = []
                    else:
                        gapping = []
                        core_datum = PyMKF.calculate_core_data(core, False)
                        for column_index, column in enumerate(core_datum['processedDescription']['columns']):
                            aux = copy.deepcopy(dummyGapping[column_index])
                            aux['coordinates'] = column['coordinates']
                            gapping.append(aux)
                        core['functionalDescription']['gapping'] = gapping

                    print(data["name"])
                    core_datum = PyMKF.calculate_core_data(core, False)
                    core = builder.Builder().get_core_gapping_technical_drawing(data['name'], core_datum)

                    filename = f"{data['name']}".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    print(f"{self.output_path}/{filename}_core_gaps_FrontView.svg")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_core_gaps_FrontView.svg"))

    def test_all_subtractive_technical_drawing_cores_generated(self):
        dummyGapping = [
            {
                'length': 0.001,
                'type': 'subtractive'
            },
            {
                'length': 0.002,
                'type': 'subtractive'
            },
            {
                'length': 0.000005,
                'type': 'subtractive'
            }
        ]

        dummyCore = {
            "functionalDescription": {
                "name": "dummy",
                "type": "two-piece set",
                "material": "N97",
                "shape": None,
                "gapping": dummyGapping,
                "numberStacks": 1
            }
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'ut', 'pqi']:

                    core = copy.deepcopy(dummyCore)
                    if data['family'] in ['t']:
                        core['functionalDescription']['type'] = "toroidal"
                    if data['family'] in ['ut']:
                        core['functionalDescription']['type'] = "closed shape"
                    core['functionalDescription']['shape'] = data

                    if data['family'] in ['t']:
                        core['functionalDescription']['gapping'] = []
                    else:
                        gapping = []
                        core_datum = PyMKF.calculate_core_data(core, False)
                        for column_index, column in enumerate(core_datum['processedDescription']['columns']):
                            aux = copy.deepcopy(dummyGapping[column_index])
                            aux['coordinates'] = column['coordinates']
                            gapping.append(aux)
                        core['functionalDescription']['gapping'] = gapping

                    core_datum = PyMKF.calculate_core_data(core, False)
                    core = builder.Builder().get_core_gapping_technical_drawing(data['name'], core_datum)

                    filename = f"{data['name']}".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    print(f"{self.output_path}/{filename}_core_gaps_FrontView.svg")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_core_gaps_FrontView.svg"))

    def test_all_subtractive_distributed_technical_drawing_cores_generated(self):
        dummyGapping = [
            {
                'length': 0.001,
                'type': 'subtractive'
            },
            {
                'length': 0.0005,
                'type': 'subtractive'
            },
            {
                'length': 0.002,
                'type': 'subtractive'
            },
            {
                'length': 0.00005,
                'type': 'residual'
            },
            {
                'length': 0.00005,
                'type': 'residual'
            }
        ]

        dummyCore = {
            "functionalDescription": {
                "name": "dummy",
                "type": "two-piece set",
                "material": "N97",
                "shape": None,
                "gapping": dummyGapping,
                "numberStacks": 1
            }
        }
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../MAS/data/core_shapes.ndjson', 'r') as f:
            for ndjson_line in f:
                data = json.loads(ndjson_line)
                if data["family"] not in ['ui', 'ut', 'pqi']:
                # if data["family"] in ['p']:
                    print(data["name"])
                    core = copy.deepcopy(dummyCore)
                    if data['family'] in ['t']:
                        core['functionalDescription']['type'] = "toroidal"
                    if data['family'] in ['ut']:
                        core['functionalDescription']['type'] = "closed shape"
                    core['functionalDescription']['shape'] = data

                    if data['family'] in ['t']:
                        core['functionalDescription']['gapping'] = []

                    core_datum = PyMKF.calculate_core_data(core, False)
                    core = builder.Builder().get_core_gapping_technical_drawing(data['name'], core_datum)

                    filename = f"{data['name']}".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
                    # print(f"{self.output_path}/{filename}_core_gaps_FrontView.svg")
                    self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_core_gaps_FrontView.svg"))

    def test_0(self):
        core = {'functionalDescription': {'bobbin': None,
                                           'gapping': [{'area': 0.000315,
                                                        'coordinates': [0.0, 0.0005, 0.0],
                                                        'distanceClosestNormalSurface': 0.009,
                                                        'length': 0.001,
                                                        'sectionDimensions': [0.02, 0.02],
                                                        'shape': 'round',
                                                        'type': 'subtractive'},
                                                       {'area': 0.000238,
                                                        'coordinates': [0.0215, 0.0, 0.0],
                                                        'distanceClosestNormalSurface': 0.0095,
                                                        'length': 1e-05,
                                                        'sectionDimensions': [0.004, 0.059501],
                                                        'shape': 'irregular',
                                                        'type': 'residual'},
                                                       {'area': 0.000238,
                                                        'coordinates': [-0.0215, 0.0, 0.0],
                                                        'distanceClosestNormalSurface': 0.0095,
                                                        'length': 1e-05,
                                                        'sectionDimensions': [0.004, 0.059501],
                                                        'shape': 'irregular',
                                                        'type': 'residual'}],
                                           'material': '3C97',
                                           'name': 'default',
                                           'numberStacks': 1,
                                           'shape': {'aliases': [],
                                                     'dimensions': {'A': 0.047,
                                                                    'B': 0.014,
                                                                    'C': 0.0,
                                                                    'D': 0.0095,
                                                                    'E': 0.039,
                                                                    'F': 0.02,
                                                                    'G': 0.0081,
                                                                    'H': 0.0055},
                                                     'family': 'p',
                                                     'familySubtype': '2',
                                                     'magneticCircuit': None,
                                                     'name': 'Custom',
                                                     'type': 'custom'},
                                           'type': 'two-piece set'},
                         'geometricalDescription': [{'coordinates': [0.0, 0.0, -0.0],
                                                     'dimensions': None,
                                                     'machining': [{'coordinates': [0.0, 0.0005, 0.0],
                                                                    'length': 0.001}],
                                                     'material': '3C97',
                                                     'rotation': [3.141592653589793,
                                                                  3.141592653589793,
                                                                  0.0],
                                                     'shape': {'aliases': [],
                                                               'dimensions': {'A': 0.047,
                                                                              'B': 0.014,
                                                                              'C': 0.0,
                                                                              'D': 0.0095,
                                                                              'E': 0.039,
                                                                              'F': 0.02,
                                                                              'G': 0.0081,
                                                                              'H': 0.0055},
                                                               'family': 'p',
                                                               'familySubtype': '2',
                                                               'magneticCircuit': None,
                                                               'name': 'Custom',
                                                               'type': 'custom'},
                                                     'type': 'half set'},
                                                    {'coordinates': [0.0, -0.0, -0.0],
                                                     'dimensions': None,
                                                     'machining': [{'coordinates': [0.0, 0.0005, 0.0],
                                                                    'length': 0.001}],
                                                     'material': '3C97',
                                                     'rotation': [0.0, 0.0, 0.0],
                                                     'shape': {'aliases': [],
                                                               'dimensions': {'A': 0.047,
                                                                              'B': 0.014,
                                                                              'C': 0.0,
                                                                              'D': 0.0095,
                                                                              'E': 0.039,
                                                                              'F': 0.02,
                                                                              'G': 0.0081,
                                                                              'H': 0.0055},
                                                               'family': 'p',
                                                               'familySubtype': '2',
                                                               'magneticCircuit': None,
                                                               'name': 'Custom',
                                                               'type': 'custom'},
                                                     'type': 'half set'}],
                         'processedDescription': {'columns': [{'area': 0.000315,
                                                               'coordinates': [0.0, 0.0, 0.0],
                                                               'depth': 0.02,
                                                               'height': 0.019,
                                                               'shape': 'round',
                                                               'type': 'central',
                                                               'width': 0.02},
                                                              {'area': 0.000238,
                                                               'coordinates': [0.0215, 0.0, 0.0],
                                                               'depth': 0.059501,
                                                               'height': 0.019,
                                                               'shape': 'irregular',
                                                               'type': 'lateral',
                                                               'width': 0.004},
                                                              {'area': 0.000238,
                                                               'coordinates': [-0.0215, 0.0, 0.0],
                                                               'depth': 0.059501,
                                                               'height': 0.019,
                                                               'shape': 'irregular',
                                                               'type': 'lateral',
                                                               'width': 0.004}],
                                                  'depth': 0.047,
                                                  'effectiveParameters': {'effectiveArea': 0.00035050517966366066,
                                                                          'effectiveLength': 0.07043961692540429,
                                                                          'effectiveVolume': 2.468945058587826e-05,
                                                                          'minimumArea': 0.00028657215486964393},
                                                  'height': 0.028,
                                                  'width': 0.047,
                                                  'windingWindows': [{'angle': None,
                                                                      'area': 0.0001805,
                                                                      'coordinates': [0.01, 0.0],
                                                                      'height': 0.019,
                                                                      'radialHeight': None,
                                                                      'width': 0.0095}]}}
        core = builder.Builder().get_core(core['functionalDescription']['shape']['name'], core['geometricalDescription'])

        # filename = f"{data['name']}".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
        # print(f"{self.output_path}/{filename}_core_gaps_FrontView.svg")
        # self.assertTrue(os.path.exists(f"{self.output_path}/{filename}_core_gaps_FrontView.svg"))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()



    # data = {'aliases': [],
    #         'dimensions': {'A': 0.0094,
    #                        'B': 0.0046,
    #                        'C': 0.0088,
    #                        'D': 0.0035,
    #                        'E': 0.0072,
    #                        'F': 0.003,
    #                        'G': 0.0,
    #                        'H': 0.0,
    #                        'K': 0.0015},
    #         'family': 'epx',
    #         'familySubtype': '1',
    #         'magneticCircuit': None,
    #         'name': 'Custom',
    #         'type': 'custom'}
    # core = builder.Builder().factory(data)
    # import pprint
    # pprint.pprint(data)
    # print("ea")
    # ea = core.get_piece_technical_drawing(data, save_files=True)
    # print("ea2")
    # filename = f"{data['name']}_piece".replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "__")
    # # print(ea)
    # # print(filename)
