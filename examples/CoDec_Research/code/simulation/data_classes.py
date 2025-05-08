class SimulationResults:
    def __init__(self, simulation_name: str, simulation_id: int):
        self.simulation_name = simulation_name
        self.simulation_id = simulation_id
        self.simulation_results = []
        self.simulation_parameters = {}
        self.simulation_data = {}
        self.simulation_metadata = {}

    def add_result(self, result):
        self.simulation_results.append(result)

    def set_parameters(self, parameters):
        self.simulation_parameters = parameters

    def set_data(self, data):
        self.simulation_data = data

    def set_metadata(self, metadata):
        self.simulation_metadata = metadata