from src.utils.dataloader.dataloader import DataLoader
from src.utils.utils import DataTool


class LSEDTemporalDataLoader(DataLoader):
    def load_data(self, data_name: str, **kwargs):
        weight_shift = kwargs.get('weight_shift', 1)
        normalize = kwargs.get('normalize', False)
        dt = DataTool('connected_data_final', weight_shift=weight_shift, weight_normalize=normalize)

        data, weights = dt.get_data()

        data['weights'] = weights
        data['type_list'] = ['+'] * len(weights)

        return data

class LSEDSnapshotDataLoader(DataLoader):
    def load_data(self, data_name: str, **kwargs):
        weight_shift = kwargs.get('weight_shift', 1)
        normalize = kwargs.get('normalize', False)




        dt = DataTool('connected_data_final', weight_shift=weight_shift, weight_normalize=normalize)

        data, weights = dt.get_data()

        data['weights'] = weights
        data['type_list'] = ['+'] * len(weights)

        return data
