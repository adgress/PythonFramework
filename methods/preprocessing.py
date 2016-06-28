from data import data as data_lib


class IdentityPreprocessor(object):
    def __init__(self):
        pass

    def preprocess(self, data, configs):
        return data

class TargetOnlyPreprocessor(IdentityPreprocessor):
    def __init__(self):
        pass

    def preprocess(self, data, configs):
        target_labels = configs.target_labels
        assert target_labels.size == 1
        target_id = target_labels[0]
        data_set_ids = data.data_set_ids
        is_target_data = data_set_ids == target_id
        target_data = data.get_subset(is_target_data)
        return target_data