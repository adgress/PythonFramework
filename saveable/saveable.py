__author__ = 'Aubrey'

import abc
import configs.base_configs as base_configs
class Saveable(object):
    #__metaclass__ = abc.ABCMeta
    def __init__(self,configs=base_configs.Configs()):
        self._name_params = {}
        self.configs = configs
        pass

    @property
    def name_params(self):
        return self._name_params

    @property
    def prefix(self):
        return "No Name"

    @property
    def name_string(self):
        s = self.prefix
        field_delim = '_'
        field_value_delim = '='
        default_options = {
            'include_field_name': True
        }
        for k, value in self.name_params.items():
            d = default_options.copy()
            d.update(value)
            s += '_'
            if d['include_field_name']:
                s += k + '='
            s += str(getattr(self,k))
        return s



if __name__ == "__main__":
    s = Saveable()
    s.x = 10
    s._name_params['x'] = {}
    print s.name_string

    s._name_params['x']['include_field_name'] = False
    print s.name_string
    print 'Test Run'
