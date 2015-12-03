__author__ = 'Aubrey'
import HelperFunctions
def remove_suffix(s, suffix):
    if s[-len(suffix):] == suffix:
        s = s[:-len(suffix)]
    return s

def get_attributes_from_string(str, pair_delim=',', key_value_pair_delim='='):
    splits = str.split(pair_delim)
    key_value_pairs = [s.split(key_value_pair_delim) for s in splits]
    d = dict(key_value_pairs)
    return d

def stringify_attributes(obj, params_to_show, delim=',', pair_delim='='):

    kwargs = dict(zip(params_to_show, map(getattr,len(params_to_show)*[obj], params_to_show)))

    name = stringify_key_val_pairs(kwargs)

    return name

def stringify_key_val_pairs(kwargs):
    name = ''
    for key,value in kwargs.iteritems():
        if name != '':
            name += ','

        if not isinstance(value,str):
            if isinstance(value,list):
                if not isinstance(value[0], int):
                    value_str = ''
                    for val in value:
                        value_str += str(val)
                    value = value_str

                else:
                    value = _get_range_of_values_from_a_list(value)
            else:
                value = str(value)

        name += key + '=' + value

    return name


def _get_range_of_values_from_a_list(the_list):
        """
        Stringify a sorted list of numbers to the form of "X-Y" where X is the first element and Y is the last element
        :param the_list: a list of numbers
        :return: a string of the form "X-Y"
        """
        the_list = HelperFunctions.convert_to_list(the_list)
        assert HelperFunctions.is_list_of_floats(the_list)
        if len(the_list) == 1:
            return str(the_list[0])
        first_number = 0
        last_number = -1
        return str(the_list[first_number]) + '-' + str(the_list[last_number])

#'Struct' class used to convert from dicts to objects to facilitate stringification
class DictStruct:
    pass

def stringify_dict(d):
    s = DictStruct()
    for key, value in d.items():
        setattr(s, key, value)
    return stringify_attributes(s, d.keys())

def make_dict_from_attributes(obj, attributes):
    d = {}
    for key in attributes:
        d[key] = getattr(obj,key)
    return d

def stringify_key_value_pairs(key_value_list):
    return stringify_dict(dict(key_value_list))

#In Configs attributes can be either a value or a list.  These functions are for stringifying such attributes
def stringify_key_and_list_or_value(value_name, list_or_value, pair_delim='='):
    keys = []
    try:
        for v in list_or_value:
            keys.append(value_name + pair_delim + str(v))
    except:
        keys.append(value_name + pair_delim + str(list_or_value))
    return keys

def stringify_list_or_value(list_or_value, delim='-'):
    try:
        s = ''
        for value in list_or_value:
            if s != '':
                s += delim
            s += str(value)
        return '[' + s + ']'
    except:
        #In case l isn't a list (e.g. l is an int or float)
        return str(list_or_value)

def string_matches_query(s, query_dict):
    properties = get_attributes_from_string(s)
    query_keys = query_dict.keys()
    if not set(query_keys) <= set(properties.keys()):
        return False
    for key in query_keys:
        query_value = query_dict[key]
        if len(query_value) == 0:
            continue
        property_value = properties[key]
        if type(query_value) == set or type(query_value) == list:
            query_value_strings = {str(v) for v in query_value}
            if property_value not in query_value_strings:
                return False
        elif property_value != str(query_value):
            return False
    return True