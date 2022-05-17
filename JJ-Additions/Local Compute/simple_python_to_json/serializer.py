from collections import namedtuple, Iterable, OrderedDict
import numpy as np
import simplejson as json

def isnamedtuple(obj):
    """Heuristic check if an object is a namedtuple."""
    return isinstance(obj, tuple) \
           and hasattr(obj, "_fields") \
           and hasattr(obj, "_asdict") \
           and callable(obj._asdict)

def serialize(data):
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [serialize(val) for val in data]
    if isinstance(data, OrderedDict):
        return {"py/collections.OrderedDict":
                [[serialize(k), serialize(v)] for k, v in data.iteritems()]}
    if isnamedtuple(data):
        return {"py/collections.namedtuple": {
            "type":   type(data).__name__,
            "fields": list(data._fields),
            "values": [serialize(getattr(data, f)) for f in data._fields]}}
    if isinstance(data, dict):
        if all(isinstance(k, basestring) for k in data):
            return {k: serialize(v) for k, v in data.iteritems()}
        return {"py/dict": [[serialize(k), serialize(v)] for k, v in data.iteritems()]}
    if isinstance(data, tuple):
        return {"py/tuple": [serialize(val) for val in data]}
    if isinstance(data, set):
        return {"py/set": [serialize(val) for val in data]}
    if isinstance(data, type):
        return {"py/numpy.type": data.__name__}
    if isinstance(data, np.integer):
        return {"py/numpy.int": int(data)}
    if isinstance(data, np.float):
        return {"py/numpy.float": data.hex()}
    if isinstance(data, np.ndarray):
        return {"py/numpy.ndarray": {
            "values": data.tolist(),
            "dtype":  str(data.dtype)}}
    raise TypeError("Serialization of Type %s is not supported." % type(data))

def restore(dct):
    if "py/dict" in dct:
        return dict(dct["py/dict"])
    if "py/collections.OrderedDict" in dct:
        return OrderedDict(dct["py/collections.OrderedDict"])
    if "py/tuple" in dct:
        return tuple(dct["py/tuple"])
    if "py/set" in dct:
        return set(dct["py/set"])
    if "py/collections.namedtuple" in dct:
        data = dct["py/collections.namedtuple"]
        return namedtuple(data["type"], data["fields"])(*data["values"])
    if "py/numpy.type" in dct:
        return np.dtype(dct["py/numpy.type"]).type
    if "py/numpy.int" in dct:
        return np.int32(dct["py/numpy.int"])
    if "py/numpy.float" in dct:
        return np.float64.fromhex(dct["py/numpy.float"])
    if "py/numpy.ndarray" in dct:
        data = dct["py/numpy.ndarray"]
        return np.array(data["values"], dtype=data["dtype"])
    return dct

def data_to_json(data):
    return json.dumps(serialize(data))

def json_to_data(s):
    return json.loads(s, object_hook=restore)

def nested_equal(v1, v2):
    """Compares two complex data structures.

    This handles the case where numpy arrays are leaf nodes.
    """
    if isinstance(v1, basestring) or isinstance(v2, basestring):
        return v1 == v2
    if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
        return np.array_equal(v1, v2)
    if isinstance(v1, dict) and isinstance(v2, dict):
        return nested_equal(v1.items(), v2.items())
    if isinstance(v1, Iterable) and isinstance(v2, Iterable):
        return all(nested_equal(sub1, sub2) for sub1, sub2 in zip(v1, v2))
    return v1 == v2