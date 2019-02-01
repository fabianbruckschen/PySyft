import numpy as np
class DPTensor():

    def __init__(self, data, max_values, min_values):

        self.data = data
        self.max_values = max_values
        self.min_values = min_values

    def minimum(self, other):

        # if other is a scalar, create a tensor with its value
        if(isinstance(other,(float,int))):
            _data = np.zeros_like(self.data) + other
            other = DPTensor(data=_data, max_values=_data, min_values=_data)

        _new_data = np.minimum(self.data, other.data)
        _new_max_values = np.minimum(self.max_values,other.max_values)
        _new_min_values = self.min_values

        return DPTensor(data=_new_data,
                        max_values=_new_max_values,
                        min_values=_new_min_values)

    def maximum(self, other):

        # if other is a scalar, create a tensor with its value
        if(isinstance(other,(float,int))):
            _data = np.zeros_like(self.data) + other
            other = DPTensor(data=_data, max_values=_data, min_values=_data)

        _new_data = np.maximum(self.data, other.data)
        _new_min_values = np.maximum(self.min_values,other.min_values)
        _new_max_values = self.max_values

        return DPTensor(data=_new_data,
                        max_values=_new_max_values,
                        min_values=_new_min_values)

    def __add__(self, other):
        # NOTE: This assumes that all entities in self.data and other.data are DIFFERENT

        _data = self.data + other.data

        # remember, it's not about the maximum value that .data could take on
        # which would be self._max_values + other._max_values, it's about how
        # the maximum amount that .data could CHANGE if an entity is removed.
        _max_values = np.maximum(self.max_values, other.max_values)
        _min_values = np.minimum(self.min_values, other.min_values)

        return DPTensor(data=_data,
                        max_values=_max_values,
                        min_values=_min_values)

    def __neg__(self):
        # NOTE: This assumes that all entities in self.data and other.data are DIFFERENT

        _data = -self.data

        return DPTensor(data=_data,
                        max_values=-self.min_values,
                        min_values=-self.max_values)

    def __sub__(self, other):
        # NOTE: This assumes that all entities in self.data and other.data are DIFFERENT
        return (-other) + self

    @property
    def sensitivity(self):
        return self.max_values - self.min_values


class DatasetTensor(DPTensor):

    #TODO: consider epsilons over epsilon
    def __init__(self, data, entities, epsilon, max_values=None, min_values=None):

        self.data = data
        self.epsilon = epsilon
        self.entities = entities

        if max_values is None:
            max_values = np.inf + np.zeros_like(self.data)
        self.max_values = max_values

        if min_values is None:
            min_values = -np.inf + np.zeros_like(self.data)
        self.min_values = min_values

    def laplace(self):
        noise = [np.random.laplace(self.sensitivity[idx] / self.epsilon[idx], 1,
            1)[0] for idx, _ in enumerate(self.data)]
        print(noise)


