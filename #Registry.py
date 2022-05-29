_module_dict = dict()


def register_module(name):
    def _register(cls):
        _module_dict[name] = cls
        return cls

    return _register


@register_module("one_class")
class OneTest(object):
    def __init__(self):
        self.name = "OneTest"


@register_module("two_class")
class TwoTest(object):
    def __init__(self):
        self.name = "TwoTest"


print(_module_dict)

if __name__ == '__main__':
    one_test = _module_dict['one_class']()
    print(one_test)
    print(one_test.name)