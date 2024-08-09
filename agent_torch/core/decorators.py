from typing import List, Any
from functools import wraps

def with_behavior(cls):
    class_behavior = None

    @wraps(cls.__init__)
    def new_init(self, *args, **kwargs):
        cls.__init__(self, *args, **kwargs)
        self.behavior = None

    @classmethod
    def set_behavior(cls, behavior):
        cls.class_behavior = behavior

    @property
    def behavior(self):
        return self._behavior if hasattr(self, '_behavior') else self.__class__.class_behavior

    @behavior.setter
    def behavior(self, value):
        self._behavior = value

    cls.__init__ = new_init
    cls.set_behavior = set_behavior
    cls.behavior = behavior

    return cls

