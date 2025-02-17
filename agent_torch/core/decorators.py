# File: agent_torch/core/decorators.py

from functools import wraps

def with_behavior(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Instead of setting to None, initialize with class behavior if it exists
        self._behavior = getattr(cls, '_class_behavior', None)
        # print(f"Initialized {cls.__name__} with behavior: {self._behavior}")
        # print(f"Class behavior exists: {hasattr(cls, '_class_behavior')}")
        # if hasattr(cls, '_class_behavior'):
        #     print(f"Class behavior value: {cls._class_behavior}")

    @classmethod
    def set_behavior(cls, behavior):
        cls._class_behavior = behavior

    def get_behavior(self):
        return self._behavior if hasattr(self, '_behavior') else getattr(self.__class__, '_class_behavior', None)

    def set_instance_behavior(self, behavior):
        self._behavior = behavior

    cls.__init__ = new_init
    cls.set_behavior = set_behavior
    cls.behavior = property(get_behavior, set_instance_behavior)

    return cls

# You can add other decorators here if needed