from typing import Any

def uneditable(cls: Any):
    """Makes a class's existing attributes uneditable by preventing reassignment or deletion.
    and function class can't be edited any value.

    Args:
        cls (Any): The class to be decorated.

    Raises:
        TypeError: If an attempt is made to reassign an existing attribute.
        TypeError: If an attempt is made to delete an existing attribute.

    Returns:
        type: The modified class with restricted attribute modification and deletion.
    """
    orig_setattr, orig_delattr = cls.__setattr__, cls.__delattr__

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise TypeError(f"'{type(self).__name__}' object does not support item assignment")
        return orig_setattr(self, name, value)

    def __delattr__(self, name: str) -> None:
        if hasattr(self, name):
            raise TypeError(f"'{type(self).__name__}' object does not support item deletion")
        return orig_delattr(self, name)

    cls.__setattr__ = __setattr__
    cls.__delattr__ = __delattr__
    return cls