import warnings


class ExtendedDataClassMixin:

    @classmethod
    def get_fields(cls):
        return list(cls.__dataclass_fields__)

    @classmethod
    def get_annotations(cls):
        return cls.__annotations__

    def asdict(self):
        warnings.warn("Use .to_dict()", DeprecationWarning)
        return self.to_dict()

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in self.get_fields()
        }

    @classmethod
    def from_dict(cls, kwargs):
        return cls(**kwargs)

    def new(self, **new_kwargs):
        kwargs = {
            k: v
            for k, v in self.asdict().items()
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)
