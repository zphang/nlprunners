class ExtendedDataClassMixin:

    @property
    def fields(self):
        return list(self.__dataclass_fields__)

    def asdict(self):
        return {
            k: getattr(self, k)
            for k in self.fields
        }

    def new(self, **new_kwargs):
        kwargs = {
            k: v
            for k, v in self.asdict().items()
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)
