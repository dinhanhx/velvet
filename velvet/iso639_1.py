import functools

from iso639 import ALL_LANGUAGES, Language


@functools.lru_cache()
def get_iso639_1_list():
    def criteria(language: Language):
        if language.part1 is not None:
            return language

    return list(filter(criteria, ALL_LANGUAGES))


def __getattr__(name):
    if name == "iso639_1_list":
        return get_iso639_1_list()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
