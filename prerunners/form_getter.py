import pandas as pd


def form_getter(pieceTitle: str) -> str:
    common_form_mapping = {
        "sonatina": "sonata",
        "sonetto": "sonata",
        "sonate": "sonata",
        "sonata": "sonata",
        "sonaten": "sonata",
        "prélude": "prelude",
        "prelude": "prelude",
        "nocturnes": "nocturne",
        "nocturne": "nocturne",
        "etude": "etude",
        "étude": "etude",
        "études": "etude",
        "vals": "waltz",
        "valse": "waltz",
        "valses": "waltz",
        "waltz": "waltz",
    }
    pieceTitle_lower = pieceTitle.lower()

    for spelling, form in common_form_mapping.items():
        if spelling in pieceTitle_lower:
            return form

    return "other"
