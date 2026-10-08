import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

def _get_required(key: str) -> str:
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


@dataclass(frozen=True)
class Config:
    pghost: str
    pgport: int
    pgdatabase: str
    pguser: str
    banxico_token: str
    pgpassword: str = field(repr=False)
    machine_cost: int = 12239
    purchase_date: str = "30/05/2025"
    latitude: float = 19.4326
    longitude: float = -99.1332
    visits_calendar_url: str = ""
    # La app de Vacaciones vive ahora como el schema "vacaciones" dentro de la base
    # "casa" (antes era una base independiente llamada "vacaciones", ya no existe).
    vacation_database: str = "casa"


def load_config() -> Config:
    return Config(
        pghost=_get_required("PGHOST"),
        pgport=int(_get_required("PGPORT")),
        pgdatabase=_get_required("PGDATABASE"),
        pguser=_get_required("PGUSER"),
        pgpassword=_get_required("PGPASSWORD"),
        banxico_token=_get_required("BANXICO_TOKEN"),
        visits_calendar_url=os.environ.get("VISITS_CALENDAR_URL", ""),
        vacation_database=os.environ.get("VACATION_DATABASE", "casa"),
    )
