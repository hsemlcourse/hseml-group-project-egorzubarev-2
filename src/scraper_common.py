from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PRICE_PATTERN = re.compile(r"(\d[\d\s]{1,12})\s*₽")
SCREEN_DIAGONAL_PATTERN = re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*\"")
RESOLUTION_PATTERN = re.compile(r"(\d{3,4})\s*[xх]\s*(\d{3,4})", re.IGNORECASE)
RAM_PATTERN = re.compile(r"(?:RAM|Оперативная память:?)(?:\s*[:,-])?\s*(\d{1,3})\s*ГБ", re.IGNORECASE)
RAM_FALLBACK_PATTERN = re.compile(r"\b(\d{1,3})\s*ГБ\s*(?:LPDDR\d+x?|DDR\d+x?)\b", re.IGNORECASE)
RAM_TYPE_PATTERN = re.compile(r"\b(LPDDR\d+x?|DDR\d+x?)\b", re.IGNORECASE)
SSD_TB_PATTERNS = [
    re.compile(r"SSD\s*(\d{1,2})\s*ТБ", re.IGNORECASE),
    re.compile(r"(\d{1,2})\s*ТБ\s*SSD", re.IGNORECASE),
]
SSD_GB_PATTERNS = [
    re.compile(r"SSD\s*(\d{2,4})\s*ГБ", re.IGNORECASE),
    re.compile(r"(\d{2,4})\s*ГБ\s*SSD", re.IGNORECASE),
]
WEIGHT_PATTERN = re.compile(r"(\d(?:[.,]\d{1,2})?)\s*кг", re.IGNORECASE)
CPU_PATTERN = re.compile(
    r"(Intel\s+Core(?:\s+Ultra)?\s+[^,;\[\]]+|"
    r"Intel\s+[^,;\[\]]+|"
    r"AMD\s+Ryzen(?:\s+AI)?\s+[^,;\[\]]+|"
    r"Apple\s+M\d(?:\s+Pro|\s+Max|\s+Ultra)?(?:\s+\d+\s*core)?|"
    r"Qualcomm\s+Snapdragon\s+[^,;\[\]]+)",
    re.IGNORECASE,
)
GPU_PATTERN = re.compile(
    r"(NVIDIA\s+GeForce\s+RTX\s*[A-Za-z0-9 -]+|"
    r"NVIDIA\s+GeForce\s+[A-Za-z0-9 -]+|"
    r"Intel\s+(?:Iris\s+Xe|Arc(?:\s+Graphics)?|UHD\s+Graphics|Graphics)|"
    r"AMD\s+Radeon(?:\s+[A-Za-z0-9 -]+)?|"
    r"Qualcomm\s+Adreno)",
    re.IGNORECASE,
)

MATRIX_TYPES = [
    "Retina XDR",
    "Mini LED",
    "Retina",
    "OLED",
    "IPS",
    "TN",
    "VA",
]

OS_PATTERNS = [
    ("windows 11 home", "Windows 11 Home"),
    ("windows 11 pro", "Windows 11 Pro"),
    ("windows 11", "Windows 11"),
    ("windows 10", "Windows 10"),
    ("без операционной системы", "No OS"),
    ("без ос", "No OS"),
    ("freedos", "FreeDOS"),
    ("dos", "DOS"),
    ("linux", "Linux"),
    ("ubuntu", "Ubuntu"),
    ("macos", "macOS"),
]

BRANDS = [
    "APPLE",
    "ASUS",
    "ACER",
    "HUAWEI",
    "HONOR",
    "LENOVO",
    "MSI",
    "HP",
    "DELL",
    "DIGMA PRO",
    "DIGMA",
    "THUNDEROBOT",
    "TECNO",
    "SAMSUNG",
    "GIGABYTE",
    "CHUWI",
    "IRU",
    "OSIO",
    "DEXP",
    "INFINIX",
    "ARDOR GAMING",
    "XIAOMI",
    "MAIBENBEN",
    "AORUS",
]


@dataclass(slots=True)
class LaptopRecord:
    source: str
    source_id: str
    product_url: str
    title: str
    price_rub: int | None = None
    brand: str | None = None
    screen_diagonal_inch: float | None = None
    screen_resolution: str | None = None
    matrix_type: str | None = None
    cpu: str | None = None
    gpu: str | None = None
    ram_gb: int | None = None
    ram_type: str | None = None
    storage_gb: int | None = None
    os: str | None = None
    weight_kg: float | None = None
    raw_specs: dict[str, Any] = field(default_factory=dict)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()
    return text


def parse_price_rub(text: str) -> int | None:
    match = PRICE_PATTERN.search(clean_text(text))
    if not match:
        return None
    return int(re.sub(r"\D", "", match.group(1)))


def parse_storage_gb(text: str) -> int | None:
    normalized = clean_text(text)
    tb_matches = [int(value) * 1024 for pattern in SSD_TB_PATTERNS for value in pattern.findall(normalized)]
    gb_matches = [int(value) for pattern in SSD_GB_PATTERNS for value in pattern.findall(normalized)]
    values = tb_matches + gb_matches
    return max(values) if values else None


def parse_brand(text: str) -> str | None:
    normalized = clean_text(text).upper()
    for brand in BRANDS:
        if brand in normalized:
            return brand.title() if brand.isupper() else brand
    stripped = re.sub(r"^(?:\d{1,2}(?:[.,]\d{1,2})?\"\s+)?(?:НОУТБУК|УЛЬТРАБУК)\s+", "", normalized)
    first_token = stripped.split(" ", maxsplit=1)[0].strip(" ,")
    return first_token.title() if first_token else None


def parse_matrix_type(text: str) -> str | None:
    normalized = clean_text(text)
    upper_text = normalized.upper()
    for matrix_type in MATRIX_TYPES:
        if matrix_type.upper() in upper_text:
            return matrix_type
    return None


def parse_os(text: str) -> str | None:
    normalized = clean_text(text)
    lower_text = normalized.lower()
    for pattern, canonical in OS_PATTERNS:
        if pattern in lower_text:
            return canonical
    return None


def parse_resolution(text: str) -> str | None:
    match = RESOLUTION_PATTERN.search(clean_text(text))
    if not match:
        return None
    return f"{match.group(1)}x{match.group(2)}"


def parse_screen_diagonal(text: str) -> float | None:
    match = SCREEN_DIAGONAL_PATTERN.search(clean_text(text))
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _parse_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(clean_text(text))
    if not match:
        return None
    return int(match.group(1))


def _parse_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(clean_text(text))
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _parse_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(clean_text(text))
    if not match:
        return None
    return clean_text(match.group(1))


def normalize_cpu(value: str | None) -> str | None:
    if not value:
        return None
    normalized = clean_text(value)
    normalized = re.sub(r"\s+\d(?:[.,]\d+)?\s*ГГц.*$", "", normalized, flags=re.IGNORECASE)
    return normalized.strip(" ,;")


def parse_core_features(text: str) -> dict[str, Any]:
    normalized = clean_text(text)
    ram_type = _parse_match(RAM_TYPE_PATTERN, normalized)
    ram_gb = _parse_int(RAM_PATTERN, normalized) or _parse_int(RAM_FALLBACK_PATTERN, normalized)
    return {
        "brand": parse_brand(normalized),
        "screen_diagonal_inch": parse_screen_diagonal(normalized),
        "screen_resolution": parse_resolution(normalized),
        "matrix_type": parse_matrix_type(normalized),
        "cpu": normalize_cpu(_parse_match(CPU_PATTERN, normalized)),
        "gpu": _parse_match(GPU_PATTERN, normalized),
        "ram_gb": ram_gb,
        "ram_type": ram_type.upper() if ram_type else None,
        "storage_gb": parse_storage_gb(normalized),
        "os": parse_os(normalized),
        "weight_kg": _parse_float(WEIGHT_PATTERN, normalized),
    }


def merge_text_features(*parts: str) -> dict[str, Any]:
    merged = clean_text(" ".join(part for part in parts if part))
    return parse_core_features(merged)
