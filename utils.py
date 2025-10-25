def get_length(v: tuple[float, float]) -> float:
    return (v[0]**2 + v[1]**2)**0.5

def normalise(v: tuple[float, float]) -> tuple[float, float]:
    mag = get_length(v)
    if mag > 0:
        return (v[0] / mag, v[1] / mag)
    return (0.0, 0.0)

def set_magnitude(v: tuple[float, float], mag: float) -> tuple[float, float]:
    norm = normalise(v)
    return (norm[0] * mag, norm[1] * mag)

def dot(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1]