import numpy as np


def get_slo_vector(profile: str) -> np.ndarray:
    """
    Return an unscaled SLO vector for the given profile.
    StateEncoder applies its own scaling, so we keep raw values here.
    """
    if profile == "quality_first":
        vec = np.array([4.0, 0.25, 0.25, 0.5], dtype=np.float32)
    elif profile == "cheap":
        vec = np.array([1.5, 2.5, 0.25, 0.75], dtype=np.float32)
    else:
        raise KeyError(f"Unknown SLO profile: {profile}")
    return vec


def infer_profile_from_slo_vec(slo_vec: np.ndarray) -> str:
    v = np.asarray(slo_vec, dtype=np.float32)
    if v[0] > v[1] and v[0] > v[2]:
        return "quality_first"
    return "cheap"


def slo_vector_to_weights(slo_vec: np.ndarray) -> dict[str, float]:
    v = np.asarray(slo_vec, dtype=np.float32)
    profile = infer_profile_from_slo_vec(v)
    refusal_penalty = 0.0 if profile == "quality_first" else 0.25
    return {
        "accuracy_w": float(v[0]),
        "cost_w": float(v[1]),
        "refusal_w": float(v[2]),
        "hallucination_w": float(v[3]),
        "refusal_penalty": float(refusal_penalty),
    }


def weights_for_profile(profile: str) -> dict[str, float]:
    return slo_vector_to_weights(get_slo_vector(profile))


if __name__ == "__main__":
    for name in ("quality_first", "cheap"):
        vec = get_slo_vector(name)
        weights = slo_vector_to_weights(vec)
        print(f"{name} vec: {vec.tolist()}")
        print(f"{name} weights: {weights}")
