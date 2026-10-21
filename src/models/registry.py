"""Model factory. Tags: dav2-{indoor,outdoor}-{small,base,large}, unidepthv2-{small,large}, depthpro."""


def build_model(tag: str, device: str | None = None):
    parts = tag.split("-")
    if parts[0] == "dav2" and len(parts) == 3:
        from src.models.depth_anything import DepthAnythingV2Metric

        return DepthAnythingV2Metric(domain=parts[1], size=parts[2], device=device)
    if parts[0] == "unidepthv2" and len(parts) == 2:
        from src.models.unidepth import UniDepthV2Model

        return UniDepthV2Model(size=parts[1], device=device)
    if tag == "depthpro":
        from src.models.depth_pro import DepthProModel

        return DepthProModel(device=device)
    raise ValueError(f"unknown model tag {tag!r}")
