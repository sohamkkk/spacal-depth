"""Dataset factory (imports lazily so h5py/PIL are only needed for the dataset used)."""

DATASETS = ("nyu", "kitti", "diode_indoor", "diode_outdoor", "eth3d")


def build_dataset(name: str):
    if name == "nyu":
        from src.data.nyu import NYUDepthV2

        return NYUDepthV2()
    if name == "kitti":
        from src.data.kitti import KITTI

        return KITTI()
    if name in ("diode_indoor", "diode_outdoor"):
        from src.data.diode import DIODE

        return DIODE(name.split("_")[1])
    if name == "eth3d":
        from src.data.eth3d import ETH3D

        return ETH3D()
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
