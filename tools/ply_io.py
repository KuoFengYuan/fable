"""極簡 PLY 讀寫：支援 ascii / binary_little_endian、x y z (+ red green blue)。"""

from __future__ import annotations

import numpy as np

_TYPE_MAP = {
    "float": "<f4", "float32": "<f4",
    "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1",
    "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2",
    "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4",
    "int": "<i4", "int32": "<i4",
}


def read_ply(path):
    """回傳 (xyz float64 (N,3), rgb uint8 (N,3) 或 None)。只解析 vertex 元素。"""
    with open(path, "rb") as f:
        line = f.readline().strip()
        if line != b"ply":
            raise ValueError(f"{path}: 不是 PLY 檔")
        fmt = None
        n_vertex = 0
        props: list[tuple[str, str]] = []
        current_element = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header 未結束")
            tokens = line.decode("ascii", "replace").strip().split()
            if not tokens:
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                current_element = tokens[1]
                if current_element == "vertex":
                    n_vertex = int(tokens[2])
            elif tokens[0] == "property" and current_element == "vertex":
                if tokens[1] == "list":
                    raise ValueError("vertex 元素不支援 list property")
                props.append((tokens[-1], _TYPE_MAP[tokens[1]]))
            elif tokens[0] == "end_header":
                break

        names = [name for name, _ in props]
        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, t) for name, t in props])
            data = np.fromfile(f, dtype=dtype, count=n_vertex)
        elif fmt == "ascii":
            rows = [f.readline().split() for _ in range(n_vertex)]
            arr = np.array(rows, dtype=np.float64)
            data = {name: arr[:, i] for i, name in enumerate(names)}
        else:
            raise ValueError(f"不支援的 PLY 格式: {fmt}")

        xyz = np.stack([np.asarray(data["x"], np.float64),
                        np.asarray(data["y"], np.float64),
                        np.asarray(data["z"], np.float64)], axis=1)
        rgb = None
        for r, g, b in (("red", "green", "blue"), ("r", "g", "b")):
            if r in names and g in names and b in names:
                rgb = np.stack([np.asarray(data[r]), np.asarray(data[g]),
                                np.asarray(data[b])], axis=1)
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                break
        return xyz, rgb


def write_ply(path, xyz: np.ndarray, rgb: np.ndarray | None = None):
    """寫出 binary little-endian PLY（float32 xyz + uint8 rgb）。"""
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    rec = np.empty(n, dtype=np.dtype(fields))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        rec.tofile(f)
