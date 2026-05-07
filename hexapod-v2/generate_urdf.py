"""
Hexapod-v2 URDF generator.

This version treats the STEP files as the source of truth for printed parts.
Those CAD files already carry useful pivot origins, so the generated STL meshes
preserve those coordinates instead of re-origining parts to their bounding-box
minimums.  The only baked transforms are:

* body pieces are centred around base_link for easier inspection;
* mirrored left/right variants are generated where the CAD set only has one;
* servo visuals are pre-rotated so each servo shaft sits exactly on its joint.

Coordinate convention:
  x = robot right
  y = robot forward
  z = up
"""

from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEP_DIR = ROOT / "STEP"
MESH_DIR = HERE / "meshes"


# Physical dimensions from chica-config-2040.txt.
COXA_LEN = 0.043
FEMUR_LEN = 0.080
TIBIA_LEN = 0.134


# Body mount locations from chica-config-2040.txt.
BODY_FRONT_Y = 0.0835
BODY_CORNER_X = 0.063
BODY_MID_X = 0.0815
BODY_Z = -0.010


# name, mount xyz, side, yaw angle from +X toward the outward leg direction
LEGS = [
    ("l1", (-BODY_CORNER_X, BODY_FRONT_Y, BODY_Z), "left", math.atan2(BODY_FRONT_Y, -BODY_CORNER_X)),
    ("l2", (-BODY_MID_X, 0.0, BODY_Z), "left", math.pi),
    ("l3", (-BODY_CORNER_X, -BODY_FRONT_Y, BODY_Z), "left", math.atan2(-BODY_FRONT_Y, -BODY_CORNER_X)),
    ("r1", (BODY_CORNER_X, BODY_FRONT_Y, BODY_Z), "right", math.atan2(BODY_FRONT_Y, BODY_CORNER_X)),
    ("r2", (BODY_MID_X, 0.0, BODY_Z), "right", 0.0),
    ("r3", (BODY_CORNER_X, -BODY_FRONT_Y, BODY_Z), "right", math.atan2(-BODY_FRONT_Y, BODY_CORNER_X)),
]


# Servo shaft location in hexapod-v2/servo-MG996R.stl, millimetres.
SERVO_SHAFT = np.array([10.08, -3.18, 39.0])
HIP_SERVO_LOWER_MM = -20.0


def rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def mirror_z() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    try:
        mesh = trimesh.load(path, force="scene")
    except ModuleNotFoundError as exc:
        if path.suffix.lower() in {".step", ".stp"} and exc.name == "cascadio":
            raise RuntimeError(
                "STEP loading requires cascadio. Run "
                "`.venv\\Scripts\\python.exe -m pip install cascadio` first."
            ) from exc
        raise
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(mesh)!r}")

    # cascadio loads STEP in metres; STL files in this repo are millimetres.
    extents = mesh.bounds[1] - mesh.bounds[0]
    if path.suffix.lower() in {".step", ".stp"} or float(extents.max()) < 1.0:
        mesh.vertices = np.asarray(mesh.vertices) * 1000.0
    return mesh


def bounds(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    centre = (lo + hi) / 2.0
    return lo, hi, centre, hi - lo


def write_mesh(target: str, mesh: trimesh.Trimesh) -> None:
    mesh.export(MESH_DIR / target)


def transformed_mesh(source: Path, rotation: np.ndarray | None = None, offset: np.ndarray | None = None) -> trimesh.Trimesh:
    mesh = load_mesh(source)
    vertices = np.asarray(mesh.vertices)
    if rotation is not None:
        vertices = (rotation @ vertices.T).T
        if np.linalg.det(rotation) < 0:
            mesh.invert()
    if offset is not None:
        vertices = vertices + offset
    mesh.vertices = vertices
    return mesh


def centred_mesh(source: Path, target: str, rotation: np.ndarray | None = None) -> None:
    mesh = transformed_mesh(source, rotation=rotation)
    _, _, centre, _ = bounds(mesh)
    mesh.vertices = np.asarray(mesh.vertices) - centre
    write_mesh(target, mesh)


def offset_mesh(source: Path, target: str, offset: np.ndarray, rotation: np.ndarray | None = None) -> None:
    write_mesh(target, transformed_mesh(source, rotation=rotation, offset=offset))


def mirrored_mesh(source: Path, target: str, axis: int) -> None:
    mirror = np.eye(3)
    mirror[axis, axis] = -1.0
    write_mesh(target, transformed_mesh(source, rotation=mirror))


def servo_mesh(target: str, rotation: np.ndarray, extra_offset: np.ndarray | None = None) -> None:
    shaft = rotation @ SERVO_SHAFT
    offset = -shaft
    if extra_offset is not None:
        offset = offset + extra_offset
    write_mesh(target, transformed_mesh(HERE / "servo-MG996R.stl", rotation=rotation, offset=offset))


def tibia_tip_for(mesh_name: str) -> tuple[float, float, float]:
    """Return the lowest-foot contact point in metres for a generated tibia."""
    mesh = load_mesh(MESH_DIR / mesh_name)
    vertices = np.asarray(mesh.vertices)
    low = vertices[:, 2].min()
    foot = vertices[vertices[:, 2] < low + 1.0].mean(axis=0)
    return tuple((foot * 0.001).tolist())


def build_meshes() -> None:
    MESH_DIR.mkdir(exist_ok=True)

    centred_mesh(STEP_DIR / "frame.step", "frame-holes-down.stl", rotation=mirror_z())
    centred_mesh(STEP_DIR / "top-cover.step", "top-cover.stl")

    top = load_mesh(STEP_DIR / "top-cover.step")
    _, _, top_centre, _ = bounds(top)
    offset_mesh(STEP_DIR / "bottom-cover-flat.step", "bottom-cover-flat.stl", offset=-top_centre)

    write_mesh("coxa-left.stl", load_mesh(STEP_DIR / "coxa.step"))
    mirrored_mesh(STEP_DIR / "coxa.step", "coxa-right.stl", axis=1)

    write_mesh("femur-right.stl", load_mesh(STEP_DIR / "femur.step"))
    # The repository's left femur STL is the right femur mirrored through Z.
    mirrored_mesh(STEP_DIR / "femur.step", "femur-left.stl", axis=2)

    write_mesh("tibia-right.stl", load_mesh(STEP_DIR / "tibia.step"))
    mirrored_mesh(STEP_DIR / "tibia.step", "tibia-left.stl", axis=1)

    # Hip servos are yaw servos: their shafts must stay vertical.  This is the
    # previous outboard pose rotated 180 degrees around the shaft and lowered.
    servo_mesh(
        "servo-hip-180-lower.stl",
        rot_z(math.pi) @ rot_z(math.pi) @ rot_x(math.pi),
        extra_offset=np.array([0.0, 0.0, HIP_SERVO_LOWER_MM]),
    )

    # Limb servos: native shaft +Z becomes joint +Y/-Y, and native body length X
    # becomes vertical Z.  This matches the tall printed coxa/femur servo pockets.
    limb_right = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    limb_left = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    servo_mesh("servo-limb-right.stl", limb_right)
    servo_mesh("servo-limb-left.stl", limb_left)


def fmt_xyz(values: tuple[float, float, float] | np.ndarray) -> str:
    return f"{values[0]:.6f} {values[1]:.6f} {values[2]:.6f}"


def fmt_rpy(roll: float, pitch: float, yaw: float) -> str:
    return f"{roll:.6f} {pitch:.6f} {yaw:.6f}"


def mesh_block(filename: str, scale: str = "0.001 0.001 0.001") -> str:
    return f"""\
      <visual>
        <geometry>
          <mesh filename="./meshes/{filename}" scale="{scale}"/>
        </geometry>
        <material name="grey"/>
      </visual>
      <collision>
        <geometry>
          <mesh filename="./meshes/{filename}" scale="{scale}"/>
        </geometry>
      </collision>"""


def inertial_box(mass: float, x: float, y: float, z: float) -> str:
    ixx = mass / 12.0 * (y**2 + z**2)
    iyy = mass / 12.0 * (x**2 + z**2)
    izz = mass / 12.0 * (x**2 + y**2)
    return f"""\
      <inertial>
        <mass value="{mass}"/>
        <inertia ixx="{ixx:.8f}" ixy="0" ixz="0"
                 iyy="{iyy:.8f}" iyz="0"
                 izz="{izz:.8f}"/>
      </inertial>"""


def add_link(lines: list[str], name: str, mass: float, size: tuple[float, float, float], visuals: list[str]) -> None:
    lines.append(f'  <link name="{name}">')
    lines.append(inertial_box(mass, *size))
    lines.extend(visuals)
    lines.append("  </link>")
    lines.append("")


def generate_urdf() -> Path:
    build_meshes()

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<robot name="hexapod_v2">',
        "",
        '  <material name="grey"><color rgba="0.7 0.7 0.7 1.0"/></material>',
        '  <material name="dark"><color rgba="0.2 0.2 0.2 1.0"/></material>',
        "",
    ]

    add_link(
        lines,
        "base_link",
        0.600,
        (0.170, 0.165, 0.040),
        [
            mesh_block("frame-holes-down.stl"),
            mesh_block("top-cover.stl"),
            mesh_block("bottom-cover-flat.stl"),
        ],
    )

    tibia_tip = {
        "left": tibia_tip_for("tibia-left.stl"),
        "right": tibia_tip_for("tibia-right.stl"),
    }

    for leg_name, mount_xyz, side, yaw in LEGS:
        coxa_mesh = f"coxa-{side}.stl"
        femur_mesh = f"femur-{side}.stl"
        tibia_mesh = f"tibia-{side}.stl"
        limb_servo = f"servo-limb-{side}.stl"

        add_link(lines, f"{leg_name}_hip_servo", 0.055, (0.0542, 0.020, 0.0465), [mesh_block("servo-hip-180-lower.stl")])
        add_link(lines, f"{leg_name}_coxa", 0.040, (COXA_LEN, 0.030, 0.060), [mesh_block(coxa_mesh)])
        add_link(lines, f"{leg_name}_shoulder_servo", 0.055, (0.0542, 0.020, 0.0465), [mesh_block(limb_servo)])
        add_link(lines, f"{leg_name}_femur", 0.060, (FEMUR_LEN, 0.020, 0.060), [mesh_block(femur_mesh)])
        add_link(lines, f"{leg_name}_knee_servo", 0.055, (0.0542, 0.020, 0.0465), [mesh_block(limb_servo)])
        add_link(lines, f"{leg_name}_tibia", 0.040, (TIBIA_LEN, 0.015, 0.015), [mesh_block(tibia_mesh)])

        lines += [
            f'  <link name="{leg_name}_tip">',
            '      <inertial>',
            '        <mass value="0.001"/>',
            '        <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>',
            '      </inertial>',
            '      <visual>',
            '        <geometry><sphere radius="0.008"/></geometry>',
            '        <material name="dark"/>',
            '      </visual>',
            '      <collision>',
            '        <geometry><sphere radius="0.008"/></geometry>',
            '      </collision>',
            "  </link>",
            "",
            f'  <joint name="{leg_name}_hip_servo_joint" type="fixed">',
            '    <parent link="base_link"/>',
            f'    <child link="{leg_name}_hip_servo"/>',
            f'    <origin xyz="{fmt_xyz(mount_xyz)}" rpy="{fmt_rpy(0.0, 0.0, yaw)}"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_hip" type="revolute">',
            '    <parent link="base_link"/>',
            f'    <child link="{leg_name}_coxa"/>',
            f'    <origin xyz="{fmt_xyz(mount_xyz)}" rpy="{fmt_rpy(0.0, 0.0, yaw)}"/>',
            '    <axis xyz="0 0 1"/>',
            '    <limit lower="-1.5708" upper="1.5708" effort="2.0" velocity="6.28"/>',
            '    <dynamics damping="0.01" friction="0.05"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_shoulder_servo_joint" type="fixed">',
            f'    <parent link="{leg_name}_coxa"/>',
            f'    <child link="{leg_name}_shoulder_servo"/>',
            f'    <origin xyz="{fmt_xyz((COXA_LEN, 0.0, 0.0))}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_shoulder" type="revolute">',
            f'    <parent link="{leg_name}_coxa"/>',
            f'    <child link="{leg_name}_femur"/>',
            f'    <origin xyz="{fmt_xyz((COXA_LEN, 0.0, 0.0))}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            '    <axis xyz="0 1 0"/>',
            '    <limit lower="-1.5708" upper="1.5708" effort="2.0" velocity="6.28"/>',
            '    <dynamics damping="0.01" friction="0.05"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_knee_servo_joint" type="fixed">',
            f'    <parent link="{leg_name}_femur"/>',
            f'    <child link="{leg_name}_knee_servo"/>',
            f'    <origin xyz="{fmt_xyz((FEMUR_LEN, 0.0, 0.0))}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_knee" type="revolute">',
            f'    <parent link="{leg_name}_femur"/>',
            f'    <child link="{leg_name}_tibia"/>',
            f'    <origin xyz="{fmt_xyz((FEMUR_LEN, 0.0, 0.0))}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            '    <axis xyz="0 1 0"/>',
            '    <limit lower="-2.0944" upper="0.5236" effort="2.0" velocity="6.28"/>',
            '    <dynamics damping="0.01" friction="0.05"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_tip_joint" type="fixed">',
            f'    <parent link="{leg_name}_tibia"/>',
            f'    <child link="{leg_name}_tip"/>',
            f'    <origin xyz="{fmt_xyz(tibia_tip[side])}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            "  </joint>",
            "",
        ]

    lines += ["</robot>", ""]

    out_path = HERE / "hexapod-v2.urdf"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    ET.parse(out_path)
    return out_path


def analyse() -> None:
    print("\nGenerated mesh bounding boxes (metres after URDF scale):")
    for path in sorted(MESH_DIR.glob("*.stl")):
        mesh = load_mesh(path)
        lo, hi, centre, extents = bounds(mesh)
        print(f"  {path.name}")
        print(f"    min : {np.round(lo * 0.001, 4)}")
        print(f"    max : {np.round(hi * 0.001, 4)}")
        print(f"    size: {np.round(extents * 0.001, 4)}")
        print(f"    ctr : {np.round(centre * 0.001, 4)}")


if __name__ == "__main__":
    written = generate_urdf()
    analyse()
    print(f"\nWrote: {written}")
