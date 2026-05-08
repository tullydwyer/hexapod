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
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEP_DIR = ROOT / "STEP"
CHIPO_DIR = ROOT / "Chipo"
MESH_DIR = HERE / "meshes"


# Physical dimensions from chica/chipo config files.
COXA_LEN = 0.043
FEMUR_LEN = 0.060
TIBIA_LEN = 0.134


BODY_Z = -0.008
HIP_SERVO_VISUAL_DROP_MM = 10.0
COXA_VISUAL_RISE_MM = 33.0


# Servo shaft location in hexapod-v2/servo-MG996R.stl, millimetres.
SERVO_SHAFT = np.array([10.08, -3.18, 39.0])

# MG996R mounting tab hole centers recovered from horizontal circular rings in
# the servo STL, in the STL's native XY frame. Subtracting SERVO_SHAFT gives
# the screw-hole rectangle in shaft-local coordinates.
SERVO_MOUNT_HOLES_NATIVE_XY_MM = np.array(
    [
        [-24.110, 1.825],
        [-24.110, -8.168],
        [24.110, 1.825],
        [24.110, -8.168],
    ]
)
SERVO_MOUNT_HOLES_LOCAL_XY_MM = SERVO_MOUNT_HOLES_NATIVE_XY_MM - SERVO_SHAFT[:2]


def servo_mount_origin_2d(
    frame_holes_mm: np.ndarray | list[tuple[float, float]],
    servo_holes_mm: np.ndarray | list[tuple[float, float]],
) -> np.ndarray:
    frame_holes = np.array(frame_holes_mm, dtype=float)
    servo_holes = np.array(servo_holes_mm, dtype=float)

    frame_mid = (frame_holes.min(axis=0) + frame_holes.max(axis=0)) / 2.0
    servo_min = servo_holes.min(axis=0)
    servo_max = servo_holes.max(axis=0)

    servo_xy = np.column_stack(
        [
            np.where(frame_holes[:, 0] < frame_mid[0], servo_min[0], servo_max[0]),
            np.where(frame_holes[:, 1] < frame_mid[1], servo_min[1], servo_max[1]),
        ]
    )
    origins = frame_holes - servo_xy
    return origins.mean(axis=0)


def frame_yaw(
    outer_xy_mm: tuple[float, float],
    inner_xy_mm: tuple[float, float],
) -> float:
    outer = np.array(outer_xy_mm, dtype=float)
    inner = np.array(inner_xy_mm, dtype=float)
    return math.atan2(*(outer - inner)[::-1])


def servo_mount_leg(
    name: str,
    outer_xy_mm: tuple[float, float],
    inner_xy_mm: tuple[float, float],
    frame_holes_mm: list[tuple[float, float]],
    side: str,
) -> tuple[str, tuple[float, float, float], str, float]:
    yaw = frame_yaw(outer_xy_mm, inner_xy_mm)
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    normal = np.array([-math.sin(yaw), math.cos(yaw)])

    holes = np.array(frame_holes_mm, dtype=float)
    frame_local = np.column_stack([holes @ axis, holes @ normal])
    hip_local = servo_mount_origin_2d(frame_local, SERVO_MOUNT_HOLES_LOCAL_XY_MM)
    hip = hip_local[0] * axis + hip_local[1] * normal
    return name, (hip[0] * 0.001, hip[1] * 0.001, BODY_Z), side, yaw


# Hip origins are solved by matching the MG996R tab-hole rectangle to each
# frame STEP screw-hole rectangle. Yaw still comes from the frame arm centerline
# between the inner r=2mm feature and the outer station feature.
LEGS = [
    servo_mount_leg(
        "l1",
        (-76.600, 84.571),
        (-49.023, 56.994),
        [(-76.246, 91.289), (-69.175, 98.360), (-42.305, 57.348), (-35.234, 64.419)],
        "left",
    ),
    servo_mount_leg(
        "l2",
        (-91.000, -8.029),
        (-52.000, -8.029),
        [(-95.500, -3.029), (-95.500, 6.971), (-47.500, -3.029), (-47.500, 6.971)],
        "left",
    ),
    servo_mount_leg(
        "l3",
        (-62.547, -94.718),
        (-34.970, -67.140),
        [(-76.224, -87.323), (-69.153, -94.395), (-42.282, -53.382), (-35.211, -60.453)],
        "left",
    ),
    servo_mount_leg(
        "r1",
        (76.600, 84.571),
        (49.023, 56.994),
        [(35.234, 64.419), (42.305, 57.348), (69.175, 98.360), (76.246, 91.289)],
        "right",
    ),
    servo_mount_leg(
        "r2",
        (91.000, -8.029),
        (52.000, -8.029),
        [(47.500, -3.029), (47.500, 6.971), (95.500, -3.029), (95.500, 6.971)],
        "right",
    ),
    servo_mount_leg(
        "r3",
        (62.547, -94.718),
        (34.970, -67.140),
        [(35.211, -60.453), (42.282, -53.382), (69.153, -94.395), (76.224, -87.323)],
        "right",
    ),
]


# Feature centers recovered from circular horn geometry in the STEP meshes.
# These are millimetres in the source CAD coordinate frames.
COXA_HIP_AXIS = np.array([3.0, 1.0, 35.0])
FEMUR_PROX_AXIS = np.array([0.0, 0.0, 35.0])
TIBIA_PROX_AXIS = np.array([51.757, 0.0, 8.787])

COXA_SOURCE_FILES = {
    "left": HERE / "coxa-996-left.stl",
    "right": HERE / "coxa-996-right.stl",
}

# The right coxa STL is exported in the assembly frame; rotate and translate it
# into the mirrored local coxa frame before applying the hip-axis alignment.
COXA_RIGHT_SOURCE_ROTATION = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ]
)
COXA_RIGHT_SOURCE_OFFSET_MM = np.array([175.358, 7.0, 131.285])

# MG996R shoulder tab-hole centers in coxa.step, in source CAD XZ.
COXA_SHOULDER_MOUNT_HOLES_STEP_XZ_MM = np.array(
    [
        [41.0, -11.5],
        [51.0, -11.5],
        [41.0, 36.5],
        [51.0, 36.5],
    ]
)

# MG996R knee tab-hole centers in tibia.step, in source CAD XZ.  The duplicate
# y=1.5/y=9.5 rings share these XZ centers.
TIBIA_KNEE_MOUNT_HOLES_STEP_XZ_MM = np.array(
    [
        [28.515, -29.801],
        [36.176, -36.229],
        [59.369, 6.969],
        [67.030, 0.541],
    ]
)


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


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
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


# Limb servos: native shaft +Z becomes joint +Y/-Y, and native body length X
# becomes vertical Z. Keeping both sides' native +X downward aligns the MG996R
# tab holes to the printed coxa shoulder pocket.
LIMB_SERVO_ROTATIONS = {
    "right": np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    ),
    "left": np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ]
    ),
}

# Knee servos need the limb mounting-hole frame rotated end-for-end so the
# output spline sits on the opposite end of the case while the same attachment
# side stays against the tibia mount. tibia_mount_pivot() then keeps the tab
# holes aligned to the printed mount.
KNEE_SERVO_ROTATIONS = {
    side: rot_y(math.pi) @ rot_z(math.pi) @ rotation
    for side, rotation in LIMB_SERVO_ROTATIONS.items()
}
SHOULDER_SERVO_ROTATIONS = {
    side: rot_z(math.pi) @ rotation
    for side, rotation in LIMB_SERVO_ROTATIONS.items()
}
SHOULDER_SERVO_SLIDE_MM = {
    "left": np.array([0.0, 20.314, 0.0]),
    "right": np.array([0.0, -20.314, 0.0]),
}
KNEE_SERVO_SLIDE_MM = {
    "left": np.array([0.0, 29.0, 0.0]),
    "right": np.array([0.0, -29.0, 0.0]),
}
TIBIA_SLIDE_MM = {
    "left": np.array([0.0, 0.0, 0.0]),
    "right": np.array([0.0, 0.0, 0.0]),
}
FEMUR_SHOULDER_INSET_MM = {
    "left": np.array([0.0, -9.0, 0.0]),
    "right": np.array([0.0, 9.0, 0.0]),
}


def tibia_to_servo_base_rotation() -> np.ndarray:
    long_axis = TIBIA_KNEE_MOUNT_HOLES_STEP_XZ_MM[2] - TIBIA_KNEE_MOUNT_HOLES_STEP_XZ_MM[0]
    return rot_y(math.atan2(-long_axis[0], long_axis[1]))


TIBIA_TO_SERVO_BASE_ROTATION = tibia_to_servo_base_rotation()
TIBIA_TO_SERVO_ROTATIONS = {
    "right": rot_z(math.pi) @ TIBIA_TO_SERVO_BASE_ROTATION,
    "left": rot_z(math.pi) @ TIBIA_TO_SERVO_BASE_ROTATION @ np.diag([1.0, -1.0, 1.0]),
}


def points_xz_to_xyz(points_xz: np.ndarray) -> np.ndarray:
    return np.column_stack([points_xz[:, 0], np.zeros(len(points_xz)), points_xz[:, 1]])


def rotated_servo_mount_holes_xz(rotation: np.ndarray) -> np.ndarray:
    mount_holes = np.column_stack(
        [SERVO_MOUNT_HOLES_LOCAL_XY_MM, np.zeros(len(SERVO_MOUNT_HOLES_LOCAL_XY_MM))]
    )
    return (rotation @ mount_holes.T).T[:, [0, 2]]


def mount_aligned_pivot(
    step_holes_xz: np.ndarray,
    part_rotation: np.ndarray,
    servo_rotation: np.ndarray,
    initial_pivot: np.ndarray,
) -> np.ndarray:
    step_holes = points_xz_to_xyz(step_holes_xz)
    rotated_holes_xz = (part_rotation @ (step_holes - initial_pivot).T).T[:, [0, 2]]
    mount_offset_xz = servo_mount_origin_2d(
        rotated_holes_xz,
        rotated_servo_mount_holes_xz(servo_rotation),
    )
    return initial_pivot + part_rotation.T @ np.array([mount_offset_xz[0], 0.0, mount_offset_xz[1]])


def shoulder_mount_origin(side: str) -> tuple[float, float, float]:
    coxa_holes = COXA_SHOULDER_MOUNT_HOLES_STEP_XZ_MM - COXA_HIP_AXIS[[0, 2]]
    coxa_holes[:, 1] += COXA_VISUAL_RISE_MM
    origin_xz = servo_mount_origin_2d(
        coxa_holes,
        rotated_servo_mount_holes_xz(SHOULDER_SERVO_ROTATIONS[side]),
    )
    return (origin_xz[0] * 0.001, 0.0, origin_xz[1] * 0.001)


def femur_shoulder_origin(side: str) -> tuple[float, float, float]:
    origin = np.array(shoulder_mount_origin(side)) + FEMUR_SHOULDER_INSET_MM[side] * 0.001
    return tuple(origin.tolist())


def tibia_mount_pivot(side: str) -> np.ndarray:
    return mount_aligned_pivot(
        TIBIA_KNEE_MOUNT_HOLES_STEP_XZ_MM,
        TIBIA_TO_SERVO_ROTATIONS[side],
        KNEE_SERVO_ROTATIONS[side],
        TIBIA_PROX_AXIS,
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


def feature_aligned_mesh(
    source: Path,
    target: str,
    rotation: np.ndarray,
    pivot: np.ndarray,
    extra_offset: np.ndarray | None = None,
) -> None:
    mesh = load_mesh(source)
    vertices = (rotation @ (np.asarray(mesh.vertices) - pivot).T).T
    if extra_offset is not None:
        vertices = vertices + extra_offset
    mesh.vertices = vertices
    if np.linalg.det(rotation) < 0:
        mesh.invert()
    write_mesh(target, mesh)


def coxa_source_mesh(side: str) -> trimesh.Trimesh:
    if side == "right":
        return transformed_mesh(
            COXA_SOURCE_FILES[side],
            rotation=COXA_RIGHT_SOURCE_ROTATION,
            offset=COXA_RIGHT_SOURCE_OFFSET_MM,
        )
    return transformed_mesh(COXA_SOURCE_FILES[side])


def write_aligned_mesh(
    target: str,
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    pivot: np.ndarray,
    extra_offset: np.ndarray | None = None,
) -> None:
    vertices = (rotation @ (np.asarray(mesh.vertices) - pivot).T).T
    if extra_offset is not None:
        vertices = vertices + extra_offset
    mesh.vertices = vertices
    if np.linalg.det(rotation) < 0:
        mesh.invert()
    write_mesh(target, mesh)


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

    coxa_visual_offset = np.array([0.0, 0.0, COXA_VISUAL_RISE_MM])
    write_aligned_mesh(
        "coxa-left.stl",
        coxa_source_mesh("left"),
        np.eye(3),
        COXA_HIP_AXIS,
        extra_offset=coxa_visual_offset,
    )
    write_aligned_mesh(
        "coxa-right.stl",
        coxa_source_mesh("right"),
        np.eye(3),
        np.array([COXA_HIP_AXIS[0], -COXA_HIP_AXIS[1], COXA_HIP_AXIS[2]]),
        extra_offset=coxa_visual_offset,
    )

    femur_to_link = rot_x(math.pi) @ rot_x(-math.pi / 2)
    feature_aligned_mesh(CHIPO_DIR / "femur-996.stl", "femur-left.stl", femur_to_link, FEMUR_PROX_AXIS)
    feature_aligned_mesh(
        CHIPO_DIR / "femur-996.stl",
        "femur-right.stl",
        femur_to_link @ np.diag([1.0, 1.0, -1.0]),
        FEMUR_PROX_AXIS,
    )

    # Align the tibia knee-end MG996R tab holes to the knee servo while keeping
    # the foot below the body in link-local Z.
    feature_aligned_mesh(
        STEP_DIR / "tibia.step",
        "tibia-right.stl",
        TIBIA_TO_SERVO_ROTATIONS["right"],
        tibia_mount_pivot("right"),
        extra_offset=TIBIA_SLIDE_MM["right"],
    )
    feature_aligned_mesh(
        STEP_DIR / "tibia.step",
        "tibia-left.stl",
        TIBIA_TO_SERVO_ROTATIONS["left"],
        tibia_mount_pivot("left"),
        extra_offset=TIBIA_SLIDE_MM["left"],
    )

    # Hip servos are yaw servos: shaft exits downward through the frame pocket.
    # Rx(π) flips the body upward so it sits inside the frame, shaft pointing down.
    servo_mesh(
        "servo-hip-shaft.stl",
        rot_x(math.pi),
        extra_offset=np.array([0.0, 0.0, -HIP_SERVO_VISUAL_DROP_MM]),
    )

    servo_mesh("servo-limb-right.stl", LIMB_SERVO_ROTATIONS["right"])
    servo_mesh("servo-limb-left.stl", LIMB_SERVO_ROTATIONS["left"])
    servo_mesh(
        "servo-shoulder-z180-slid-right.stl",
        SHOULDER_SERVO_ROTATIONS["right"],
        extra_offset=SHOULDER_SERVO_SLIDE_MM["right"],
    )
    servo_mesh(
        "servo-shoulder-z180-slid-left.stl",
        SHOULDER_SERVO_ROTATIONS["left"],
        extra_offset=SHOULDER_SERVO_SLIDE_MM["left"],
    )
    servo_mesh(
        "servo-knee-right.stl",
        KNEE_SERVO_ROTATIONS["right"],
        extra_offset=KNEE_SERVO_SLIDE_MM["right"],
    )
    servo_mesh(
        "servo-knee-left.stl",
        KNEE_SERVO_ROTATIONS["left"],
        extra_offset=KNEE_SERVO_SLIDE_MM["left"],
    )


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
        shoulder_servo = f"servo-shoulder-z180-slid-{side}.stl"
        knee_servo = f"servo-knee-{side}.stl"
        shoulder_servo_xyz = shoulder_mount_origin(side)
        shoulder_xyz = femur_shoulder_origin(side)
        knee_xyz = (FEMUR_LEN, 0.0, 0.0)
        knee_servo_xyz = (0.0, 0.0, 0.0)

        add_link(lines, f"{leg_name}_hip_servo", 0.055, (0.0542, 0.020, 0.0465), [mesh_block("servo-hip-shaft.stl")])
        add_link(lines, f"{leg_name}_coxa", 0.040, (COXA_LEN, 0.030, 0.060), [mesh_block(coxa_mesh)])
        add_link(lines, f"{leg_name}_shoulder_servo", 0.055, (0.0542, 0.020, 0.0465), [mesh_block(shoulder_servo)])
        add_link(lines, f"{leg_name}_femur", 0.060, (FEMUR_LEN, 0.020, 0.060), [mesh_block(femur_mesh)])
        add_link(lines, f"{leg_name}_knee_servo", 0.055, (0.0542, 0.020, 0.0465), [mesh_block(knee_servo)])
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
            f'    <origin xyz="{fmt_xyz(shoulder_servo_xyz)}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_shoulder" type="revolute">',
            f'    <parent link="{leg_name}_coxa"/>',
            f'    <child link="{leg_name}_femur"/>',
            f'    <origin xyz="{fmt_xyz(shoulder_xyz)}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            '    <axis xyz="0 1 0"/>',
            '    <limit lower="-1.5708" upper="1.5708" effort="2.0" velocity="6.28"/>',
            '    <dynamics damping="0.01" friction="0.05"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_knee_servo_joint" type="fixed">',
            f'    <parent link="{leg_name}_tibia"/>',
            f'    <child link="{leg_name}_knee_servo"/>',
            f'    <origin xyz="{fmt_xyz(knee_servo_xyz)}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
            "  </joint>",
            "",
            f'  <joint name="{leg_name}_knee" type="revolute">',
            f'    <parent link="{leg_name}_femur"/>',
            f'    <child link="{leg_name}_tibia"/>',
            f'    <origin xyz="{fmt_xyz(knee_xyz)}" rpy="{fmt_rpy(0.0, 0.0, 0.0)}"/>',
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
