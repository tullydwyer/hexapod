"""
Hexapod-v2 URDF Generator
Analyzes STL meshes to determine bounding boxes / pivot offsets, then writes hexapod-v2.urdf.

Coordinate convention (ROS/URDF standard, matching CAD frame notes):
  x = robot right
  y = robot forward
  z = up

Physical parameters sourced from chica-config-2040.txt:
  COXA_LEN  =  43 mm
  FEMUR_LEN =  80 mm
  TIBIA_LEN = 134 mm
  L1_TO_R1  = 126 mm  → lateral offset of front/rear mounts = ±63 mm
  L2_TO_R2  = 163 mm  → lateral offset of middle mounts   = ±81.5 mm
  L1_TO_L3  = 167 mm  → longitudinal offset of front/rear  = ±83.5 mm
"""

import math
import os
import trimesh
import numpy as np

# ── physical constants (metres) ──────────────────────────────────────────────
COXA_LEN   = 0.043
FEMUR_LEN  = 0.080
TIBIA_LEN  = 0.134

# Body frame half-dimensions
BODY_FRONT_Y  =  0.0835   # front/rear mount offset along Y
BODY_CORNER_X =  0.063    # front/rear mount offset along X (abs)
BODY_MID_X    =  0.0815   # middle mount offset along X (abs)
BODY_Z        =  0.0      # coxa joints are at body mid-plane z=0

# Leg definitions  (name, mount_xyz, side, angle_deg_from_+X_axis)
# angle_from_X: angle in the XY plane from the +X axis to the outward direction
LEGS = [
    ("l1", (-BODY_CORNER_X,  BODY_FRONT_Y,  BODY_Z), "left",   math.degrees(math.atan2( BODY_FRONT_Y, -BODY_CORNER_X))),
    ("l2", (-BODY_MID_X,     0.0,           BODY_Z), "left",   180.0),
    ("l3", (-BODY_CORNER_X, -BODY_FRONT_Y,  BODY_Z), "left",   math.degrees(math.atan2(-BODY_FRONT_Y, -BODY_CORNER_X))),
    ("r1", ( BODY_CORNER_X,  BODY_FRONT_Y,  BODY_Z), "right",  math.degrees(math.atan2( BODY_FRONT_Y,  BODY_CORNER_X))),
    ("r2", ( BODY_MID_X,     0.0,           BODY_Z), "right",  0.0),
    ("r3", ( BODY_CORNER_X, -BODY_FRONT_Y,  BODY_Z), "right",  math.degrees(math.atan2(-BODY_FRONT_Y,  BODY_CORNER_X))),
]

HERE = os.path.dirname(os.path.abspath(__file__))


# ── mesh analysis helpers ─────────────────────────────────────────────────────

def load_mesh(name: str) -> trimesh.Trimesh | None:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"  [warn] mesh not found: {name}")
        return None
    try:
        mesh = trimesh.load_mesh(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        return mesh
    except Exception as e:
        print(f"  [warn] could not load {name}: {e}")
        return None


def bbox(mesh: trimesh.Trimesh):
    """Return (min_xyz, max_xyz, center_xyz, extents) in metres (meshes are in mm)."""
    v = mesh.vertices * 1e-3          # mm → m
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    return lo, hi, (lo + hi) / 2, hi - lo


# ── print mesh stats ──────────────────────────────────────────────────────────

def analyse():
    files = [
        "coxa-996-left.stl",
        "coxa-996-right.stl",
        "femur-996-left.stl",
        "femur-996-right.stl",
        "tibia-996-left.stl",
        "tibia-996-right.stl",
        "tip.stl",
        "top-cover2 (a).stl",
        "servo2040-bottom-cover (a).stl",
        "frame.stl",
        "servo-MG996R.stl",
    ]
    print("\n=== Mesh bounding boxes (metres) ===")
    for f in files:
        m = load_mesh(f)
        if m is None:
            continue
        lo, hi, ctr, ext = bbox(m)
        print(f"  {f}")
        print(f"    min : {lo.round(4)}")
        print(f"    max : {hi.round(4)}")
        print(f"    size: {ext.round(4)}")
        print(f"    ctr : {ctr.round(4)}")


# ── URDF helpers ──────────────────────────────────────────────────────────────

def fmt_xyz(v):
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"

def fmt_rpy(r, p, y):
    return f"{r:.6f} {p:.6f} {y:.6f}"

def mesh_tag(fname, scale="0.001 0.001 0.001", xyz="0 0 0", rpy="0 0 0"):
    """Return a <visual> + <collision> block referencing the given STL (in mm → m scale)."""
    return f"""\
      <visual>
        <origin xyz="{xyz}" rpy="{rpy}"/>
        <geometry>
          <mesh filename="package://hexapod_v2/{fname}" scale="{scale}"/>
        </geometry>
        <material name="grey">
          <color rgba="0.7 0.7 0.7 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="{xyz}" rpy="{rpy}"/>
        <geometry>
          <mesh filename="package://hexapod_v2/{fname}" scale="{scale}"/>
        </geometry>
      </collision>"""


# ── mesh pivot estimation ─────────────────────────────────────────────────────
# For each part, trimesh tells us the bounding box in the STL's own frame (mm).
# We need to find which point in that frame corresponds to the joint pivot.
#
# Coxa (left): the coxa servo sits at one end, the femur pivot is COXA_LEN away.
#   The STL is printed flat; its longest axis spans the coxa length.
#   Pivot A (hip attach):  at the "start" end of the coxa.
#   Pivot B (femur joint): at the "far" end, COXA_LEN = 43 mm along the longest axis.
#
# Femur (left): similarly, pivots at 0 and FEMUR_LEN = 80 mm along longest axis.
# Tibia (left): pivots at 0 and TIBIA_LEN = 134 mm along longest axis.
#
# We estimate the pivot by finding the two ends of the principal axis that align
# with the expected segment length, and choose the end closest to the bbox minimum
# on that axis as pivot A.

def estimate_coxa_mesh_offset(side: str):
    """
    Return (mesh_origin_xyz, mesh_origin_rpy) for the coxa link.
    The coxa joint origin is placed at the hip axis (output shaft of servo 1).
    The mesh needs to be translated so that hip-shaft end of the coxa aligns with (0,0,0)
    and the coxa extends along +X.

    For the LEFT coxa the mesh's principal axis should already point outward;
    for the RIGHT coxa it may be mirrored.
    """
    fname = f"coxa-996-{side}.stl"
    m = load_mesh(fname)
    if m is None:
        return "0 0 0", "0 0 0"
    lo, hi, ctr, ext = bbox(m)

    # Identify the principal (longest) axis index
    ax = int(np.argmax(ext))
    # The hip-attach end is at lo[ax]; the femur pivot end is at hi[ax].
    # We want the joint origin (hip pivot) to be at world (0,0,0) in the link frame.
    # So we offset the mesh so that lo[ax] on principal axis maps to 0.
    offset = np.zeros(3)
    offset[ax] = -lo[ax]  # shift so that hip-attach start is at 0
    # also centre transverse axes
    for i in range(3):
        if i != ax:
            offset[i] = -ctr[i]

    rpy = "0 0 0"
    # For right coxa, the mesh may be mirrored. We just use the mesh as-is and rely
    # on the joint yaw to orient it correctly.
    return fmt_xyz(offset), rpy


def estimate_femur_mesh_offset(side: str):
    fname = f"femur-996-{side}.stl"
    m = load_mesh(fname)
    if m is None:
        return "0 0 0", "0 0 0"
    lo, hi, ctr, ext = bbox(m)
    ax = int(np.argmax(ext))
    offset = np.zeros(3)
    offset[ax] = -lo[ax]
    for i in range(3):
        if i != ax:
            offset[i] = -ctr[i]
    return fmt_xyz(offset), "0 0 0"


def estimate_tibia_mesh_offset(side: str):
    fname = f"tibia-996-{side}.stl"
    m = load_mesh(fname)
    if m is None:
        return "0 0 0", "0 0 0"
    lo, hi, ctr, ext = bbox(m)
    ax = int(np.argmax(ext))
    offset = np.zeros(3)
    offset[ax] = -lo[ax]
    for i in range(3):
        if i != ax:
            offset[i] = -ctr[i]
    return fmt_xyz(offset), "0 0 0"


def estimate_body_mesh_offset():
    fname = "top-cover2 (a).stl"
    m = load_mesh(fname)
    if m is None:
        return "0 0 0", "0 0 0"
    lo, hi, ctr, ext = bbox(m)
    # Centre the body mesh on the origin
    offset = -ctr
    return fmt_xyz(offset), "0 0 0"


def estimate_tip_mesh_offset():
    m = load_mesh("tip.stl")
    if m is None:
        return "0 0 0", "0 0 0"
    lo, hi, ctr, ext = bbox(m)
    offset = -ctr
    return fmt_xyz(offset), "0 0 0"


# ── inertial placeholders ─────────────────────────────────────────────────────

def inertial_box(mass, x, y, z):
    ixx = mass / 12 * (y**2 + z**2)
    iyy = mass / 12 * (x**2 + z**2)
    izz = mass / 12 * (x**2 + y**2)
    return f"""\
      <inertial>
        <mass value="{mass}"/>
        <inertia ixx="{ixx:.8f}" ixy="0" ixz="0"
                 iyy="{iyy:.8f}" iyz="0"
                 izz="{izz:.8f}"/>
      </inertial>"""


# ── URDF generation ───────────────────────────────────────────────────────────

def generate_urdf():
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<robot name="hexapod_v2">',
             '',
             '  <!-- ═══════════════════════════════════════════════════════════',
             '       Hexapod-v2  URDF',
             '       Generated by generate_urdf.py',
             '       Coordinate frame: x=right  y=forward  z=up',
             '       All meshes in hexapod-v2/ directory (STLs in mm → scaled 0.001).',
             '  ════════════════════════════════════════════════════════════ -->',
             '',
             '  <!-- ─── Materials ────────────────────────────────────────── -->',
             '  <material name="grey">  <color rgba="0.7 0.7 0.7 1.0"/> </material>',
             '  <material name="dark">  <color rgba="0.2 0.2 0.2 1.0"/> </material>',
             '  <material name="white"> <color rgba="0.9 0.9 0.9 1.0"/> </material>',
             '']

    # ── base_link ─────────────────────────────────────────────────────────────
    body_xyz, body_rpy = estimate_body_mesh_offset()
    # frame mesh is already centred at origin; small Y/Z offset to match body
    frame_m = load_mesh("frame.stl")
    if frame_m is not None:
        _, _, frame_ctr, _ = bbox(frame_m)
        frame_xyz = fmt_xyz(-frame_ctr)
    else:
        frame_xyz = "0 0 0"

    lines += [
        '  <!-- ─── Body / base_link ─────────────────────────────────────── -->',
        '  <link name="base_link">',
        inertial_box(0.600, 0.170, 0.165, 0.040),
        mesh_tag("frame.stl",            xyz=frame_xyz, rpy=body_rpy),
        mesh_tag("top-cover2 (a).stl",   xyz=body_xyz,  rpy=body_rpy),
        mesh_tag("servo2040-bottom-cover (a).stl",
                 xyz=body_xyz, rpy=body_rpy),
        '  </link>',
        '',
    ]

    # ── six legs ──────────────────────────────────────────────────────────────
    for leg_name, (mx, my, mz), side, yaw_deg in LEGS:
        yaw_rad  = math.radians(yaw_deg)
        coxa_xyz, coxa_rpy     = estimate_coxa_mesh_offset(side)
        femur_xyz, femur_rpy   = estimate_femur_mesh_offset(side)
        tibia_xyz, tibia_rpy   = estimate_tibia_mesh_offset(side)
        tip_xyz,   tip_rpy     = estimate_tip_mesh_offset()

        coxa_stl  = f"coxa-996-{side}.stl"
        femur_stl = f"femur-996-{side}.stl"
        tibia_stl = f"tibia-996-{side}.stl"

        lines.append(f'  <!-- ─── Leg {leg_name.upper()} ──────────────────────────────────────── -->')

        # ── coxa link ──
        lines += [
            f'  <link name="{leg_name}_coxa">',
            inertial_box(0.040, COXA_LEN, 0.030, 0.025),
            mesh_tag(coxa_stl, xyz=coxa_xyz, rpy=coxa_rpy),
            '  </link>',
            '',
        ]

        # ── femur link ──
        lines += [
            f'  <link name="{leg_name}_femur">',
            inertial_box(0.060, FEMUR_LEN, 0.020, 0.025),
            mesh_tag(femur_stl, xyz=femur_xyz, rpy=femur_rpy),
            '  </link>',
            '',
        ]

        # ── tibia link ──
        lines += [
            f'  <link name="{leg_name}_tibia">',
            inertial_box(0.040, TIBIA_LEN, 0.015, 0.015),
            mesh_tag(tibia_stl, xyz=tibia_xyz, rpy=tibia_rpy),
            '  </link>',
            '',
        ]

        # ── tip link (contact sphere) ──
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
            '  </link>',
            '',
        ]

        # ── joint: base → coxa (hip yaw, Z axis) ──
        #   Joint origin is at the mount point on the body; yaw aligns coxa outward.
        lines += [
            f'  <joint name="{leg_name}_hip" type="revolute">',
            f'    <parent link="base_link"/>',
            f'    <child  link="{leg_name}_coxa"/>',
            f'    <origin xyz="{fmt_xyz((mx, my, mz))}" rpy="{fmt_rpy(0, 0, yaw_rad)}"/>',
            f'    <axis xyz="0 0 1"/>',
            f'    <limit lower="-1.5708" upper="1.5708" effort="2.0" velocity="6.28"/>',
            f'    <dynamics damping="0.01" friction="0.05"/>',
            f'  </joint>',
            '',
        ]

        # ── joint: coxa → femur (shoulder pitch, Y axis) ──
        #   Joint origin is at the far end of the coxa link (+X in coxa frame).
        lines += [
            f'  <joint name="{leg_name}_shoulder" type="revolute">',
            f'    <parent link="{leg_name}_coxa"/>',
            f'    <child  link="{leg_name}_femur"/>',
            f'    <origin xyz="{fmt_xyz((COXA_LEN, 0.0, 0.0))}" rpy="{fmt_rpy(0, 0, 0)}"/>',
            f'    <axis xyz="0 1 0"/>',
            f'    <limit lower="-1.5708" upper="1.5708" effort="2.0" velocity="6.28"/>',
            f'    <dynamics damping="0.01" friction="0.05"/>',
            f'  </joint>',
            '',
        ]

        # ── joint: femur → tibia (knee pitch, Y axis) ──
        #   Joint origin is at the far end of the femur (+X in femur frame).
        lines += [
            f'  <joint name="{leg_name}_knee" type="revolute">',
            f'    <parent link="{leg_name}_femur"/>',
            f'    <child  link="{leg_name}_tibia"/>',
            f'    <origin xyz="{fmt_xyz((FEMUR_LEN, 0.0, 0.0))}" rpy="{fmt_rpy(0, 0, 0)}"/>',
            f'    <axis xyz="0 1 0"/>',
            f'    <limit lower="-2.0944" upper="0.5236" effort="2.0" velocity="6.28"/>',
            f'    <dynamics damping="0.01" friction="0.05"/>',
            f'  </joint>',
            '',
        ]

        # ── joint: tibia → tip (fixed contact) ──
        lines += [
            f'  <joint name="{leg_name}_tip_joint" type="fixed">',
            f'    <parent link="{leg_name}_tibia"/>',
            f'    <child  link="{leg_name}_tip"/>',
            f'    <origin xyz="{fmt_xyz((TIBIA_LEN, 0.0, 0.0))}" rpy="{fmt_rpy(0, 0, 0)}"/>',
            f'  </joint>',
            '',
        ]

    lines += ['</robot>', '']

    out_path = os.path.join(HERE, "hexapod-v2.urdf")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote: {out_path}")
    return out_path


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Analysing meshes …")
    analyse()
    print("\nGenerating URDF …")
    generate_urdf()
    print("Done.")
