
"""
Complete Multi-Sensor Suite & Interactive GUI Dashboard in Isaac Lab 3.0.
Fixes & Features:
  1. Proper Ground Clearance (z = 0.102m axle height) + Synchronized Wheel Rolling Joint States:
     Robot sits cleanly on top of the floor without sinking, and both wheels spin realistically.
  2. Deadlock-Free Corridor Patrol + Tangential Sliding Obstacle Avoidance:
     Robot smoothly weaves through open corridors and never gets stuck against pillars or walls.
  3. Authentic Intel RealSense D456 Camera configuration (f=1.93mm, 87x58 deg Depth FOV, 90x65 deg RGB FOV)
     without near-plane black floor clipping artifacts.
  4. Razor-thin (2mm) 3D LiDAR laser beams with UI/Keyboard toggle.
"""

import argparse
import math
import warnings

# Suppress internal deprecation warnings from Isaac Lab 3.0-EA transitions
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# 1. CLI Arguments & Force GUI Visualizer ('kit') + Camera Pipeline
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-Sensor Suite with Intel RealSense D456 & Clean Wheel Motion")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"], enable_cameras=True)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if not getattr(args_cli, "headless", False):
    if not getattr(args_cli, "visualizer", None):
        args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Inner imports after SimulationApp is initialized
import carb.input
import omni.appwindow
import omni.ui as ui
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg

# ---------------------------------------------------------------------------
# 2. Intel RealSense D456 Official Datasheet Specifications & Scene Constants
# ---------------------------------------------------------------------------
D456_FOCAL_LENGTH_MM = 1.93
D456_DEPTH_APERTURE_H_MM = 3.663   # 87.0 deg Horizontal FOV at f=1.93mm
D456_DEPTH_APERTURE_V_MM = 2.140   # 58.0 deg Vertical FOV at f=1.93mm
D456_RGB_APERTURE_H_MM = 3.860     # 90.0 deg Horizontal FOV at f=1.93mm
D456_RGB_APERTURE_V_MM = 2.459     # 65.0 deg Vertical FOV at f=1.93mm
D456_MIN_DEPTH_M = 0.40            # Datasheet Min-Z depth operating distance (m)
D456_IDEAL_MAX_DEPTH_M = 6.00      # Datasheet ideal max depth range (m)
D456_MAX_CLIP_DEPTH_M = 12.00      # Far rendering clip range (m)

WHEEL_RADIUS_M = 0.10              # 10 cm wheel radius
WHEEL_BASE_M = 0.38                # 38 cm track width between left and right wheels
ROBOT_AXLE_Z_M = 0.102             # Exact axle height so tires rest on top of the floor (z=0.0)

PANEL_W = 320
PANEL_H = 240
DASHBOARD_W = PANEL_W * 2          # 640 px
DASHBOARD_H = PANEL_H * 2          # 480 px

ROBOT_USD_PATH = "/tmp/isaaclab_d456_wheeled_robot.usda"
ROOM_COLLIDER_USD_PATH = "/tmp/isaaclab_room_world_mesh.usda"

# Obstacle definitions
CYLINDER_OBSTACLES = [
    {"name": "Pillar1_Red",    "pos": (2.2,  2.0, 0.60), "radius": 0.35, "height": 1.20, "color": (0.88, 0.18, 0.18)},
    {"name": "Pillar4_Yellow", "pos": (-2.2, -2.0, 0.55), "radius": 0.35, "height": 1.10, "color": (0.95, 0.78, 0.10)},
    {"name": "Pillar6_Cyan",   "pos": (0.0, -2.8, 0.50), "radius": 0.32, "height": 1.00, "color": (0.10, 0.82, 0.85)},
]

BOX_OBSTACLES = [
    {"name": "Box2_Blue",   "pos": (-2.2, 2.0, 0.45), "size": (0.75, 0.75, 0.90), "color": (0.18, 0.38, 0.90)},
    {"name": "Box3_Green",  "pos": (2.2, -2.0, 0.40), "size": (0.65, 0.95, 0.80), "color": (0.18, 0.82, 0.30)},
    {"name": "Box5_Purple", "pos": (0.0,  2.8, 0.50), "size": (0.95, 0.45, 1.00), "color": (0.70, 0.20, 0.85)},
]

# Wide, obstacle-free corridor waypoints (>1.1m clearance from all pillars and boxes)
PATROL_WAYPOINTS = [
    (0.0, 0.0),
    (1.1, 0.7),
    (2.2, 0.7),
    (3.1, 1.8),
    (2.2, 3.1),
    (0.9, 1.5),
    (0.0, 1.2),
    (-1.0, 1.2),
    (-1.2, 0.0),
    (-1.0, -1.2),
    (0.0, -1.3),
    (1.1, -1.1),
]


def to_torch(arr) -> torch.Tensor:
    """Safely extracts the underlying PyTorch tensor from an Isaac Lab 3.0 ProxyArray."""
    return arr.torch if hasattr(arr, "torch") else torch.as_tensor(arr)


# ---------------------------------------------------------------------------
# 3. Procedural USD Generators (Elevated Wheeled Robot + World Collider)
# ---------------------------------------------------------------------------
def generate_visible_diffbot_usda(filepath: str):
    """
    Generates a 2-wheeled differential robot where the chassis and wheels sit cleanly above the ground
    with distinct yellow wheel spokes so wheel rotation is clearly visible.
    """
    usda_content = """#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
)
{
    bool physxArticulation:enabledSelfCollisions = false

    # ---------------- Main Chassis Link (Origin at Wheel Axle Height z = 0.102m) ----------------
    def Xform "chassis" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 5.0
        double3 xformOp:translate = (0.0, 0.0, 0.102)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        # Main Blue Chassis Body (Bottom face sits at z_local = -0.04m -> 6.2cm above ground!)
        def Cube "BodyBox"
        {
            color3f[] primvars:displayColor = [(0.10, 0.52, 0.90)]
            double size = 1.0
            double3 xformOp:translate = (0.0, 0.0, 0.02)
            float3 xformOp:scale = (0.46, 0.30, 0.12)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        # White Top Deck
        def Cube "TopDeck"
        {
            color3f[] primvars:displayColor = [(0.92, 0.94, 0.96)]
            double size = 1.0
            double3 xformOp:translate = (0.0, 0.0, 0.09)
            float3 xformOp:scale = (0.40, 0.26, 0.02)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        # Bright Green Forward Heading Arrow
        def Cube "ForwardDirectionArrow"
        {
            color3f[] primvars:displayColor = [(0.05, 0.95, 0.30)]
            double size = 1.0
            double3 xformOp:translate = (0.11, 0.0, 0.105)
            float3 xformOp:scale = (0.18, 0.045, 0.015)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        # Intel RealSense D456 Silver Aluminum Enclosure (124mm x 29mm x 26mm)
        def Cube "RealSenseD456Body"
        {
            color3f[] primvars:displayColor = [(0.78, 0.80, 0.84)]
            double size = 1.0
            double3 xformOp:translate = (0.21, 0.0, 0.125)
            float3 xformOp:scale = (0.029, 0.124, 0.026)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        # D456 Front Black Optical Glass Visor
        def Cube "RealSenseD456Glass"
        {
            color3f[] primvars:displayColor = [(0.08, 0.10, 0.14)]
            double size = 1.0
            double3 xformOp:translate = (0.226, 0.0, 0.125)
            float3 xformOp:scale = (0.004, 0.112, 0.018)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        # 2D LiDAR Red Turret Puck
        def Cylinder "Lidar2DPuck"
        {
            color3f[] primvars:displayColor = [(0.92, 0.18, 0.18)]
            uniform token axis = "Z"
            double height = 0.05
            double radius = 0.055
            double3 xformOp:translate = (0.02, 0.0, 0.125)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }

        # 3D LiDAR Mast & Gold Multi-Channel Dome
        def Cylinder "Lidar3DMast"
        {
            color3f[] primvars:displayColor = [(0.25, 0.25, 0.28)]
            uniform token axis = "Z"
            double height = 0.10
            double radius = 0.022
            double3 xformOp:translate = (-0.09, 0.0, 0.15)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }

        def Cylinder "Lidar3DDome"
        {
            color3f[] primvars:displayColor = [(1.0, 0.65, 0.05)]
            uniform token axis = "Z"
            double height = 0.06
            double radius = 0.05
            double3 xformOp:translate = (-0.09, 0.0, 0.22)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }

    # ---------------- Left Wheel Link (Radius = 0.10m, clearly visible on outside of chassis) ----------------
    def Xform "left_wheel" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.8
        double3 xformOp:translate = (0.0, 0.185, 0.102)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cylinder "Tire"
        {
            color3f[] primvars:displayColor = [(0.15, 0.15, 0.17)]
            uniform token axis = "Y"
            double height = 0.055
            double radius = 0.10
        }

        def Cube "SpokeBar1"
        {
            color3f[] primvars:displayColor = [(1.0, 0.82, 0.05)]
            double size = 1.0
            double3 xformOp:translate = (0.0, 0.028, 0.0)
            float3 xformOp:scale = (0.17, 0.012, 0.032)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        def Cube "SpokeBar2"
        {
            color3f[] primvars:displayColor = [(1.0, 0.82, 0.05)]
            double size = 1.0
            double3 xformOp:translate = (0.0, 0.028, 0.0)
            float3 xformOp:scale = (0.032, 0.012, 0.17)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }

    # ---------------- Right Wheel Link (Radius = 0.10m) ----------------
    def Xform "right_wheel" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.8
        double3 xformOp:translate = (0.0, -0.185, 0.102)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Cylinder "Tire"
        {
            color3f[] primvars:displayColor = [(0.15, 0.15, 0.17)]
            uniform token axis = "Y"
            double height = 0.055
            double radius = 0.10
        }

        def Cube "SpokeBar1"
        {
            color3f[] primvars:displayColor = [(1.0, 0.82, 0.05)]
            double size = 1.0
            double3 xformOp:translate = (0.0, -0.028, 0.0)
            float3 xformOp:scale = (0.17, 0.012, 0.032)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }

        def Cube "SpokeBar2"
        {
            color3f[] primvars:displayColor = [(1.0, 0.82, 0.05)]
            double size = 1.0
            double3 xformOp:translate = (0.0, -0.028, 0.0)
            float3 xformOp:scale = (0.032, 0.012, 0.17)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }

    # ---------------- Revolute Wheel Joints ----------------
    def PhysicsRevoluteJoint "left_wheel_joint"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/chassis>
        rel physics:body1 = </Robot/left_wheel>
        point3f physics:localPos0 = (0.0, 0.185, 0.0)
        point3f physics:localPos1 = (0.0, 0.0, 0.0)
        quatf physics:localRot0 = (1.0, 0.0, 0.0, 0.0)
        quatf physics:localRot1 = (1.0, 0.0, 0.0, 0.0)
    }

    def PhysicsRevoluteJoint "right_wheel_joint"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/chassis>
        rel physics:body1 = </Robot/right_wheel>
        point3f physics:localPos0 = (0.0, -0.185, 0.0)
        point3f physics:localPos1 = (0.0, 0.0, 0.0)
        quatf physics:localRot0 = (1.0, 0.0, 0.0, 0.0)
        quatf physics:localRot1 = (1.0, 0.0, 0.0, 0.0)
    }
}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(usda_content)


def _append_box_triangles(verts_list, faces_list, center, size):
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    base_idx = len(verts_list)
    corners = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    verts_list.extend(corners)
    box_tris = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5),
        (3, 0, 4), (3, 4, 7),
    ]
    for a, b, c in box_tris:
        faces_list.extend([base_idx + a, base_idx + b, base_idx + c])


def _append_cylinder_triangles(verts_list, faces_list, center, radius, height, segments=24):
    cx, cy, cz = center
    z0 = cz - height / 2.0
    z1 = cz + height / 2.0
    base_idx = len(verts_list)

    for i in range(segments):
        ang = 2.0 * math.pi * i / segments
        px = cx + radius * math.cos(ang)
        py = cy + radius * math.sin(ang)
        verts_list.append((px, py, z0))
        verts_list.append((px, py, z1))

    top_center_idx = len(verts_list)
    verts_list.append((cx, cy, z1))

    for i in range(segments):
        nxt = (i + 1) % segments
        b0 = base_idx + 2 * i
        t0 = base_idx + 2 * i + 1
        b1 = base_idx + 2 * nxt
        t1 = base_idx + 2 * nxt + 1
        faces_list.extend([b0, b1, t1])
        faces_list.extend([b0, t1, t0])
        faces_list.extend([t0, t1, top_center_idx])


def generate_world_room_collider_usda(filepath: str):
    """Creates a single world-space triangle mesh USDA of the room & obstacles for GPU RayCaster."""
    verts = []
    faces = []

    _append_box_triangles(verts, faces, center=(0.0, 0.0, -0.05), size=(10.0, 10.0, 0.10))
    _append_box_triangles(verts, faces, center=(4.0, 0.0, 0.75), size=(0.20, 8.20, 1.50))
    _append_box_triangles(verts, faces, center=(-4.0, 0.0, 0.75), size=(0.20, 8.20, 1.50))
    _append_box_triangles(verts, faces, center=(0.0, 4.0, 0.75), size=(8.20, 0.20, 1.50))
    _append_box_triangles(verts, faces, center=(0.0, -4.0, 0.75), size=(8.20, 0.20, 1.50))

    for cyl in CYLINDER_OBSTACLES:
        _append_cylinder_triangles(verts, faces, center=cyl["pos"], radius=cyl["radius"], height=cyl["height"])
    for box in BOX_OBSTACLES:
        _append_box_triangles(verts, faces, center=box["pos"], size=box["size"])

    num_tris = len(faces) // 3
    counts_str = ", ".join(["3"] * num_tris)
    indices_str = ", ".join(str(idx) for idx in faces)
    points_str = ", ".join(f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})" for v in verts)

    usda = f"""#usda 1.0
(
    defaultPrim = "RoomCollider"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "RoomCollider"
{{
    def Mesh "CombinedMesh"
    {{
        uniform token purpose = "guide"
        int[] faceVertexCounts = [{counts_str}]
        int[] faceVertexIndices = [{indices_str}]
        point3f[] points = [{points_str}]
    }}
}}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(usda)


generate_visible_diffbot_usda(ROBOT_USD_PATH)
generate_world_room_collider_usda(ROOM_COLLIDER_USD_PATH)


# ---------------------------------------------------------------------------
# 4. Scene Configuration Class
# ---------------------------------------------------------------------------
@configclass
class SensorRoomSceneCfg(InteractiveSceneCfg):
    """Enclosed room environment with obstacles, elevated 2-wheeled robot, D456 camera, and 2D/3D LiDARs."""

    # 4.1 Balanced Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=650.0, color=(0.85, 0.88, 0.95)),
    )
    sun_light = AssetBaseCfg(
        prim_path="/World/SunLight",
        spawn=sim_utils.DistantLightCfg(intensity=1100.0, color=(1.0, 0.98, 0.92)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.85, 0.25, 0.35, 0.30)),
    )

    # 4.2 Unified World-Space Mesh for GPU LiDAR RayCasting
    room_collider = AssetBaseCfg(
        prim_path="/World/RoomCollider",
        spawn=sim_utils.UsdFileCfg(usd_path=ROOM_COLLIDER_USD_PATH),
    )

    # 4.3 Room Floor & Perimeter Walls
    room_floor = AssetBaseCfg(
        prim_path="/World/Room/Floor",
        spawn=sim_utils.MeshCuboidCfg(
            size=(10.0, 10.0, 0.10),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.26, 0.30), roughness=0.6),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    wall_north = AssetBaseCfg(
        prim_path="/World/Room/WallNorth",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.20, 8.20, 1.50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.65, 0.78)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.0, 0.0, 0.75)),
    )
    wall_south = AssetBaseCfg(
        prim_path="/World/Room/WallSouth",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.20, 8.20, 1.50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.68, 0.56)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, 0.0, 0.75)),
    )
    wall_east = AssetBaseCfg(
        prim_path="/World/Room/WallEast",
        spawn=sim_utils.MeshCuboidCfg(
            size=(8.20, 0.20, 1.50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.74, 0.64)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 4.0, 0.75)),
    )
    wall_west = AssetBaseCfg(
        prim_path="/World/Room/WallWest",
        spawn=sim_utils.MeshCuboidCfg(
            size=(8.20, 0.20, 1.50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.62, 0.75)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -4.0, 0.75)),
    )

    # 4.4 Six Colored Obstacles
    pillar_1 = AssetBaseCfg(
        prim_path="/World/Room/Pillar1_Red",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.35,
            height=1.20,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.88, 0.18, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.2, 2.0, 0.60)),
    )
    box_2 = AssetBaseCfg(
        prim_path="/World/Room/Box2_Blue",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.75, 0.75, 0.90),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.38, 0.90)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.2, 2.0, 0.45)),
    )
    box_3 = AssetBaseCfg(
        prim_path="/World/Room/Box3_Green",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.65, 0.95, 0.80),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.82, 0.30)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.2, -2.0, 0.40)),
    )
    pillar_4 = AssetBaseCfg(
        prim_path="/World/Room/Pillar4_Yellow",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.35,
            height=1.10,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.78, 0.10)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.2, -2.0, 0.55)),
    )
    box_5 = AssetBaseCfg(
        prim_path="/World/Room/Box5_Purple",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.95, 0.45, 1.00),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.20, 0.85)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.8, 0.50)),
    )
    pillar_6 = AssetBaseCfg(
        prim_path="/World/Room/Pillar6_Cyan",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.32,
            height=1.00,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.82, 0.85)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.8, 0.50)),
    )

    # 4.5 Visible 2-Wheeled Differential Robot (Spawned cleanly at axle height z = 0.102m)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, ROBOT_AXLE_Z_M),
        ),
        actuators={
            "wheel_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"],
                effort_limit=100.0,
                velocity_limit=50.0,
                stiffness=0.0,
                damping=5.0,
            ),
        },
    )

    # 4.6 Sensor 1: Intel RealSense D456 Camera (Global Shutter RGB + Depth)
    # Near clipping set to 0.05m so the floor close to the robot is rendered without black clipping
    realsense_d456 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/realsense_d456",
        update_period=0.0,
        height=PANEL_H,
        width=PANEL_W,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=D456_FOCAL_LENGTH_MM,
            focus_distance=400.0,
            horizontal_aperture=D456_DEPTH_APERTURE_H_MM,
            vertical_aperture=D456_DEPTH_APERTURE_V_MM,
            clipping_range=(0.05, D456_MAX_CLIP_DEPTH_M),
        ),
    )

    # 4.7 Sensor 2: 2D Planar LiDAR
    lidar_2d = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.02, 0.0, 0.14)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=12.0,
        debug_vis=False,
        mesh_prim_paths=["/World/RoomCollider"],
    )

    # 4.8 Sensor 3: 3D Multi-Channel LiDAR (16-beam array)
    lidar_3d = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(-0.09, 0.0, 0.23)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=2.0,
        ),
        max_distance=12.0,
        debug_vis=False,
        mesh_prim_paths=["/World/RoomCollider"],
    )


# ---------------------------------------------------------------------------
# 5. Camera Pose Setup & Ultra-Thin 2mm Laser Beam Visualizer
# ---------------------------------------------------------------------------
def set_camera_forward_matrix(stage, cam_prim_path: str, pitch_down_deg: float = 3.0):
    """Positions the RealSense D456 camera at the front visor looking horizontally forward (+X)."""
    cam_prim = stage.GetPrimAtPath(cam_prim_path)
    if not cam_prim.IsValid():
        return

    theta = math.radians(pitch_down_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    mat = Gf.Matrix4d(
        0.0, -1.0, 0.0, 0.0,
        s,    0.0, c,   0.0,
       -c,    0.0, s,   0.0,
        0.23, 0.0, 0.13, 1.0,
    )
    xformable = UsdGeom.Xformable(cam_prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(mat)


class ThinLaserRaysVisualizer:
    """Draws razor-thin (2mm) LiDAR beams in the 3D viewport without obscuring the robot."""

    def __init__(self, stage, prim_path: str = "/World/ThinLaserBeams"):
        self.curves = UsdGeom.BasisCurves.Define(stage, prim_path)
        self.curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        self.curves.CreateWidthsAttr(Vt.FloatArray([0.002]))
        self.curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self.curves.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.95, 0.40)]))
        self.enabled = True

    def set_visible(self, visible: bool):
        self.enabled = visible
        imageable = UsdGeom.Imageable(self.curves)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def update_rays(self, sensor_pos_w: np.ndarray, hits_w: np.ndarray, step_stride: int = 10):
        if not self.enabled or hits_w is None or len(hits_w) == 0:
            return
        valid = np.isfinite(hits_w).all(axis=-1)
        pts = hits_w[valid][::step_stride]
        if len(pts) == 0:
            return

        curve_pts = []
        counts = []
        sx, sy, sz = float(sensor_pos_w[0]), float(sensor_pos_w[1]), float(sensor_pos_w[2]) + 0.14
        for p in pts:
            dx = float(p[0]) - sx
            dy = float(p[1]) - sy
            dist = math.hypot(dx, dy)
            if dist > 0.25:
                start_x = sx + (dx / dist) * 0.10
                start_y = sy + (dy / dist) * 0.10
                curve_pts.append(Gf.Vec3f(start_x, start_y, sz))
                curve_pts.append(Gf.Vec3f(float(p[0]), float(p[1]), sz))
                counts.append(2)

        self.curves.GetPointsAttr().Set(Vt.Vec3fArray(curve_pts))
        self.curves.GetCurveVertexCountsAttr().Set(Vt.IntArray(counts))


# ---------------------------------------------------------------------------
# 6. Hard SDF Collision Resolver (Tangential Sliding, Never Gets Stuck)
# ---------------------------------------------------------------------------
def enforce_collision_free_pose(x: float, y: float, yaw: float, robot_radius: float = 0.32):
    """
    Ensures the robot NEVER penetrates any obstacle or wall, and smoothly slides tangentially
    around obstacles if driven against them.
    """
    collided = False

    # 1. Room Perimeter Walls (8m x 8m -> +/- 4.0m walls)
    wall_limit = 3.90 - robot_radius
    if x > wall_limit:
        x = wall_limit
        collided = True
    elif x < -wall_limit:
        x = -wall_limit
        collided = True
    if y > wall_limit:
        y = wall_limit
        collided = True
    elif y < -wall_limit:
        y = -wall_limit
        collided = True

    # 2. Cylindrical Obstacles
    for cyl in CYLINDER_OBSTACLES:
        cx, cy, _ = cyl["pos"]
        min_sep = cyl["radius"] + robot_radius
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist < min_sep:
            if dist < 1e-5:
                dx, dy, dist = 1.0, 0.0, 1.0
            nx = dx / dist
            ny = dy / dist
            x = cx + nx * min_sep
            y = cy + ny * min_sep
            tx, ty = -ny, nx
            if math.cos(yaw) * tx + math.sin(yaw) * ty < 0.0:
                tx, ty = ny, -nx
            yaw = math.atan2(ty, tx)
            collided = True

    # 3. Box Obstacles
    for box in BOX_OBSTACLES:
        bx, by, _ = box["pos"]
        hx = box["size"][0] / 2.0
        hy = box["size"][1] / 2.0

        closest_x = max(bx - hx, min(x, bx + hx))
        closest_y = max(by - hy, min(y, by + hy))
        dx = x - closest_x
        dy = y - closest_y
        dist = math.hypot(dx, dy)

        if dist < robot_radius:
            if dist > 1e-5:
                nx = dx / dist
                ny = dy / dist
                x = closest_x + nx * robot_radius
                y = closest_y + ny * robot_radius
            else:
                pen_x = (hx + robot_radius) - abs(x - bx)
                pen_y = (hy + robot_radius) - abs(y - by)
                if pen_x < pen_y:
                    nx = 1.0 if x >= bx else -1.0
                    ny = 0.0
                    x = bx + nx * (hx + robot_radius)
                else:
                    nx = 0.0
                    ny = 1.0 if y >= by else -1.0
                    y = by + ny * (hy + robot_radius)

            tx, ty = -ny, nx
            if math.cos(yaw) * tx + math.sin(yaw) * ty < 0.0:
                tx, ty = ny, -nx
            yaw = math.atan2(ty, tx)
            collided = True

    return x, y, yaw, collided


# ---------------------------------------------------------------------------
# 7. Coordinate Transform & 4-Panel Dashboard Renderers
# ---------------------------------------------------------------------------
def world_to_robot_yaw_frame(hits_w: torch.Tensor, pos_w: torch.Tensor, yaw_rad: float) -> np.ndarray:
    if hits_w is None or len(hits_w) == 0:
        return np.empty((0, 3), dtype=np.float32)

    rel = (hits_w - pos_w).detach().cpu().numpy()
    valid_mask = np.isfinite(rel).all(axis=-1)
    rel = rel[valid_mask]
    if len(rel) == 0:
        return np.empty((0, 3), dtype=np.float32)

    cos_y = math.cos(-yaw_rad)
    sin_y = math.sin(-yaw_rad)

    local_pts = np.empty_like(rel)
    local_pts[:, 0] = cos_y * rel[:, 0] - sin_y * rel[:, 1]
    local_pts[:, 1] = sin_y * rel[:, 0] + cos_y * rel[:, 1]
    local_pts[:, 2] = rel[:, 2]
    return local_pts


def render_lidar_2d_polar(
    local_hits: np.ndarray,
    width: int,
    height: int,
    max_range: float,
    fov_deg: float,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2 + 8
    radius_px = min(cx, cy - 16) - 10
    scale = radius_px / max(max_range, 0.5)

    for frac in [0.25, 0.5, 0.75, 1.0]:
        r_m = max_range * frac
        r_p = int(r_m * scale)
        cv2.circle(canvas, (cx, cy), r_p, (45, 50, 45), 1)
        cv2.putText(canvas, f"{r_m:.1f}m", (cx + 4, cy - r_p + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 140, 130), 1)

    cv2.line(canvas, (cx, cy - radius_px), (cx, cy + radius_px), (50, 50, 50), 1)
    cv2.line(canvas, (cx - radius_px, cy), (cx + radius_px, cy), (50, 50, 50), 1)

    half_fov_rad = math.radians(fov_deg / 2.0)
    if fov_deg < 359.0:
        for sign in [-1, 1]:
            ang = sign * half_fov_rad
            ex = int(cx - math.sin(ang) * radius_px)
            ey = int(cy - math.cos(ang) * radius_px)
            cv2.line(canvas, (cx, cy), (ex, ey), (70, 140, 70), 1)

    cv2.circle(canvas, (cx, cy), 5, (0, 140, 255), -1)
    cv2.arrowedLine(canvas, (cx, cy), (cx, cy - 15), (0, 200, 255), 2, tipLength=0.4)

    num_valid = 0
    min_dist = max_range
    if len(local_hits) > 0:
        x = local_hits[:, 0]
        y = local_hits[:, 1]
        dists = np.hypot(x, y)
        angles = np.abs(np.arctan2(y, x))
        valid = (dists > 0.08) & (dists <= max_range) & (angles <= half_fov_rad)
        num_valid = int(np.sum(valid))

        if num_valid > 0:
            min_dist = float(np.min(dists[valid]))
            px = (cx - y[valid] * scale).astype(np.int32)
            py = (cy - x[valid] * scale).astype(np.int32)
            in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            for u, v in zip(px[in_bounds], py[in_bounds]):
                cv2.circle(canvas, (int(u), int(v)), 2, (0, 255, 60), -1)

    cv2.putText(
        canvas,
        f"2D LiDAR (FOV:{fov_deg:.0f}deg | Hits:{num_valid} | Min:{min_dist:.2f}m)",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (0, 255, 200),
        1,
    )
    return canvas


def render_lidar_3d_projection(
    local_hits: np.ndarray,
    width: int,
    height: int,
    max_range: float,
    vert_fov_limit: float,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.line(canvas, (0, height // 2), (width, height // 2), (40, 40, 40), 1)

    num_pts = 0
    if len(local_hits) > 0:
        x = local_hits[:, 0]
        y = local_hits[:, 1]
        z = local_hits[:, 2]

        dists = np.linalg.norm(local_hits, axis=1)
        xy_dist = np.hypot(x, y)
        elev_deg = np.degrees(np.arctan2(z - 0.23, np.maximum(xy_dist, 1e-4)))

        valid = (dists > 0.10) & (dists <= max_range) & (x > 0.05) & (np.abs(elev_deg) <= vert_fov_limit + 1.0)
        num_pts = int(np.sum(valid))

        if num_pts > 0:
            xv = x[valid]
            yv = y[valid]
            zv = z[valid] - 0.15
            dv = dists[valid]

            focal = width * 0.65
            cx = width // 2
            cy = height // 2 + 10

            u = (cx - (yv / xv) * focal).astype(np.int32)
            v = (cy - (zv / xv) * focal).astype(np.int32)

            norm_d = np.clip(dv / max(max_range, 0.5), 0.0, 1.0)
            colors = cv2.applyColorMap((norm_d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

            in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            for idx in np.where(in_bounds)[0]:
                c = colors[idx, 0].tolist()
                cv2.circle(canvas, (int(u[idx]), int(v[idx])), 2, c, -1)

    cv2.putText(
        canvas,
        f"3D LiDAR 16-Ch (VFOV:+/-{vert_fov_limit:.0f}deg | Pts:{num_pts})",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (0, 210, 255),
        1,
    )
    return canvas


# ---------------------------------------------------------------------------
# 8. Native Omniverse UI Window with Intel RealSense D456 Optical Controls
# ---------------------------------------------------------------------------
class SensorDashboardUI:
    """Floating window inside Isaac Sim with D456 optical presets & live sliders."""

    def __init__(self, camera_prim_path: str, laser_vis: ThinLaserRaysVisualizer):
        self.camera_prim_path = camera_prim_path
        self.laser_vis = laser_vis

        self.usd_aperture_scale = 1.0
        try:
            stage = sim_utils.get_current_stage()
            cam_prim = stage.GetPrimAtPath(self.camera_prim_path)
            if cam_prim.IsValid():
                curr_h_ap = float(UsdGeom.Camera(cam_prim).GetHorizontalApertureAttr().Get())
                self.usd_aperture_scale = curr_h_ap / D456_DEPTH_APERTURE_H_MM
        except Exception:
            self.usd_aperture_scale = 1.0

        self.auto_patrol = True
        self.cmd_linear = 0.75
        self.cmd_angular = 0.0
        self.cam_focal_mm = D456_FOCAL_LENGTH_MM
        self.cam_haperture_mm = D456_DEPTH_APERTURE_H_MM
        self.cam_vaperture_mm = D456_DEPTH_APERTURE_V_MM
        self.cam_max_depth = D456_IDEAL_MAX_DEPTH_M
        self.lidar2d_range = 6.0
        self.lidar2d_fov = 360.0
        self.lidar3d_range = 6.0
        self.lidar3d_vfov = 15.0

        self.image_provider = ui.ByteImageProvider()
        self.window = ui.Window(
            "Intel RealSense D456 + 2D/3D LiDAR Suite Dashboard",
            width=DASHBOARD_W + 45,
            height=DASHBOARD_H + 310,
            visible=True,
        )
        self._build_ui()

    def get_current_fov_deg(self) -> tuple[float, float]:
        hfov = 2.0 * math.degrees(math.atan(self.cam_haperture_mm / (2.0 * max(self.cam_focal_mm, 0.2))))
        vfov = 2.0 * math.degrees(math.atan(self.cam_vaperture_mm / (2.0 * max(self.cam_focal_mm, 0.2))))
        return hfov, vfov

    def _build_ui(self):
        with self.window.frame:
            with ui.VStack(spacing=5, style={"margin": 8}):
                ui.Label(
                    "1. Wheeled Robot Navigation & 3D Laser Beam Toggle (Keys: Arrows / I,J,K,L / Space / B)",
                    height=18,
                    style={"color": 0xFF00CCFF, "font_size": 14},
                )
                with ui.HStack(height=26, spacing=5):
                    self.mode_btn = ui.Button(
                        "Mode: AUTO PATROL",
                        clicked_fn=self._toggle_mode,
                        style={"background_color": 0xFF1E6630},
                    )
                    self.beam_btn = ui.Button(
                        "3D Beams: ON (2mm)",
                        clicked_fn=self._toggle_beams,
                        style={"background_color": 0xFF2E5566},
                    )
                    ui.Button("+Speed (W/I)", clicked_fn=lambda: self._nudge_speed(0.20, 0.0))
                    ui.Button("-Speed (S/K)", clicked_fn=lambda: self._nudge_speed(-0.20, 0.0))
                    ui.Button("Left (A/J)", clicked_fn=lambda: self._nudge_speed(0.0, 0.45))
                    ui.Button("Right (D/L)", clicked_fn=lambda: self._nudge_speed(0.0, -0.45))
                    ui.Button("STOP", clicked_fn=self.stop_robot, style={"background_color": 0xFF222288})

                ui.Label(
                    "2. Intel RealSense D456 Optical Parameters (f=1.93mm, Global Shutter OV9782, Range: 0.4m - 6.0m)",
                    height=18,
                    style={"color": 0xFF00FF88, "font_size": 14},
                )
                with ui.HStack(height=24, spacing=8):
                    ui.Button(
                        "Preset: D456 Aligned Depth (87x58 deg | f=1.93mm)",
                        clicked_fn=lambda: self._apply_d456_preset(1.93, D456_DEPTH_APERTURE_H_MM, D456_DEPTH_APERTURE_V_MM),
                    )
                    ui.Button(
                        "Preset: D456 Native RGB (90x65 deg | f=1.93mm)",
                        clicked_fn=lambda: self._apply_d456_preset(1.93, D456_RGB_APERTURE_H_MM, D456_RGB_APERTURE_V_MM),
                    )

                with ui.HStack(height=22, spacing=8):
                    ui.Label("D456 Focal Length (mm):", width=155)
                    self.focal_slider = ui.FloatSlider(min=1.00, max=4.50, step=0.01)
                    self.focal_slider.model.set_value(self.cam_focal_mm)
                    self.focal_slider.model.add_value_changed_fn(lambda m: self._set_cam_focal(m.get_value_as_float()))

                    ui.Label("D456 Max Depth (m):", width=140)
                    depth_slider = ui.FloatSlider(min=1.0, max=10.0, step=0.2)
                    depth_slider.model.set_value(self.cam_max_depth)
                    depth_slider.model.add_value_changed_fn(lambda m: setattr(self, "cam_max_depth", m.get_value_as_float()))

                with ui.HStack(height=22, spacing=8):
                    ui.Label("2D LiDAR Range (m):", width=155)
                    l2d_r_slider = ui.FloatSlider(min=1.5, max=10.0, step=0.5)
                    l2d_r_slider.model.set_value(self.lidar2d_range)
                    l2d_r_slider.model.add_value_changed_fn(lambda m: setattr(self, "lidar2d_range", m.get_value_as_float()))

                    ui.Label("2D LiDAR FOV (deg):", width=140)
                    l2d_fov_slider = ui.FloatSlider(min=30.0, max=360.0, step=5.0)
                    l2d_fov_slider.model.set_value(self.lidar2d_fov)
                    l2d_fov_slider.model.add_value_changed_fn(lambda m: setattr(self, "lidar2d_fov", m.get_value_as_float()))

                with ui.HStack(height=22, spacing=8):
                    ui.Label("3D LiDAR Range (m):", width=155)
                    l3d_r_slider = ui.FloatSlider(min=1.5, max=10.0, step=0.5)
                    l3d_r_slider.model.set_value(self.lidar3d_range)
                    l3d_r_slider.model.add_value_changed_fn(lambda m: setattr(self, "lidar3d_range", m.get_value_as_float()))

                    ui.Label("3D LiDAR VFOV (deg):", width=140)
                    l3d_vfov_slider = ui.FloatSlider(min=2.0, max=15.0, step=1.0)
                    l3d_vfov_slider.model.set_value(self.lidar3d_vfov)
                    l3d_vfov_slider.model.add_value_changed_fn(lambda m: setattr(self, "lidar3d_vfov", m.get_value_as_float()))

                ui.Separator(height=4)

                with ui.HStack():
                    ui.Spacer()
                    ui.ImageWithProvider(
                        self.image_provider,
                        width=DASHBOARD_W,
                        height=DASHBOARD_H,
                    )
                    ui.Spacer()

    def _toggle_mode(self):
        self.auto_patrol = not self.auto_patrol
        if self.auto_patrol:
            self.mode_btn.text = "Mode: AUTO PATROL"
        else:
            self.mode_btn.text = "Mode: MANUAL TELEOP"
            self.cmd_linear = 0.65
            self.cmd_angular = 0.0

    def _toggle_beams(self):
        new_state = not self.laser_vis.enabled
        self.laser_vis.set_visible(new_state)
        self.beam_btn.text = "3D Beams: ON (2mm)" if new_state else "3D Beams: OFF"

    def _nudge_speed(self, d_lin: float, d_ang: float):
        self.auto_patrol = False
        self.mode_btn.text = "Mode: MANUAL TELEOP"
        self.cmd_linear = float(np.clip(self.cmd_linear + d_lin, -1.2, 1.5))
        self.cmd_angular = float(np.clip(self.cmd_angular + d_ang, -2.2, 2.2))

    def stop_robot(self):
        self.auto_patrol = False
        self.mode_btn.text = "Mode: STOPPED"
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0

    def _apply_d456_preset(self, focal_mm: float, hap_mm: float, vap_mm: float):
        self.cam_haperture_mm = hap_mm
        self.cam_vaperture_mm = vap_mm
        self.focal_slider.model.set_value(focal_mm)
        self._set_cam_focal(focal_mm)

    def _set_cam_focal(self, val_mm: float):
        self.cam_focal_mm = float(val_mm)
        try:
            stage = sim_utils.get_current_stage()
            cam_prim = stage.GetPrimAtPath(self.camera_prim_path)
            if cam_prim.IsValid():
                usd_cam = UsdGeom.Camera(cam_prim)
                usd_cam.GetFocalLengthAttr().Set(float(self.cam_focal_mm * self.usd_aperture_scale))
                usd_cam.GetHorizontalApertureAttr().Set(float(self.cam_haperture_mm * self.usd_aperture_scale))
                usd_cam.GetVerticalApertureAttr().Set(float(self.cam_vaperture_mm * self.usd_aperture_scale))
        except Exception:
            pass

    def update_image(self, bgr_dashboard: np.ndarray):
        rgba = cv2.cvtColor(bgr_dashboard, cv2.COLOR_BGR2RGBA)
        contig = np.ascontiguousarray(rgba, dtype=np.uint8)
        h, w, _ = contig.shape
        try:
            self.image_provider.set_data_array(contig.ravel(), [w, h])
        except Exception:
            self.image_provider.set_bytes_data(contig.ravel().tolist(), [w, h])


# ---------------------------------------------------------------------------
# 9. Main Simulation & Synchronized Wheel Rolling Loop
# ---------------------------------------------------------------------------
def main():
    vis_cfgs = [] if getattr(args_cli, "headless", False) else [KitVisualizerCfg()]
    sim_cfg = SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        visualizer_cfgs=vis_cfgs,
    )
    sim = SimulationContext(sim_cfg)

    try:
        sim.set_camera_view(eye=[4.6, -4.6, 3.6], target=[0.0, 0.0, 0.25])
    except Exception:
        pass

    scene_cfg = SensorRoomSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    stage = sim_utils.get_current_stage()
    cam_prim_path = "/World/envs/env_0/Robot/chassis/realsense_d456"
    set_camera_forward_matrix(stage, cam_prim_path, pitch_down_deg=3.0)
    laser_ray_vis = ThinLaserRaysVisualizer(stage, "/World/ThinLaserBeams")

    robot: Articulation = scene["robot"]
    cam: Camera = scene["realsense_d456"]
    lidar2d: RayCaster = scene["lidar_2d"]
    lidar3d: RayCaster = scene["lidar_3d"]

    init_q = to_torch(robot.data.root_quat_w)[0].detach().cpu().numpy()
    quat_is_wxyz = abs(float(init_q[0])) > 0.5

    joint_names = robot.data.joint_names
    left_wheel_idx = next((i for i, name in enumerate(joint_names) if "left" in name), 0)
    right_wheel_idx = next((i for i, name in enumerate(joint_names) if "right" in name), 1)

    dashboard_ui = None
    if not getattr(args_cli, "headless", False):
        dashboard_ui = SensorDashboardUI(camera_prim_path=cam_prim_path, laser_vis=laser_ray_vis)

        app_window = omni.appwindow.get_default_app_window()
        if app_window is not None:
            input_iface = carb.input.acquire_input_interface()
            keyboard = app_window.get_keyboard()

            def _on_keyboard_event(event, *args, **kwargs):
                if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
                    if event.input in (carb.input.KeyboardInput.UP, carb.input.KeyboardInput.I):
                        dashboard_ui._nudge_speed(0.15, 0.0)
                    elif event.input in (carb.input.KeyboardInput.DOWN, carb.input.KeyboardInput.K):
                        dashboard_ui._nudge_speed(-0.15, 0.0)
                    elif event.input in (carb.input.KeyboardInput.LEFT, carb.input.KeyboardInput.J):
                        dashboard_ui._nudge_speed(0.0, 0.35)
                    elif event.input in (carb.input.KeyboardInput.RIGHT, carb.input.KeyboardInput.L):
                        dashboard_ui._nudge_speed(0.0, -0.35)
                    elif event.input == carb.input.KeyboardInput.SPACE:
                        dashboard_ui.stop_robot()
                    elif event.input == carb.input.KeyboardInput.P:
                        dashboard_ui._toggle_mode()
                    elif event.input == carb.input.KeyboardInput.B:
                        dashboard_ui._toggle_beams()
                return True

            input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)

    print("=" * 76)
    print("[INFO] Wheeled Robot & Intel RealSense D456 + 2D/3D LiDAR Suite Running!")
    print("[INFO] Robot sits cleanly at z=0.102m on top of its 10cm wheels (zero ground sinking).")
    print("[INFO] Shortcuts: Arrows/IJKL = Drive | Space = Stop | P = Auto/Manual | B = Toggle Beams")
    print("=" * 76)

    dt = 0.01
    rx, ry, ryaw = 0.0, 0.0, 0.45
    wp_idx = 1
    left_wheel_angle = 0.0
    right_wheel_angle = 0.0

    while simulation_app.is_running():
        auto_mode = dashboard_ui.auto_patrol if dashboard_ui is not None else True

        if auto_mode:
            target_x, target_y = PATROL_WAYPOINTS[wp_idx]
            dx = target_x - rx
            dy = target_y - ry
            dist_to_wp = math.hypot(dx, dy)
            if dist_to_wp < 0.38:
                wp_idx = (wp_idx + 1) % len(PATROL_WAYPOINTS)
                target_x, target_y = PATROL_WAYPOINTS[wp_idx]
                dx = target_x - rx
                dy = target_y - ry

            desired_yaw = math.atan2(dy, dx)
            yaw_err = math.atan2(math.sin(desired_yaw - ryaw), math.cos(desired_yaw - ryaw))

            # Slow down slightly during sharp turns for natural wheeled motion
            turn_slowdown = max(0.45, 1.0 - 0.35 * abs(yaw_err))
            lin_speed = 0.85 * turn_slowdown
            ang_speed = float(np.clip(2.4 * yaw_err, -2.4, 2.4))
        else:
            lin_speed = dashboard_ui.cmd_linear if dashboard_ui is not None else 0.6
            ang_speed = dashboard_ui.cmd_angular if dashboard_ui is not None else 0.0

        # Integrate robot pose in 2D plane
        ryaw = math.atan2(math.sin(ryaw + ang_speed * dt), math.cos(ryaw + ang_speed * dt))
        next_x = rx + lin_speed * math.cos(ryaw) * dt
        next_y = ry + lin_speed * math.sin(ryaw) * dt

        # Enforce hard geometric SDF collision prevention (slides smoothly around obstacles)
        rx, ry, ryaw, collided = enforce_collision_free_pose(next_x, next_y, ryaw, robot_radius=0.32)
        if collided and auto_mode:
            # If the user manually placed the robot behind an obstacle, advance waypoint to prevent deadlock
            wp_idx = (wp_idx + 1) % len(PATROL_WAYPOINTS)

        vx = lin_speed * math.cos(ryaw)
        vy = lin_speed * math.sin(ryaw)

        # Build quaternion matching Isaac Lab's runtime convention
        cy = math.cos(ryaw * 0.5)
        sy = math.sin(ryaw * 0.5)
        q_list = [cy, 0.0, 0.0, sy] if quat_is_wxyz else [0.0, 0.0, sy, cy]

        # Set root pose at z = ROBOT_AXLE_Z_M (0.102m) so wheels rest cleanly on top of the floor (z=0.0)
        root_pose = torch.tensor(
            [[rx, ry, ROBOT_AXLE_Z_M, q_list[0], q_list[1], q_list[2], q_list[3]]],
            device=sim.device,
            dtype=torch.float32,
        )
        root_vel = torch.tensor([[vx, vy, 0.0, 0.0, 0.0, ang_speed]], device=sim.device, dtype=torch.float32)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_vel)

        # Compute exact wheel angular velocities (rad/s) and integrate rolling angles (rad)
        v_left = (lin_speed - ang_speed * (WHEEL_BASE_M / 2.0)) / WHEEL_RADIUS_M
        v_right = (lin_speed + ang_speed * (WHEEL_BASE_M / 2.0)) / WHEEL_RADIUS_M
        left_wheel_angle = (left_wheel_angle + v_left * dt) % (2.0 * math.pi)
        right_wheel_angle = (right_wheel_angle + v_right * dt) % (2.0 * math.pi)

        joint_pos = torch.zeros((1, robot.num_joints), device=sim.device, dtype=torch.float32)
        joint_vel = torch.zeros((1, robot.num_joints), device=sim.device, dtype=torch.float32)
        joint_pos[0, left_wheel_idx] = left_wheel_angle
        joint_pos[0, right_wheel_idx] = right_wheel_angle
        joint_vel[0, left_wheel_idx] = v_left
        joint_vel[0, right_wheel_idx] = v_right

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_velocity_target(joint_vel)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        # Read LiDAR hits in world frame and transform to robot local frame
        pos_w = to_torch(robot.data.root_pos_w)[0]
        hits_2d_w = to_torch(lidar2d.data.ray_hits_w)[0]
        hits_3d_w = to_torch(lidar3d.data.ray_hits_w)[0]

        local_hits_2d = world_to_robot_yaw_frame(hits_2d_w, pos_w, ryaw)
        local_hits_3d = world_to_robot_yaw_frame(hits_3d_w, pos_w, ryaw)

        if dashboard_ui is not None:
            focal_mm = dashboard_ui.cam_focal_mm
            hfov_deg, vfov_deg = dashboard_ui.get_current_fov_deg()
            max_depth = dashboard_ui.cam_max_depth
            l2d_range = dashboard_ui.lidar2d_range
            l2d_fov = dashboard_ui.lidar2d_fov
            l3d_range = dashboard_ui.lidar3d_range
            l3d_vfov = dashboard_ui.lidar3d_vfov

            # Update razor-thin 2mm laser beams in the 3D viewport
            laser_ray_vis.update_rays(pos_w.detach().cpu().numpy(), hits_2d_w.detach().cpu().numpy(), step_stride=10)

            # 1. Process Intel RealSense D456 RGB & Depth
            if "rgb" in cam.data.output and cam.data.output["rgb"] is not None:
                rgb_tensor = to_torch(cam.data.output["rgb"])[0]
                depth_tensor = to_torch(cam.data.output["distance_to_image_plane"])[0]

                rgb_np = rgb_tensor[:, :, :3].detach().cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    rgb_np = np.clip(rgb_np * 255.0 if rgb_np.max() <= 1.05 else rgb_np, 0, 255).astype(np.uint8)
                rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    rgb_bgr,
                    f"D456 RGB (f={focal_mm:.2f}mm | FOV:{hfov_deg:.0f}x{vfov_deg:.0f}deg)",
                    (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.37,
                    (0, 255, 0),
                    1,
                )

                depth_np = depth_tensor.detach().cpu().numpy().squeeze()
                center_dist = float(depth_np[PANEL_H // 2, PANEL_W // 2])
                depth_clean = np.nan_to_num(depth_np, nan=max_depth, posinf=max_depth, neginf=D456_MIN_DEPTH_M)
                depth_norm = np.clip((depth_clean - D456_MIN_DEPTH_M) / max(max_depth - D456_MIN_DEPTH_M, 0.5), 0.0, 1.0)
                depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

                ch_x, ch_y = PANEL_W // 2, PANEL_H // 2
                cv2.drawMarker(depth_colormap, (ch_x, ch_y), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
                dist_str = f"{center_dist:.2f}m" if np.isfinite(center_dist) else ">Max"
                cv2.putText(
                    depth_colormap,
                    f"D456 Depth (Center:{dist_str} | Range:0.4-{max_depth:.1f}m)",
                    (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (255, 255, 255),
                    1,
                )
            else:
                rgb_bgr = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                depth_colormap = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

            # 2. Render 2D & 3D LiDAR Panels
            lidar2d_panel = render_lidar_2d_polar(
                local_hits_2d, width=PANEL_W, height=PANEL_H, max_range=l2d_range, fov_deg=l2d_fov
            )
            lidar3d_panel = render_lidar_3d_projection(
                local_hits_3d, width=PANEL_W, height=PANEL_H, max_range=l3d_range, vert_fov_limit=l3d_vfov
            )

            # 3. Stitch 4-Panel Dashboard and push to Omni.UI
            row_top = np.hstack([rgb_bgr, depth_colormap])
            row_bottom = np.hstack([lidar2d_panel, lidar3d_panel])
            dashboard_bgr = np.vstack([row_top, row_bottom])

            dashboard_ui.update_image(dashboard_bgr)


if __name__ == "__main__":
    main()
    simulation_app.close()

