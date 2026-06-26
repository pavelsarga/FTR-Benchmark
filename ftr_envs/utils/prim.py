import os
from itertools import chain

import omni.usd
from omni.isaac.core.utils.prims import (
    create_prim,
    find_matching_prim_paths,
    get_prim_at_path,
)
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdPhysics, UsdShade


def set_material_color(path, color):
    prim = get_prim_at_path(f"{path}/Shader")
    prim.GetAttribute("inputs:diffuse_tint").Set(color)


def get_prim_radius(path):
    prim = get_prim_at_path(path)
    return prim.GetAttribute("radius").Get()


def set_render_radius(path, radius):
    prim = get_prim_at_path(path)
    prim.GetAttribute("radius").Set(radius)


def set_prim_invisible(path):
    prim = get_prim_at_path(path)
    prim.GetAttribute("visibility").Set("invisible")


def set_material_friction(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("physics:dynamicFriction").Set(v)
    prim.GetAttribute("physics:staticFriction").Set(v)


def set_cylinder_collision_radius(prim_path, radius):
    """Set the radius of a UsdGeomCylinder collision shape directly.

    Independent of any kinematic radius used elsewhere (e.g. for wheel velocity-target
    scaling) — use this to test whether a wheel's *contact geometry* size, on its own,
    affects how often it snags on terrain mesh features, without changing how fast the
    wheel is commanded to roll.
    """
    from pxr import UsdGeom

    prim = get_prim_at_path(prim_path)
    cyl = UsdGeom.Cylinder(prim)
    if not cyl:
        return False
    cyl.GetRadiusAttr().Set(radius)
    return True


def disable_collision_geometry(prim_path):
    """Disable all PhysX collision shapes under prim_path.

    Walks the prim subtree and sets physics:collisionEnabled=False on every prim
    that carries a UsdPhysics.CollisionAPI. Does NOT affect sibling or cousin prims
    (e.g. child link collision shapes connected via joints are stored as siblings in
    the USD stage after URDF import, so they are unaffected).

    Returns the number of collision shapes disabled, or 0 if prim_path is invalid.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return 0
    disabled = 0
    for child in Usd.PrimRange(prim):
        api = UsdPhysics.CollisionAPI(child)
        if api:
            api.CreateCollisionEnabledAttr(False)
            disabled += 1
    return disabled


def set_collision_offsets(prim_path, contact_offset, rest_offset=0.0):
    """Set PhysX contact/rest offset on a collision prim.

    PhysX auto-derives default offsets from object size, so larger collision shapes
    get a larger speculative-contact margin than smaller ones. Call with an explicit
    small contact_offset to tighten that margin for an oversized/high-load shape.
    Must be called before the simulation starts (e.g. from set_robot_env).
    """
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.GetContactOffsetAttr().Set(contact_offset)
    api.GetRestOffsetAttr().Set(rest_offset)
    return True


def ensure_physics_material(material_path, friction, restitution=0.0):
    """Create (or update) a UsdPhysics material with the given friction at material_path.

    Unlike set_material_friction, this works on prims that don't already carry
    physics:*Friction attributes (e.g. assets imported without any PhysicsMaterial,
    such as the MARV URDF import) by applying UsdPhysics.MaterialAPI itself.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(material_path)
    material = UsdShade.Material(prim) if prim.IsValid() else UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(friction)
    material_api.CreateDynamicFrictionAttr().Set(friction)
    material_api.CreateRestitutionAttr().Set(restitution)
    return material


def bind_physics_material(prim_path, material):
    """Bind a UsdShade.Material (with UsdPhysics.MaterialAPI applied) to prim_path for physics."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )
    return True


def set_joint_stiffness(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("drive:angular:physics:stiffness").Set(v)


def set_joint_damping(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("drive:angular:physics:damping").Set(v)


def set_joint_max_vel(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("physxJoint:maxJointVelocity").Set(v)


def get_rotate_value(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute("xformOp:rotateXYZ").Get()


def get_translate_value(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute("xformOp:translate").Get()


def get_joint_pos_state(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute("state:angular:physics:position").Get()


def set_joint_pos_state(prim, pos):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    prim.GetAttribute("state:angular:physics:position").Set(pos)


def add_usd(usd_file, prim_path, pos=(0, 0, 0), orient=(1, 0, 0, 0), scale=(1, 1, 1)):
    orientation = orient if len(orient) == 4 else euler_angles_to_quat(orient, degrees=True)

    create_prim(
        prim_path=prim_path,
        prim_type="Xform",
        position=pos,
        orientation=orientation,
        scale=scale,
    )

    assert os.path.exists(usd_file) == True, f"{usd_file} file Not Found"
    return add_reference_to_stage(os.path.abspath(usd_file), prim_path)
