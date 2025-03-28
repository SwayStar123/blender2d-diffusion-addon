import sys
# ------------------------------------------------------------------
# 1) Adjust this path to wherever you installed fal_client:
packages_path = "C:\\Users\\ASUS\\AppData\\Roaming\\Python\\Python311\\site-packages"
sys.path.insert(0, packages_path)
# ------------------------------------------------------------------

bl_info = {
    "name": "SD ControlNet Plane Image (Upright)",
    "author": "Your Name",
    "version": (1, 9),
    "blender": (4, 3, 2),
    "location": "2D Animation/2D Full Canvas > Sidebar > ControlNet",
    "description": (
        "Captures the viewport or camera view, sends it to SD ControlNet via fal-client, "
        "and creates a vertically oriented plane with the returned image. "
        "Restores previous mode after execution."
    ),
    "category": "Object",
}

import bpy
import os
import tempfile
import requests

# Attempt to import fal_client
try:
    import fal_client
except ImportError:
    fal_client = None
    print("fal_client not found. Please ensure it's installed in the specified packages path.")


# -------------------------------------------------------------------
# ADD-ON PREFERENCES: Fal API Key
# -------------------------------------------------------------------
class SDControlNetAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    fal_api_key: bpy.props.StringProperty(
        name="Fal API Key",
        description="Enter your Fal API key (FAL_KEY) here",
        subtype='PASSWORD',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Set your Fal API key below:")
        layout.prop(self, "fal_api_key")


# -------------------------------------------------------------------
# HELPER: Capture the current view (camera view if available) to a .png
# -------------------------------------------------------------------
def capture_viewport():
    """
    Renders the current view to a temporary PNG file and returns its filepath.
    If a camera is available, it will render from the camera view (thus including 
    any previously generated image planes), otherwise it falls back to a viewport capture.
    """
    frame_number = bpy.context.scene.frame_current
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"viewport_capture_{frame_number}.png")

    old_filepath = bpy.context.scene.render.filepath
    try:
        bpy.context.scene.render.filepath = temp_path
        # Using OpenGL render capture for the current viewport.
        bpy.ops.render.opengl(write_still=True)
    finally:
        bpy.context.scene.render.filepath = old_filepath

    return temp_path


# -------------------------------------------------------------------
# HELPER: Create a plane aligned to the view, apply image as texture
# -------------------------------------------------------------------
def create_plane_with_image(img: bpy.types.Image, name="ControlNet Plane"):
    """
    If a camera is available, creates a plane that exactly fits the camera's view.
    Otherwise, it falls back to aligning the plane to the current 3D view.
    Then it creates a material using the given image and assigns it to the plane.
    Finally, it keyframes the plane's visibility so that it only appears on the current frame.
    """

    # Ensure we're in Object Mode.
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    cam = bpy.context.scene.camera
    if cam is not None:
        # Get the camera's view frame in camera local space.
        frame = cam.data.view_frame(scene=bpy.context.scene)
        bl, tl, tr, br = frame

        # Compute the center in camera space and convert to world space.
        center_local = (bl + tl + tr + br) / 4.0
        center_world = cam.matrix_world @ center_local

        # Create the plane at the computed center.
        bpy.ops.mesh.primitive_plane_add(size=1, location=center_world)
        plane_obj = bpy.context.active_object
        plane_obj.name = name

        # Align the plane's orientation with the camera's.
        plane_obj.rotation_euler = cam.rotation_euler

        # Compute the width and height of the view frame in camera space.
        width = (tr - tl).length
        height = (tl - bl).length

        # Set the scale so that the plane covers the view.
        plane_obj.scale = (width, height, 1)
    else:
        # Fallback: if no active camera, align the plane to the current 3D view.
        bpy.ops.mesh.primitive_plane_add(size=1, align='VIEW', location=(0, 0, 0))
        plane_obj = bpy.context.active_object
        plane_obj.name = name

        # Adjust plane scale using the image’s aspect ratio.
        base_size = 5
        img_width = img.size[0]
        img_height = img.size[1]
        if img_width >= img_height:
            scale_x = base_size
            scale_y = base_size * (img_height / img_width)
        else:
            scale_x = base_size * (img_width / img_height)
            scale_y = base_size
        plane_obj.scale = (scale_x, scale_y, 1)

    # Create a new material that uses the image as a texture.
    mat = bpy.data.materials.new(name="ControlNetImageMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Remove default nodes.
    for n in nodes:
        nodes.remove(n)

    # Set up nodes.
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    diffuse_node = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse_node.location = (0, 0)

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.location = (-300, 0)
    tex_node.image = img

    # Connect nodes: Image Texture → Diffuse BSDF → Material Output.
    links.new(tex_node.outputs["Color"], diffuse_node.inputs["Color"])
    links.new(diffuse_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign the material to the plane.
    plane_obj.data.materials.append(mat)

    # ----------------------------------------------------------------
    # Set keyframes so that the plane is visible only on the current frame.
    # We add a hidden keyframe before and after the current frame.
    # ----------------------------------------------------------------
    current_frame = bpy.context.scene.frame_current

    # Ensure the plane is hidden for frames before the current frame.
    plane_obj.hide_viewport = True
    plane_obj.hide_render = True
    plane_obj.keyframe_insert(data_path='hide_viewport', frame=current_frame - 1)
    plane_obj.keyframe_insert(data_path='hide_render', frame=current_frame - 1)

    # Make it visible on the current frame.
    plane_obj.hide_viewport = False
    plane_obj.hide_render = False
    plane_obj.keyframe_insert(data_path='hide_viewport', frame=current_frame)
    plane_obj.keyframe_insert(data_path='hide_render', frame=current_frame)

    # Hide it again on the next frame.
    plane_obj.hide_viewport = True
    plane_obj.hide_render = True
    plane_obj.keyframe_insert(data_path='hide_viewport', frame=current_frame + 1)
    plane_obj.keyframe_insert(data_path='hide_render', frame=current_frame + 1)

    # Optional: Set the interpolation of these keyframes to CONSTANT
    # to avoid any interpolation between keyframes.
    if plane_obj.animation_data and plane_obj.animation_data.action:
        for fcurve in plane_obj.animation_data.action.fcurves:
            if fcurve.data_path in ['hide_viewport', 'hide_render']:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'CONSTANT'

    return plane_obj


# -------------------------------------------------------------------
# OPERATOR: Capture → ControlNet → Plane with texture, then restore previous mode
# -------------------------------------------------------------------
class OBJECT_OT_run_controlnet_2d(bpy.types.Operator):
    bl_idname = "object.run_controlnet_2d"
    bl_label = "Run SD ControlNet"
    bl_description = (
        "Capture the current view (camera view if available), send it to SD ControlNet via fal-client, "
        "and create an upright plane textured with the returned image. "
        "Supports processing a single frame (current frame) or a range of frames."
    )
    bl_options = {'REGISTER', 'UNDO'}

    # Property for the prompt
    user_prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Enter prompt for SD ControlNet",
        default="Ice fortress, aurora skies, polar wildlife, twilight",
    )

    # New properties for frame selection:
    frame_mode: bpy.props.EnumProperty(
        items=[
            ("CURRENT", "Current Frame", "Use the current frame"),
            ("SPECIFY", "Specify Frames", "Specify a start and end frame"),
        ],
        default="CURRENT",
        name="Frame Mode",
        description="Choose whether to process only the current frame or a range of frames."
    )
    frame_start: bpy.props.IntProperty(
        name="Start Frame",
        description="Starting frame number",
        default=1,
    )
    frame_end: bpy.props.IntProperty(
        name="End Frame",
        description="Ending frame number",
        default=1,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "user_prompt")
        layout.prop(self, "frame_mode")
        if self.frame_mode == 'SPECIFY':
            layout.prop(self, "frame_start")
            layout.prop(self, "frame_end")

    def execute(self, context):
        # --- Store the previous active object, mode, and current frame ---
        previous_obj = context.active_object
        previous_mode = previous_obj.mode if previous_obj else None
        original_frame = context.scene.frame_current

        # Force Object Mode if necessary for processing
        if previous_obj and previous_obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        if fal_client is None:
            self.report({'ERROR'}, "fal_client not found. Please install it in the specified path.")
            return {'CANCELLED'}

        # --- Grab Fal API key ---
        prefs = context.preferences.addons[__name__].preferences
        api_key = prefs.fal_api_key.strip()
        if not api_key:
            self.report({'ERROR'}, "Fal API key is empty. Set it in the add-on preferences.")
            return {'CANCELLED'}

        os.environ["FAL_KEY"] = api_key

        # --- Determine which frames to process ---
        if self.frame_mode == "CURRENT":
            frames = [original_frame]
        else:
            if self.frame_start > self.frame_end:
                self.report({'ERROR'}, "Start frame must be less than or equal to end frame.")
                return {'CANCELLED'}
            frames = list(range(self.frame_start, self.frame_end + 1))

        # --- Loop over each frame ---
        for frame in frames:
            context.scene.frame_set(frame)
            self.report({'INFO'}, f"Processing frame {frame}...")

            # 1) Capture the current view (camera view if available)
            try:
                capture_path = capture_viewport()
            except Exception as e:
                self.report({'ERROR'}, f"Failed to capture view at frame {frame}: {e}")
                return {'CANCELLED'}

            # 2) Upload the captured image
            try:
                uploaded_url = fal_client.upload_file(capture_path)
            except Exception as e:
                self.report({'ERROR'}, f"File upload failed at frame {frame}: {e}")
                return {'CANCELLED'}

            # 3) Call ControlNet using the user-specified prompt
            arguments = {
                "prompt": self.user_prompt,
                "control_image_url": uploaded_url,
            }

            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        print(log.get("message", ""))

            try:
                result = fal_client.subscribe(
                    "fal-ai/sd15-depth-controlnet",
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )
            except Exception as e:
                self.report({'ERROR'}, f"API call failed at frame {frame}: {e}")
                return {'CANCELLED'}

            try:
                output_url = result["images"][0]["url"]
            except Exception:
                self.report({'ERROR'}, f"Failed to parse output image URL at frame {frame}.")
                return {'CANCELLED'}

            # 4) Download the resulting image
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"controlnet_output_{frame}.png")

            try:
                resp = requests.get(output_url)
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to download output image at frame {frame}: {e}")
                return {'CANCELLED'}

            # 5) Load the image into Blender
            try:
                processed_image = bpy.data.images.load(output_path)
            except Exception as e:
                self.report({'ERROR'}, f"Unable to load the output image at frame {frame}: {e}")
                return {'CANCELLED'}

            # 6) Create an upright plane with the returned image as texture
            try:
                create_plane_with_image(processed_image, name=f"ControlNet Plane (Frame {frame})")
            except Exception as e:
                self.report({'ERROR'}, f"Failed to create plane at frame {frame}: {e}")
                return {'CANCELLED'}

        self.report({'INFO'}, f"SD ControlNet result applied to {len(frames)} frame(s).")

        # --- Restore the original frame ---
        context.scene.frame_set(original_frame)

        # --- Restore the previous active object and mode if possible ---
        if previous_obj and previous_obj.name in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            previous_obj.select_set(True)
            context.view_layer.objects.active = previous_obj
            if previous_mode and previous_mode != 'OBJECT':
                try:
                    bpy.ops.object.mode_set(mode=previous_mode)
                except Exception as e:
                    print("Failed to restore previous mode:", e)

        return {'FINISHED'}


# -------------------------------------------------------------------
# PANEL: UI in the 3D View sidebar
# -------------------------------------------------------------------
class VIEW2D_PT_controlnet_panel(bpy.types.Panel):
    bl_label = "SD ControlNet"
    bl_idname = "VIEW2D_PT_controlnet_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ControlNet"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.run_controlnet_2d", icon='IMAGE_DATA')


classes = (
    SDControlNetAddonPreferences,
    OBJECT_OT_run_controlnet_2d,
    VIEW2D_PT_controlnet_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    print("SD ControlNet Plane Image (Upright) Add-on registered.")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("SD ControlNet Plane Image (Upright) Add-on unregistered.")

if __name__ == "__main__":
    register()
