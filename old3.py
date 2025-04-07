# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import bpy
import os
import tempfile
import json
import urllib.request
import urllib.parse
import uuid
import time
import subprocess  # To run ffmpeg
import threading   # For modal operator background task
import queue       # For thread communication
import mathutils   # For Vector math (less needed now)
import math        # Less needed now
import shutil      # For robust directory removal

# ------------------------------------------------------------------
# 1) Adjust this path if necessary for websocket-client or other packages:
packages_path = "C:\\Users\\ASUS\\AppData\\Roaming\\Python\\Python311\\site-packages"
if packages_path not in sys.path:
    sys.path.insert(0, packages_path)

# Try importing websocket-client
try:
    import websocket # Need to install: pip install websocket-client
except ImportError:
    print(
        "ERROR: Could not import 'websocket'. Ensure 'websocket-client' is installed "
        f"in the specified packages_path ('{packages_path}') or Blender's Python environment."
    )
    websocket = None # Set to None to handle checks later

# --- Try importing SPA Studios Grease Pencil Core ---
try:
    # Assuming the SPA addon is installed and follows this structure
    from spa_anim2D.gpencil_references import core as spa_gp_core
    # Check if the necessary function exists as a sanity check
    if not hasattr(spa_gp_core, 'import_image_as_gp_reference'):
         raise AttributeError("SPA GP Core module loaded, but 'import_image_as_gp_reference' not found.")
    SPA_GP_AVAILABLE = True
    print("SPA Studios Grease Pencil reference core module loaded successfully.")
except (ImportError, AttributeError) as e:
    print(
        "ERROR: Could not import SPA Studios Grease Pencil reference core module ('spa_anim2D.gpencil_references.core'). "
        "Ensure the SPA Studios Blender fork/addon is correctly installed and enabled. "
        f"Details: {e}"
    )
    spa_gp_core = None
    SPA_GP_AVAILABLE = False
# ------------------------------------------------------------------

bl_info = {
    "name": "ComfyUI GPencil Integration", # Renamed
    "author": "Your Name (Modified for ComfyUI Modal + GPencil)",
    "version": (2, 3), # Incremented version
    "blender": (3, 3, 0),
    "location": "View3D > Sidebar > ComfyUI",
    "description": (
        "Captures frames, sends to ComfyUI, and adds results as Grease Pencil references fitted to camera view. "
        "Requires an ACTIVE Grease Pencil object, the SPA Studios Blender fork/addon, ffmpeg in PATH, and websocket-client."
    ),
    "category": "Object",
    "warning": "Requires ACTIVE GP OBJECT, SPA Studios fork/addon, ffmpeg in PATH, and 'websocket-client'.", # Updated warning
    "doc_url": "",
    "tracker_url": "",
}

# -------------------------------------------------------------------
# ComfyUI Workflow JSON (Embed the specific workflow here)
# -------------------------------------------------------------------
# MAKE SURE this JSON is valid and correctly represents your desired workflow.
# Node IDs are crucial. Default IDs used:
# - Video Input: "107" (VHS_LoadVideoPath)
# - Prompt Input: "3" (CLIPTextEncode)
# - Image Output: "12" (SaveImage)
# Change the operator's `video_path_node_id`, `prompt_node_id`, `output_node_id`
# variables below if your workflow uses different IDs.
COMFYUI_WORKFLOW_JSON = """
{
  "2": {
    "inputs": {
      "vae_name": "sd1.5vae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "3": {
    "inputs": {
      "text": "1girl, hakurei reimu, dancing. Plain white background",
      "clip": [
        "124",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "6": {
    "inputs": {
      "text": "ugly, deformed, bad lighting, blurry, text, watermark, extra hands, bad quality, deformed hands, deformed fingers, nostalgic, drawing, painting, bad anatomy, worst quality, blurry, blurred, normal quality, bad focus",
      "clip": [
        "124",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "seed": 665058481726322,
      "steps": 25,
      "cfg": 7,
      "sampler_name": "euler_ancestral",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "111",
        0
      ],
      "positive": [
        "114",
        0
      ],
      "negative": [
        "114",
        1
      ],
      "latent_image": [
        "119",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "10": {
    "inputs": {
      "samples": [
        "7",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "12": {
    "inputs": {
      "filename_prefix": "BlenderComfyOutput",
      "images": [
        "10",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "50": {
    "inputs": {
      "images": [
        "107",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "70": {
    "inputs": {
      "control_net_name": "control_v11f1p_sd15_depth.pth"
    },
    "class_type": "ControlNetLoaderAdvanced",
    "_meta": {
      "title": "Load Advanced ControlNet Model 🛂🅐🅒🅝"
    }
  },
  "92": {
    "inputs": {
      "images": [
        "121",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "107": {
    "inputs": {
      "video": "PLACEHOLDER_VIDEO_PATH",
      "force_rate": 0,
      "custom_width": 0,
      "custom_height": 0,
      "frame_load_cap": 0,
      "skip_first_frames": 0,
      "select_every_nth": 1,
      "format": "AnimateDiff"
    },
    "class_type": "VHS_LoadVideoPath",
    "_meta": {
      "title": "Load Video (Path) 🎥🅥🅗🅢"
    }
  },
  "109": {
    "inputs": {
      "frame_rate": 8,
      "loop_count": 0,
      "filename_prefix": "AnimateDiff",
      "format": "image/gif",
      "pingpong": false,
      "save_output": true,
      "images": [
        "10",
        0
      ]
    },
    "class_type": "VHS_VideoCombine",
    "_meta": {
      "title": "Video Combine 🎥🅥🅗🅢"
    }
  },
  "110": {
    "inputs": {
      "ckpt_name": "counterfeitV30_v30.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "111": {
    "inputs": {
      "beta_schedule": "autoselect",
      "model": [
        "124",
        0
      ],
      "m_models": [
        "115",
        0
      ],
      "context_options": [
        "112",
        0
      ],
      "sample_settings": [
        "113",
        0
      ]
    },
    "class_type": "ADE_UseEvolvedSampling",
    "_meta": {
      "title": "Use Evolved Sampling 🎭🅐🅓②"
    }
  },
  "112": {
    "inputs": {
      "context_length": 16,
      "context_stride": 1,
      "context_overlap": 4,
      "fuse_method": "pyramid",
      "use_on_equal_length": false,
      "start_percent": 0,
      "guarantee_steps": 1
    },
    "class_type": "ADE_StandardUniformContextOptions",
    "_meta": {
      "title": "Context Options◆Standard Uniform 🎭🅐🅓"
    }
  },
  "113": {
    "inputs": {
      "batch_offset": 0,
      "noise_type": "FreeNoise",
      "seed_gen": "comfy",
      "seed_offset": 0,
      "adapt_denoise_steps": false
    },
    "class_type": "ADE_AnimateDiffSamplingSettings",
    "_meta": {
      "title": "Sample Settings 🎭🅐🅓"
    }
  },
  "114": {
    "inputs": {
      "strength": 0.5,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "3",
        0
      ],
      "negative": [
        "6",
        0
      ],
      "control_net": [
        "70",
        0
      ],
      "image": [
        "121",
        0
      ]
    },
    "class_type": "ACN_AdvancedControlNetApply",
    "_meta": {
      "title": "Apply Advanced ControlNet 🛂🅐🅒🅝"
    }
  },
  "115": {
    "inputs": {
      "motion_model": [
        "116",
        0
      ]
    },
    "class_type": "ADE_ApplyAnimateDiffModelSimple",
    "_meta": {
      "title": "Apply AnimateDiff Model 🎭🅐🅓②"
    }
  },
  "116": {
    "inputs": {
      "model_name": "temporaldiff-v1-animatediff.safetensors"
    },
    "class_type": "ADE_LoadAnimateDiffModel",
    "_meta": {
      "title": "Load AnimateDiff Model 🎭🅐🅓②"
    }
  },
  "119": {
    "inputs": {
      "width": 1024,
      "height": 576,
      "batch_size": [
        "107",
        1
      ]
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "120": {
    "inputs": {
      "image": [
        "107",
        0
      ]
    },
    "class_type": "ImageInvert",
    "_meta": {
      "title": "Invert Image"
    }
  },
  "121": {
    "inputs": {
      "condition": "A is TRUE",
      "require_inputs": true,
      "NOT": false,
      "custom_expression": "",
      "A": [
        "123",
        0
      ],
      "TRUE_IN": [
        "120",
        0
      ],
      "FALSE_IN": [
        "107",
        0
      ]
    },
    "class_type": "IfConditionSelector",
    "_meta": {
      "title": "🔀 IF (Condition Selector)"
    }
  },
  "123": {
    "inputs": {
      "value": true
    },
    "class_type": "PrimitiveBoolean",
    "_meta": {
      "title": "Invert input for depth"
    }
  },
  "124": {
    "inputs": {
      "lora_name": "reimu.safetensors",
      "strength_model": 0.7000000000000002,
      "strength_clip": 0.7000000000000002,
      "model": [
        "110",
        0
      ],
      "clip": [
        "110",
        1
      ]
    },
    "class_type": "LoraLoader",
    "_meta": {
      "title": "Load LoRA"
    }
  }
}
"""


# -------------------------------------------------------------------
# ADD-ON PREFERENCES
# -------------------------------------------------------------------
class ComfyUIAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    comfyui_address: bpy.props.StringProperty(
        name="ComfyUI Server Address",
        description="Address of your running ComfyUI server (e.g., http://127.0.0.1:8188)",
        default="http://127.0.0.1:8188",
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="ComfyUI Server Settings:")
        layout.prop(self, "comfyui_address")
        layout.separator()
        box = layout.box()
        box.label(text="Dependencies & Notes:", icon="INFO")
        box.label(text="- Requires an ACTIVE Grease Pencil object selected.") # New requirement
        box.label(text="- Requires SPA Studios Blender fork/addon features.") # New requirement
        box.label(text="- Requires 'ffmpeg' installed and in system PATH.")
        box.label(text="- Requires 'websocket-client' Python package.")
        box.label(text=f"  (Attempting websocket load from: {packages_path})")
        box.label(
            text="- Frame capture uses the scene's render settings (if in camera view) or OpenGL."
        )


# -------------------------------------------------------------------
# HELPER: Capture the current view
# -------------------------------------------------------------------
def capture_viewport(output_path, context):
    """
    Renders the current view to the specified PNG file path.
    Uses camera render if available and active, otherwise OpenGL viewport render.
    """
    # Store current settings
    scene = context.scene
    render = scene.render
    original_filepath = render.filepath

    # Find 3D view area and space in the *current context's window/screen*
    area = next((a for a in context.screen.areas if a.type == "VIEW_3D"), None)
    space = (
        next((s for s in area.spaces if s.type == "VIEW_3D"), None) if area else None
    )

    # Check if space exists, needed for camera view check and OpenGL fallback
    if not space:
        print(
            "Warning: Could not find active 3D Viewport space. Capture might fail or be incorrect."
        )
        # Optionally raise an error here if a 3D view is strictly required
        # raise RuntimeError("Active 3D Viewport required for capture.")

    try:
        render.filepath = output_path
        render.image_settings.file_format = "PNG" # Ensure PNG format

        use_opengl = True # Default to faster OpenGL
        if scene.camera:
            # Check if we are actually in camera view in the *active* 3D viewport
            is_camera_view = (
                space
                and space.region_3d
                and space.region_3d.view_perspective == "CAMERA"
            )

            if is_camera_view:
                print("Capturing using Render Engine (might be slow).")
                # Ensure correct render settings (resolution etc. are used from scene)
                bpy.ops.render.render(write_still=True)
                use_opengl = False # Render command was used
            else:
                print(
                    "Camera exists, but not in camera view in active viewport. Using OpenGL viewport capture."
                )
        else:
            print("No scene camera found, using OpenGL viewport capture.")

        if use_opengl:
            if space:
                # Use opengl render which respects viewport settings
                # Need to pass the context correctly for 'view_context'
                print("Capturing using OpenGL render.")
                # Find appropriate region (usually the last one is the main view)
                render_region = None
                for region in area.regions:
                    if region.type == 'WINDOW':
                         render_region = region
                         break
                if not render_region:
                    print("ERROR: Could not find WINDOW region in 3D Viewport area for OpenGL render.")
                    raise RuntimeError("Cannot perform OpenGL capture without a valid WINDOW region.")

                with context.temp_override(area=area, region=render_region):
                    bpy.ops.render.opengl(write_still=True, view_context=True)
            else:
                print(
                    "Error: Could not find 3D Viewport space for OpenGL render. Cannot capture."
                )
                raise RuntimeError(
                    "Cannot perform OpenGL capture without a valid 3D Viewport space."
                )

    except Exception as e:
        print(f"Error during viewport capture: {e}")
        raise # Re-raise the exception to be caught by the operator
    finally:
        # Restore original settings
        render.filepath = original_filepath
        # render.engine = original_engine # Restore if changed

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Output image file not found after capture attempt: {output_path}"
        )
    elif os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"Output image file is empty after capture attempt: {output_path}"
        )

    return output_path


# -------------------------------------------------------------------
# HELPER: Create video from sequence of frames 
# -------------------------------------------------------------------
def create_video_from_frames(
    frame_dir,
    output_video_path,
    frame_rate=24,
    frame_pattern="frame_%04d.png",
    start_number=None,
):
    """
    Uses ffmpeg to create a video from a sequence of PNG frames.
    Requires ffmpeg to be in the system PATH.
    frame_pattern example: 'frame_%04d.png'
    start_number: Optional first frame number if pattern needs it explicitly.
    """
    input_pattern = os.path.join(frame_dir, frame_pattern)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-y", # Overwrite output without asking
        "-framerate",
        str(frame_rate),
    ]
    # If frame numbers don't start at 0 or 1, ffmpeg might need -start_number
    if start_number is not None:
        command.extend(["-start_number", str(start_number)])

    command.extend(
        [
            "-i",
            input_pattern,
            "-c:v",
            "libx264", # Video codec
            "-crf",
            "18", # Quality (lower is better, 18=visually lossless)
            "-preset",
            "fast", # Encoding speed vs compression (faster is usually fine for temp)
            "-pix_fmt",
            "yuv420p", # Pixel format for compatibility
            output_video_path,
        ]
    )

    print(f"Running ffmpeg command: {' '.join(command)}")
    try:
        startupinfo = None
        if os.name == "nt": # Windows specific: hide console
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            startupinfo=startupinfo,
            encoding="utf-8",
        )
        # print("ffmpeg stdout:", result.stdout) # Can be noisy
        if result.stderr:
            print("ffmpeg stderr:", result.stderr) # Print stderr for warnings/info
        print(f"Video created successfully: {output_video_path}")
        return output_video_path
    except FileNotFoundError:
        print("\nERROR: ffmpeg command not found.")
        print(
            "Make sure ffmpeg is installed and its directory is included in your system's PATH environment variable."
        )
        print("You can download ffmpeg from: https://ffmpeg.org/download.html\n")
        raise
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed with exit code {e.returncode}")
        print("ffmpeg stdout:", e.stdout)
        print("ffmpeg stderr:", e.stderr)
        print("\nCheck if the input frame pattern is correct and frames exist.")
        print(f"Input pattern used: {input_pattern}")
        print(f"Start number used: {start_number}")
        print(f"Frame directory: {frame_dir}\n")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during video creation: {e}")
        raise


# -------------------------------------------------------------------
# HELPER: ComfyUI API Interaction
# - get_comfyui_results_from_history
# - queue_comfyui_prompt
# - get_comfyui_image_data
# - fetch_images_from_ws_or_history_data
# - get_comfyui_images_ws
# (These functions interact with the ComfyUI server API and are independent
# of how the results are used in Blender, so they remain the same.)
# -------------------------------------------------------------------
def get_comfyui_results_from_history(prompt_id, server_address, target_node_id):
    """
    Fetches prompt history via HTTP and extracts image data for a specific node.
    Returns a list of image byte data, or None if history not found/complete
    or node output missing. Returns [] if node found but has no images.
    """
    images_data = None # Default to None (history fetch failed or not ready)
    try:
        url = f"{server_address}/history/{prompt_id}"
        # print(f"Fetching history from: {url}") # DEBUG
        with urllib.request.urlopen(
            url, timeout=10
        ) as response: # Add timeout to HTTP request
            if response.status == 200:
                history = json.loads(response.read())
                # History is a dict {prompt_id: {outputs: {node_id: {images: [...]}}}}
                prompt_data = history.get(prompt_id)
                if not prompt_data:
                    print(f"History found, but data for prompt {prompt_id} is missing.")
                    return None # History exists but is incomplete?

                outputs = prompt_data.get("outputs")
                if not outputs:
                    # This is expected if the prompt hasn't finished generating outputs
                    # print(f"History for prompt {prompt_id} has no 'outputs' section (likely still running/processing).") # DEBUG - Can be noisy
                    return None # Not finished processing outputs

                node_output = outputs.get(
                    str(target_node_id)
                ) # Node IDs in history keys are strings
                if (
                    node_output is None
                ): # Check specifically for None, as empty dict is possible
                    print(
                        f"History outputs for prompt {prompt_id} do not contain node ID {target_node_id}."
                    )
                    # Log available nodes for debugging
                    print(f"Available output nodes in history: {list(outputs.keys())}")
                    return (
                        []
                    ) # Return empty list indicating node wasn't found in output

                images_info = node_output.get("images")
                if images_info is None: # Check specifically for None
                    print(
                        f"Node {target_node_id} output found in history but has no 'images' key."
                    )
                    return [] # Node output exists but no images key
                if not images_info:
                    print(
                        f"Node {target_node_id} output found in history but the 'images' list is empty."
                    )
                    return [] # Node output exists but images list is empty

                print(
                    f"Found {len(images_info)} images for node {target_node_id} in history."
                )
                images_data = [] # Initialize list now that we expect images
                fetch_errors = 0
                fetch_start_time = time.time()
                for i, image_info in enumerate(images_info):
                    filename = image_info.get("filename")
                    subfolder = image_info.get(
                        "subfolder", ""
                    ) # Default to empty string if missing
                    img_type = image_info.get("type")
                    if filename and img_type:
                        # print(f"  Fetching image {i+1}/{len(images_info)} from history ref: {filename}") # DEBUG
                        # Use the existing helper to fetch actual image data
                        img_data = get_comfyui_image_data(
                            filename, subfolder, img_type, server_address
                        )
                        if img_data:
                            images_data.append(img_data)
                        else:
                            print(
                                f"  ERROR: Failed to retrieve image data for {filename} from /view API (referenced in history)."
                            )
                            fetch_errors += 1
                    else:
                        print(
                            f"  Warning: Incomplete image info in history output: {image_info}"
                        )
                        fetch_errors += 1

                # print(f"  History image fetching took {time.time()-fetch_start_time:.2f}s.") # DEBUG
                if fetch_errors > 0:
                    print(
                        f"Warning: Failed to fetch {fetch_errors} image(s) referenced in history."
                    )
                # Return partial list if some succeeded, or empty if all failed

                return (
                    images_data # Return list (might be empty if fetch failed for all)
                )

            elif response.status == 404:
                # Expected if prompt ID doesn't exist or hasn't finished writing history yet
                # print(f"History API returned 404 Not Found for prompt {prompt_id}. Prompt likely hasn't finished or ID is wrong.") # DEBUG - Can be noisy
                return None
            else:
                print(
                    f"Error fetching history for prompt {prompt_id}: HTTP Status {response.status}"
                )
                return None # Indicate history fetch failed

    except urllib.error.URLError as e:
        # Can happen if server down or history endpoint changes
        print(f"URL Error fetching history {prompt_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error processing history for {prompt_id}: {e}")
        return None
    except TimeoutError:
        print(f"Timeout Error fetching history {prompt_id}.")
        return None # History check timed out
    except Exception as e:
        print(f"Unexpected error fetching/processing history for {prompt_id}: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Should not be reached if logic is correct, but acts as a final failure case
    return images_data

def queue_comfyui_prompt(prompt_workflow, server_address, client_id):
    """Sends the workflow to the ComfyUI queue."""
    p = {"prompt": prompt_workflow, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    url = f"{server_address}/prompt"
    print(f"Queueing prompt to {url}")
    req = urllib.request.Request(url, data=data)
    try:
        response = urllib.request.urlopen(
            req, timeout=20
        ) # Increased timeout for queueing
        return json.loads(response.read())
    except urllib.error.URLError as e:
        print(f"Error queueing prompt: {e}")
        error_message = f"Error queueing prompt: {e}"
        if hasattr(e, "reason"):
            error_message += f" Reason: {e.reason}"
        if hasattr(e, "read"):
            try:
                error_message += f" Server Response: {e.read().decode()}"
            except Exception as decode_e:
                error_message += f" Server Response (decode error): {decode_e}"
        print(error_message)
        raise ConnectionError(error_message) from e # Raise a more specific error
    except json.JSONDecodeError as e:
        print(f"Error decoding queue response: {e}")
        raise

def get_comfyui_image_data(filename, subfolder, image_type, server_address):
    """Fetches image data from ComfyUI's /view endpoint."""
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    url = f"{server_address}/view?{url_values}"
    # print(f"Fetching image from: {url}") # DEBUG
    try:
        with urllib.request.urlopen(
            url, timeout=30
        ) as response: # Increased timeout for image fetch
            if response.status == 200:
                return response.read()
            else:
                print(f"Error fetching image {filename}: Status code {response.status}")
                return None
    except urllib.error.URLError as e:
        print(f"Error fetching image {filename}: {e}")
        if hasattr(e, "read"):
            try:
                print(f"Server response: {e.read().decode()}")
            except:
                pass
        return None # Indicate failure
    except TimeoutError:
        print(f"Timeout error fetching image {filename}.")
        return None
    except Exception as e:
        print(f"Unexpected error fetching image {filename}: {e}")
        return None

def fetch_images_from_ws_or_history_data(
    data_source,
    source_type,
    prompt_id,
    server_address,
    target_node_id,
    progress_callback=None,
):
    """
    Helper function to fetch image data given either WebSocket 'executed' data
    or History API data for the target node.
    """
    images_data = []
    fetch_errors = 0
    start_time = time.time()

    if source_type == "ws":
        node_outputs = data_source.get("outputs", {})
        images_info = node_outputs.get("images")
        source_name = "WebSocket message"
    elif source_type == "history":
        images_info = data_source # History function already returns the images list
        source_name = "History API"
    else:
        print("Error: Invalid source_type for image fetching.")
        return []

    if not images_info:
        print(f"No 'images' key or empty list found in data from {source_name}.")
        return []

    print(f"Found {len(images_info)} image refs in {source_name}. Fetching data...")

    for i, image_info in enumerate(images_info):
        filename = image_info.get("filename")
        subfolder = image_info.get("subfolder", "")
        img_type = image_info.get("type")

        if filename and img_type:
            if progress_callback:
                progress_callback(f"Fetching image {i+1}/{len(images_info)}")
            # print(f"  Retrieving image via {source_name} ref: {filename}") # DEBUG
            img_data = get_comfyui_image_data(
                filename, subfolder, img_type, server_address
            )
            if img_data:
                images_data.append(img_data)
            else:
                print(
                    f"  ERROR: Failed to retrieve image data for {filename} (from {source_name})."
                )
                fetch_errors += 1
        else:
            print(
                f"  Warning: Incomplete image info in {source_name} data: {image_info}"
            )
            fetch_errors += 1

    print(f"  Image fetching from {source_name} took {time.time() - start_time:.2f}s.")
    if fetch_errors > 0:
        print(
            f"Warning: Failed to fetch {fetch_errors} image(s) referenced in {source_name} data."
        )

    return images_data

def get_comfyui_images_ws(
    prompt_id, server_address, client_id, output_node_id="12", progress_callback=None
):
    """
    Connects via WebSocket, waits for results from a specific node.
    Includes timeout, progress reporting, and smarter fallback to /history API.
    Raises exceptions on critical errors (connection, timeout without completion).
    """
    if not websocket:
        raise RuntimeError("websocket-client package not loaded.")

    ws_url = f"ws://{server_address.split('//')[1]}/ws?clientId={client_id}"
    print(f"Connecting to WebSocket: {ws_url}")
    ws = websocket.WebSocket()

    try:
        ws.connect(ws_url, timeout=20) # Increased connection timeout
        print("WebSocket connected.")
    except websocket.WebSocketTimeoutException:
        print(
            f"WebSocket connection timed out ({ws_url}). Check if ComfyUI server is running and accessible."
        )
        raise ConnectionError(f"WebSocket connection timed out: {ws_url}") from None
    except ConnectionRefusedError:
        print(
            f"WebSocket connection refused ({ws_url}). Check if ComfyUI server is running and the address/port are correct."
        )
        raise ConnectionError(f"WebSocket connection refused: {ws_url}") from None
    except websocket.WebSocketException as e:
        print(f"WebSocket connection failed: {e}")
        raise ConnectionError(f"WebSocket connection failed: {e}") from e
    except Exception as e:
        print(f"WebSocket connection failed unexpectedly: {e}")
        raise ConnectionError(f"WebSocket connection failed unexpectedly: {e}") from e

    images_data = None # Use None to indicate not yet retrieved
    prompt_execution_finished_signal = (
        False # Flag for the overall prompt completion signal (node=None)
    )
    output_node_executed_signal = (
        False # Flag if specific output node finished message received
    )
    retrieved_via_history = False # Flag to track if history fallback was used

    consecutive_timeouts = 0
    max_consecutive_timeouts_before_warn = (
        5 # How many timeouts before checking history proactively
    )
    max_consecutive_timeouts_overall = (
        20 # Give up after this many timeouts without progress
    )
    overall_timeout_seconds = 900 # Overall safety timeout (15 minutes)
    ws_receive_timeout = 15 # How long to wait for a single message

    start_time = time.time()
    last_message_time = start_time

    try:
        while images_data is None and (
            time.time() - start_time < overall_timeout_seconds
        ):
            try:
                ws.settimeout(ws_receive_timeout)
                out = ws.recv()
                consecutive_timeouts = 0 # Reset timeout counter on successful receive
                last_message_time = time.time()
                # print(f"WS Recv: {out[:100]}...") # DEBUG Very verbose

            except websocket.WebSocketTimeoutException:
                consecutive_timeouts += 1
                # print(f"WebSocket receive timed out ({consecutive_timeouts}/{max_consecutive_timeouts_overall}). Checking state...") # DEBUG

                if consecutive_timeouts >= max_consecutive_timeouts_overall:
                    print(
                        f"Exceeded max consecutive WebSocket timeouts ({max_consecutive_timeouts_overall})."
                    )
                    # --- Try history one last time before raising timeout ---
                    history_result = get_comfyui_results_from_history(
                        prompt_id, server_address, output_node_id
                    )
                    if history_result: # Not None or empty []
                        print("Final history check after max timeouts succeeded.")
                        images_data = history_result
                        retrieved_via_history = True
                        break # Exit loop
                    else:
                        raise TimeoutError(
                            f"WebSocket stopped receiving messages for {ws_receive_timeout * max_consecutive_timeouts_overall} seconds. Final history check failed."
                        )

                # --- History Check Logic ---
                should_check_history = False
                if prompt_execution_finished_signal and not output_node_executed_signal:
                    print(
                        "Prompt finished signal received, but output node message not seen yet. Checking history..."
                    )
                    should_check_history = True
                elif consecutive_timeouts >= max_consecutive_timeouts_before_warn:
                    # print(f"Reached {consecutive_timeouts} timeouts. Proactively checking history...") # DEBUG
                    should_check_history = True

                if should_check_history:
                    history_result = get_comfyui_results_from_history(
                        prompt_id, server_address, output_node_id
                    )
                    if (
                        history_result is not None
                    ): # History fetch succeeded (result might be [] or list of bytes)
                        if history_result: # Found images (list of bytes) in history
                            print(
                                f"Received {len(history_result)} images from /history API fallback."
                            )
                            images_data = history_result
                            retrieved_via_history = True
                            break # Exit WebSocket loop, we have results
                        else:
                            # History exists, but node/images not found or empty list returned
                            if prompt_execution_finished_signal:
                                print(
                                    f"/history API confirms prompt finished but node {output_node_id} has no image output."
                                )
                                images_data = [] # Mark as finished with no images
                                break
                            # else: # Prompt not finished according to history, or node not listed yet
                                # print("History doesn't have the final output images yet. Continuing WS listen.") # DEBUG
                                # pass
                    # else: # History fetch returned None (e.g. 404, error, timeout) - handled below by continuing loop

                # Keep connection alive if idle
                if time.time() - last_message_time > 60:
                    try:
                        # print("Sending WebSocket ping...") # DEBUG
                        ws.ping()
                        last_message_time = time.time()
                    except websocket.WebSocketConnectionClosedException:
                        print(
                            "WebSocket connection closed while idle (detected by ping)."
                        )
                        # --- Try one last history check ---
                        history_result = get_comfyui_results_from_history(
                            prompt_id, server_address, output_node_id
                        )
                        if history_result: # Not None or empty []
                            print("Final history check after WS ping close succeeded.")
                            images_data = history_result
                            retrieved_via_history = True
                        else:
                            images_data = []
                        break # Exit loop
                    except Exception as ping_e:
                        print(f"Error sending WebSocket ping: {ping_e}")

                continue # Go to next loop iteration after handling timeout

            except websocket.WebSocketConnectionClosedException:
                print("WebSocket connection closed by server.")
                # --- Try one last history check ---
                history_result = get_comfyui_results_from_history(
                    prompt_id, server_address, output_node_id
                )
                if history_result: # Not None or empty []
                    print("Final history check after WS close succeeded.")
                    images_data = history_result
                    retrieved_via_history = True
                else:
                    images_data = []
                break # Exit loop

            # --- Process Received Message ---
            if isinstance(out, str):
                try:
                    message = json.loads(out)
                except json.JSONDecodeError as e:
                    print(
                        f"Error decoding WebSocket message: {e}\nReceived raw: {out[:200]}..."
                    )
                    continue

                msg_type = message.get("type")
                data = message.get("data", {})

                if msg_type == "status":
                    status = data.get("status", {})
                    exec_info = status.get("exec_info", {})
                    queue_remaining = exec_info.get("queue_remaining", 0)
                    if progress_callback:
                        progress_callback(f"Queue: {queue_remaining}")

                elif msg_type == "progress":
                    value = data.get("value", 0)
                    max_val = data.get("max", 0)
                    if max_val > 0 and progress_callback:
                        percent = int((value / max_val) * 100)
                        progress_callback(f"Executing: {percent}% ({value}/{max_val})")

                elif msg_type == "executing":
                    prompt_id_msg = data.get("prompt_id")
                    node_id_msg = data.get("node")

                    if node_id_msg is None and prompt_id_msg == prompt_id:
                        print("Execution finished signal received (node=None).")
                        prompt_execution_finished_signal = True
                        # Don't break yet

                    elif (
                        node_id_msg == int(output_node_id) # Node IDs in WS messages are ints
                        and prompt_id_msg == prompt_id
                    ):
                        print(f"Output node {output_node_id} is executing...")
                        if progress_callback:
                            progress_callback(f"Node {output_node_id} running")

                elif msg_type == "executed":
                    prompt_id_msg = data.get("prompt_id")
                    node_id_msg = data.get("node_id") # Node IDs in WS messages are ints

                    if (
                        node_id_msg == int(output_node_id)
                        and prompt_id_msg == prompt_id
                    ):
                        print(
                            f"Output node {output_node_id} finished execution (message received)."
                        )
                        output_node_executed_signal = True
                        # --- Use fetch helper for WS data (list of refs) ---
                        images_data = fetch_images_from_ws_or_history_data(
                            data,
                            "ws",
                            prompt_id,
                            server_address,
                            output_node_id,
                            progress_callback,
                        )
                        retrieved_via_history = False
                        break # Exit loop, we have results via WS

            # End of message processing loop

    # --- Handle specific exceptions that might break the loop ---
    except ConnectionAbortedError as e:
        print(f"WebSocket Error: {e}")
        # --- Try history on connection abort ---
        history_result = get_comfyui_results_from_history(
            prompt_id, server_address, output_node_id
        )
        if history_result:
            print("History check after WS ConnectionAbortedError succeeded.")
            images_data = history_result
            retrieved_via_history = True
        else:
            raise # Re-raise if history also fails

    except TimeoutError as e: # Catch specific timeout from inner logic
        print(f"Operation Timeout Error: {e}")
        # History check is already done inside timeout logic, re-raise
        raise

    except Exception as e:
        print(f"Unexpected error during WebSocket processing: {e}")
        import traceback

        traceback.print_exc()
        # --- Try final history check on unexpected error ---
        history_result = get_comfyui_results_from_history(
            prompt_id, server_address, output_node_id
        )
        if history_result:
            print("Final history check after unexpected WS error succeeded.")
            images_data = history_result
            retrieved_via_history = True
        else:
            # If history fails too, wrap original exception
            raise RuntimeError(
                f"Unexpected error during WebSocket processing and history fallback failed: {e}"
            ) from e

    finally:
        # Ensure WebSocket is closed regardless of how the loop exits
        if ws and ws.connected:
            ws.close()
            print("WebSocket closed.")

    # --- Post-Loop / Final Checks ---

    # If loop finished without results, but overall timeout NOT reached (e.g., normal finish signal but no 'executed' msg)
    if images_data is None and (time.time() - start_time < overall_timeout_seconds):
        print(
            "WebSocket loop finished without direct results. Performing final /history check."
        )
        history_result = get_comfyui_results_from_history(
            prompt_id, server_address, output_node_id
        )
        if history_result is not None: # Check includes empty list []
            if history_result: # List has contents (bytes)
                print(
                    f"Final history check successful, received {len(history_result)} images."
                )
                images_data = history_result
                retrieved_via_history = True
            else: # Empty list returned
                print(
                    "Final history check returned empty list (no images found for node)."
                )
                images_data = []
        else: # History fetch returned None (error)
            print("Final history check failed.")
            images_data = [] # Assume no images if history fails at the end

    # If loop finished because of overall timeout (and not already handled by inner timeout logic)
    elif images_data is None and (time.time() - start_time >= overall_timeout_seconds):
        print(
            f"ERROR: Overall timeout ({overall_timeout_seconds}s) waiting for ComfyUI results for prompt {prompt_id}."
        )
        # Try one last desperate history check if not done by inner timeout logic
        history_result = get_comfyui_results_from_history(
            prompt_id, server_address, output_node_id
        )
        if history_result: # Not None or empty []
            print(
                f"Final history check after overall timeout succeeded, received {len(history_result)} images."
            )
            images_data = history_result
            retrieved_via_history = True
        else:
            # Raise the timeout error if history still fails
            raise TimeoutError(
                f"Overall timeout ({overall_timeout_seconds}s) waiting for ComfyUI results for prompt {prompt_id}. Final history check also failed."
            )

    # Ensure images_data is always a list (even if empty) before returning
    if images_data is None:
        images_data = []

    # --- Final Logging and Return ---
    log_source = "History API fallback" if retrieved_via_history else "WebSocket"
    if not images_data:
        print(
            f"Warning: Operation finished, but no images were retrieved via {log_source}."
        )
    else:
        print(f"Info: Images retrieved via {log_source} ({len(images_data)} images).")

    return images_data

# -------------------------------------------------------------------
# HELPER: Create a plane with image texture fitted to camera view (REMOVED)
# (Function create_plane_with_image_fit_camera is no longer needed)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# OPERATOR: Main Modal Operator
# -------------------------------------------------------------------
class OBJECT_OT_run_comfyui_modal(bpy.types.Operator):
    bl_idname = "object.run_comfyui_modal"
    bl_label = "Run ComfyUI GPencil Workflow (Modal)" # Updated Label
    bl_description = (
        "Capture frames, create video, send to ComfyUI, create Grease Pencil references fitted to camera. "
        "Requires ACTIVE GP OBJECT and SPA addon features. Runs in background." # Updated Desc
    )
    bl_options = {"REGISTER", "UNDO"}

    # --- Configurable Node IDs (Unchanged) ---
    video_path_node_id: bpy.props.StringProperty(
        name="Video Input Node ID", default="107"
    )
    prompt_node_id: bpy.props.StringProperty(name="Prompt Node ID", default="3")
    latent_node_id: bpy.props.StringProperty(
        name="Empty Latent Node ID", default="119"
    )
    output_node_id: bpy.props.StringProperty(name="Image Output Node ID", default="12")

    # --- Modal State Variables (Unchanged) ---
    _timer = None
    _thread = None
    _result_queue = None
    _thread_status: bpy.props.StringProperty(options={"SKIP_SAVE"})
    _thread_progress: bpy.props.StringProperty(options={"SKIP_SAVE"})

    # --- Stored State for Thread & Cleanup ---
    temp_dir_obj = None
    temp_dir_path = None
    frames_to_process = []
    frame_paths = []
    frame_pattern = "frame_%04d.png"
    temp_video_path = None
    workflow = None
    client_id = None
    original_frame = 0
    previous_obj = None # Still store original object
    previous_mode = None # Still store original mode
    final_result = None
    # created_planes = [] # REMOVED - No longer creating planes
    # planes_parent = None # REMOVED - No parent for GP strokes needed
    output_img_paths = []
    server_address = ""
    target_gp_object_name: bpy.props.StringProperty(options={"SKIP_SAVE"}) # Store name of target GP object

    # --- Class Method: Poll ---
    @classmethod
    def poll(cls, context):
        # Check base requirements
        if not isinstance(context.region.data, bpy.types.RegionView3D):
            cls.poll_message_set("Operator requires a 3D Viewport")
            return False
        if not websocket:
            cls.poll_message_set("Missing 'websocket-client' Python package (check console/prefs)")
            return False

        # Check SPA GP Addon availability
        if not SPA_GP_AVAILABLE:
            cls.poll_message_set("Requires SPA Studios Grease Pencil features (addon missing/disabled?)")
            return False

        # Check for Active Grease Pencil Object
        active_obj = context.active_object
        if not active_obj or active_obj.type != 'GPENCIL':
            cls.poll_message_set("Requires an active Grease Pencil object")
            return False

        # Check if GP object has data
        if not active_obj.data:
            cls.poll_message_set("Active Grease Pencil object has no data")
            return False

        # Check if running
        wm = context.window_manager
        if getattr(wm, "comfyui_modal_operator_running", False):
            # Don't set poll message if just running, button will be disabled
            return False # Prevent running again

        return True


    # --- Worker Thread Function (Unchanged logic for video/comfy interaction) ---
    def _comfyui_worker_thread(self):
        """
        Function executed in a separate thread. Performs video creation,
        ComfyUI queuing, and waits for results. Communicates via queue and status vars.
        IMPORTANT: Do NOT interact with bpy data from here except setting simple instance vars.
        """
        try:
            # --- 2) Create Video ---
            self._thread_status = "Creating temporary video..."
            self._thread_progress = ""

            start_num = self.frames_to_process[0] if self.frames_to_process else None
            video_filename = f"input_video_{self.frames_to_process[0] if start_num is not None else 0}.mp4"
            self.temp_video_path = os.path.join(self.temp_dir_path, video_filename)

            # Need to get frame rate from main thread data (stored during invoke)
            # We can pass it as an argument or store it on self if needed.
            # For now, assume it was stored on self.frame_rate during invoke.
            frame_rate_for_video = getattr(self, "scene_frame_rate", 24) # Get stored frame rate

            create_video_from_frames(
                self.temp_dir_path,
                self.temp_video_path,
                frame_rate_for_video, # Use the stored value
                self.frame_pattern,
                start_number=start_num,
            )

            # --- 3) Modify and Queue Workflow ---
            abs_video_path = os.path.abspath(self.temp_video_path).replace("\\", "/")
            self.workflow[self.video_path_node_id]["inputs"]["video"] = abs_video_path
            # Get prompt/CN settings stored on self during invoke
            self.workflow[self.prompt_node_id]["inputs"]["text"] = getattr(self, "scene_user_prompt", "")
            depth_strength = getattr(self, "scene_depth_strength", 0.5)
            invert_depth = getattr(self, "scene_invert_depth", False)

            depth_apply_node_id = "114" # Example ID, ensure it matches your workflow
            depth_invert_node_id = "123" # Example ID, ensure it matches your workflow

            # Update Depth ControlNet strength
            if depth_apply_node_id in self.workflow and "inputs" in self.workflow[depth_apply_node_id]:
                self.workflow[depth_apply_node_id]["inputs"]["strength"] = depth_strength
                print(f"  Set Depth CN Strength (Node {depth_apply_node_id}) to: {depth_strength}")
            else:
                print(f"  Warning: Depth CN Apply Node '{depth_apply_node_id}' or its 'inputs' not found in workflow.")

            # Update Depth Inversion setting
            if depth_invert_node_id in self.workflow and "inputs" in self.workflow[depth_invert_node_id]:
                self.workflow[depth_invert_node_id]["inputs"]["value"] = invert_depth
                print(f"  Set Depth Inversion (Node {depth_invert_node_id}) to: {invert_depth}")
            else:
                print(f"  Warning: Depth Inversion Node '{depth_invert_node_id}' or its 'inputs' not found in workflow.")

            self._thread_status = (
                f"Queueing ComfyUI prompt (Client: {self.client_id[:8]}...)"
            )
            self._thread_progress = ""

            queue_response = queue_comfyui_prompt(
                self.workflow, self.server_address, self.client_id
            )
            prompt_id = queue_response.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(
                    f"Did not receive prompt_id. Response: {queue_response}"
                )
            print(f"ComfyUI Prompt ID: {prompt_id}")

            # --- 4) Wait for Results via WebSocket ---
            self._thread_status = (
                f"Waiting for ComfyUI results (Prompt: {prompt_id[:8]}...)"
            )
            self._thread_progress = ""

            def progress_update(message):
                self._thread_status = (
                    f"Waiting for ComfyUI results (Prompt: {prompt_id[:8]}...)"
                )
                self._thread_progress = message

            print("Getting images")
            output_images_data = get_comfyui_images_ws(
                prompt_id,
                self.server_address,
                self.client_id,
                self.output_node_id,
                progress_callback=progress_update,
            )
            print("Got images")

            # --- Success ---
            self._thread_status = "Received results from ComfyUI."
            self._thread_progress = ""
            self._result_queue.put(output_images_data) # Put list of image data

        except Exception as e:
            # --- Failure ---
            error_short = str(e).splitlines()[0]
            self._thread_status = "Error during ComfyUI interaction."
            self._thread_progress = f"{type(e).__name__}: {error_short}"
            print(f"Error in worker thread: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            self._result_queue.put(e) # Put the exception object in the queue


    # --- Main Execution Logic (Called by Modal on Completion) ---
    def execute_finish(self, context):
        """
        Processes the results received from the worker thread. Creates GPencil References.
        Assumes it's called from the modal method when results are ready (main thread).
        Uses context override and ensures correct mode for GP ops.
        """
        print("Entering Execute Finish method (GPencil)...")
        wm = context.window_manager

        # --- Check for Target Grease Pencil Object ---
        gp_object = bpy.data.objects.get(self.target_gp_object_name)
        if not gp_object or gp_object.type != 'GPENCIL' or not gp_object.data:
            # (Error reporting unchanged)
            self.report({"ERROR"}, f"Target GP object '{self.target_gp_object_name}' not found/invalid.")
            wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = "Target GP object lost"
            return {"CANCELLED"}
        gpd = gp_object.data

        # --- Handle Final Result (Error or Image List) ---
        # (Error handling for self.final_result unchanged)
        if isinstance(self.final_result, Exception):
            error_short = str(self.final_result).splitlines()[0]
            self.report({"ERROR"}, f"Worker thread error: {type(self.final_result).__name__}: {error_short}")
            wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = f"{type(self.final_result).__name__}: {error_short}"
            return {"CANCELLED"}
        elif self.final_result is None:
             self.report({"WARNING"}, "No valid response from ComfyUI."); wm.comfyui_modal_status = "Finished (No Response)"; wm.comfyui_modal_progress = ""
             return {"FINISHED"}
        elif not isinstance(self.final_result, list):
             self.report({"ERROR"}, f"Unexpected result type: {type(self.final_result)}"); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = f"Internal error: Bad result type"
             return {"CANCELLED"}
        elif not self.final_result:
             self.report({"INFO"}, "No images generated/retrieved."); wm.comfyui_modal_status = "Finished (No Images)"; wm.comfyui_modal_progress = ""
             return {"FINISHED"}

        # --- Prepare for GP Reference Creation ---
        print(f"Execute Finish received {len(self.final_result)} image(s) for GPencil.")
        self.report({"INFO"}, f"Received {len(self.final_result)} image(s). Creating GP references...")
        wm.progress_begin(0, len(self.final_result))
        self.output_img_paths = []

        # --- Find a suitable 3D Viewport context ---
        # (Finding view3d_area and view3d_region unchanged)
        view3d_area = None
        view3d_region = None
        if context.area and context.area.type == 'VIEW_3D':
            view3d_area = context.area
            for region in view3d_area.regions:
                 if region.type == 'WINDOW': view3d_region = region; break
        if not (view3d_area and view3d_region):
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    view3d_area = area
                    for region in area.regions:
                        if region.type == 'WINDOW': view3d_region = region; break
                    if view3d_region: break
        if not (view3d_area and view3d_region):
             self.report({"ERROR"}, "Could not find a suitable 3D Viewport context."); wm.progress_end(); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = "Context Error (No 3D View)"
             return {"CANCELLED"}

        # Store original mode for final restoration
        original_active_at_start = context.active_object
        original_mode_at_start = original_active_at_start.mode if original_active_at_start else "OBJECT"

        # --- Create GP References Loop within Context Override ---
        success_count = 0
        num_results = len(self.final_result)
        num_frames_requested = len(self.frames_to_process)
        if num_results != num_frames_requested: print(f"Warning: Received {num_results} images, but requested {num_frames_requested} frames.")
        original_scene_frame = context.scene.frame_current

        # Use the found 3D View context for all operations within the loop
        print(f"Using context override: Area Type='{view3d_area.type}', Region='{view3d_region.type}'")
        try: # Add try...finally to ensure mode is reset
            with context.temp_override(window=context.window, area=view3d_area, region=view3d_region):

                # --- Ensure correct context and GP Edit mode BEFORE the loop ---
                print("Setting active object and GP Edit mode...")
                context.view_layer.objects.active = gp_object
                gp_object.select_set(True) # Ensure it's selected too
                # Try setting GP Edit mode
                try:
                    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
                    print(f"  Mode set to: {context.object.mode}")
                    if context.object.mode != 'EDIT_GPENCIL':
                         raise RuntimeError("Failed to enter GP Edit Mode.")
                except Exception as mode_e:
                     self.report({"ERROR"}, f"Could not set GP Edit mode: {mode_e}. Aborting.")
                     wm.progress_end(); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = f"Mode Error: {mode_e}"
                     # Exiting override context here - mode restoration needs care
                     # We should still attempt to go back to Object mode outside the 'with'
                     # Set success_count to -1 to indicate early failure for mode reset logic? No, just return.
                     # Need to reset mode manually here if possible before returning
                     try: bpy.ops.object.mode_set(mode='OBJECT')
                     except: pass
                     return {"CANCELLED"}

                # --- Get or Create Target GP Layer ---
                # (Moved inside override and mode check for safety)
                target_layer_name = "ComfyUI Output"
                target_layer = gpd.layers.get(target_layer_name)
                if not target_layer:
                    print(f"Creating Grease Pencil layer: {target_layer_name}")
                    target_layer = gpd.layers.new(target_layer_name, set_active=True)
                else:
                    print(f"Using existing Grease Pencil layer: {target_layer_name}")
                    gpd.layers.active = target_layer # Ensure it's active

                # --- Loop through images ---
                for i, img_data in enumerate(self.final_result):
                    # (Status updates unchanged)
                    current_status = f"Processing output {i+1}/{num_results}"
                    current_progress = f"Frame {frame_num}" if 'frame_num' in locals() else ""
                    wm.comfyui_modal_status = current_status; wm.comfyui_modal_progress = current_progress
                    try: context.workspace.status_text_set(f"ComfyUI: {current_status} ({current_progress})")
                    except: pass

                    if not isinstance(img_data, bytes) or not img_data:
                        self.report({"WARNING"}, f"Skipping invalid image data at index {i}.")
                        wm.progress_update(i + 1); continue

                    # Determine frame number
                    if i < num_frames_requested: frame_num = self.frames_to_process[i]
                    else: frame_num = self.original_frame + i; print(f"  Using fallback frame num {frame_num} for index {i}")
                    wm.comfyui_modal_progress = f"Frame {frame_num}" # Update progress again with frame number

                    # (Saving image unchanged)
                    output_img_filename = f"comfy_out_{frame_num:04d}.png"
                    if not self.temp_dir_path or not os.path.isdir(self.temp_dir_path):
                         self.report({"ERROR"}, f"Temp directory missing ({self.temp_dir_path}). Aborting."); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = "Temp dir lost"; break
                    output_img_path = os.path.join(self.temp_dir_path, output_img_filename)
                    print(f"\nProcessing image {i+1}/{num_results} for frame {frame_num} -> GP Ref...")

                    try:
                        print(f"  Saving image data to: {output_img_path}")
                        with open(output_img_path, "wb") as f: f.write(img_data)
                        if not os.path.exists(output_img_path) or os.path.getsize(output_img_path) == 0: raise IOError("Temp image file not found/empty after writing.")
                        print(f"  Image data saved successfully.")
                        self.output_img_paths.append(output_img_path)

                        # Set Current Frame
                        print(f"  Setting scene frame to: {frame_num}")
                        context.scene.frame_set(frame_num)

                        # Call SPA GP Reference Import Function (Should now have correct mode and context)
                        print(f"  Calling import_image_as_gp_reference for '{output_img_path}'...")
                        spa_gp_core.import_image_as_gp_reference(
                            context=context, obj=gp_object, img_filepath=output_img_path,
                            pack_image=False, add_new_layer=False, add_new_keyframe=True,
                        )
                        print(f"  Successfully created GP reference for frame {frame_num}.")
                        success_count += 1

                    except Exception as e:
                        self.report({"ERROR"}, f"Failed creating GP reference for image {i} (frame {frame_num}): {e}")
                        import traceback; traceback.print_exc()
                        if os.path.exists(output_img_path):
                            try: os.remove(output_img_path);
                            except OSError as rem_e: print(f"Could not remove failed temp image: {rem_e}")
                        # Continue loop even if one frame fails
                    finally:
                         wm.progress_update(i + 1)
            # --- End For Loop ---

        finally:
            # --- Ensure mode is reset to OBJECT after processing ---
            # This runs even if errors occurred inside the 'with' block
            print("Attempting to restore OBJECT mode...")
            try:
                # Check if the object still exists and is active GP object
                if context.view_layer.objects.active == gp_object and gp_object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                    print("  Mode restored to OBJECT.")
                elif context.mode != 'OBJECT': # Fallback if active object changed
                     bpy.ops.object.mode_set(mode='OBJECT')
                     print("  Mode restored to OBJECT (fallback).")
            except Exception as mode_reset_e:
                print(f"  Warning: Could not restore OBJECT mode: {mode_reset_e}")
        # --- End Context Override block ---


        # --- Restore original frame ---
        # (Frame restoration unchanged)
        if context.scene:
             if context.scene.frame_start <= original_scene_frame <= context.scene.frame_end: context.scene.frame_set(original_scene_frame)
             else: print(f"  Original scene frame {original_scene_frame} no longer valid.")
        else: print("  Could not restore frame, scene context lost.")
        wm.progress_end()

        # --- Restore original mode *if different from OBJECT* ---
        # (Mode restoration slightly modified to check against OBJECT)
        print(f"Original mode at start was: {original_mode_at_start}")
        if original_mode_at_start != 'OBJECT':
            print(f"Attempting to restore original mode ({original_mode_at_start}) for object '{gp_object.name}'...")
            try:
                if gp_object and gp_object.name in context.view_layer.objects:
                     context.view_layer.objects.active = gp_object
                     valid_modes = {"OBJECT", "EDIT", "SCULPT", "PAINT", "WEIGHT"} # Add EDIT_GPENCIL? Check API
                     if original_mode_at_start in valid_modes or original_mode_at_start == 'EDIT_GPENCIL':
                         bpy.ops.object.mode_set(mode=original_mode_at_start)
                         print(f"  Restored original mode to: {original_mode_at_start}")
                     else: print(f"  Cannot restore original mode: '{original_mode_at_start}'.")
                else: print(f"  Cannot restore original mode, GP object '{self.target_gp_object_name}' not found.")
            except Exception as e: print(f"Could not restore original mode {original_mode_at_start}: {e}")
        else:
            print("Original mode was OBJECT, no final mode restoration needed.")


        final_op_status = f"Successfully created {success_count} / {num_results} GP references in '{gp_object.name}'."
        self.report({"INFO"}, final_op_status)
        wm.comfyui_modal_status = "Finished"
        wm.comfyui_modal_progress = final_op_status
        print("Exiting Execute Finish method (GPencil).")
        return {"FINISHED"}

    # --- Modal Method ---
    def modal(self, context, event):
        wm = context.window_manager

        # --- Check thread status REGARDLESS of event type (except ESC/Cancel) ---
        # Check only if the thread exists and we haven't already processed the result
        if self._thread and self.final_result is None and not self._thread.is_alive():
            print(f"Worker thread finished (detected on event type: {event.type} at {time.time():.4f}).")
            try:
                # Retrieve result from the queue
                self.final_result = self._result_queue.get_nowait()
                print(f"Result retrieved from queue. Type: {type(self.final_result)}")

            except queue.Empty:
                self.report({"ERROR"}, "Thread finished but result queue is empty upon check.")
                self.final_result = RuntimeError("Thread finished but queue empty.")
                # Update status immediately if error occurs here
                wm.comfyui_modal_status = "Finished with Error"
                wm.comfyui_modal_progress = "Internal error: Queue empty"
            except Exception as e:
                self.report({"ERROR"}, f"Error retrieving result from queue: {e}")
                self.final_result = e
                # Update status immediately if error occurs here
                wm.comfyui_modal_status = "Finished with Error"
                wm.comfyui_modal_progress = f"Internal error: Queue retrieval failed {type(e).__name__}"

            # If we successfully got a result (or an error object), proceed to finish
            if self.final_result is not None:
                # Call the main execution logic (now runs in main thread)
                # This needs self.final_result to be set before calling
                final_status = self.execute_finish(context)
                # Finish the modal operator (finish_or_cancel handles cleanup)
                return self.finish_or_cancel(
                    context, cancelled=("CANCELLED" in final_status)
                )
            # If self.final_result is still None after checks (shouldn't happen often), continue modal

        # --- Handle specific events ---
        # Force UI update (panel status) - Still useful
        if context.area:
            context.area.tag_redraw()

        # Handle cancellation
        if event.type == "ESC" or not getattr(wm, "comfyui_modal_operator_running", True):
            print("ESC pressed or operator cancelled externally.")
            return self.finish_or_cancel(context, cancelled=True)

        # Process timer events for UI STATUS UPDATES ONLY while running
        if event.type == "TIMER":
            # Update UI Status only if the thread is still running and we haven't processed results
            if self._thread and self.final_result is None:
                current_status = self._thread_status
                current_progress = self._thread_progress

                if wm.comfyui_modal_status != current_status:
                    wm.comfyui_modal_status = current_status
                if wm.comfyui_modal_progress != current_progress:
                    wm.comfyui_modal_progress = current_progress

                status_bar_text = f"ComfyUI: {current_status}"
                if current_progress:
                    status_bar_text += f" ({current_progress})"
                context.workspace.status_text_set(status_bar_text)

            # The actual check for thread completion is now done *above* for all events

        # Allow other events (like navigation) to pass through
        # We've already checked thread status if relevant, so just pass through
        return {"PASS_THROUGH"}

    # --- Invoke Method (Starts the process) ---
    def invoke(self, context, event):
        # --- POLL checks are now done in poll() method ---
        # Redundant checks removed here for clarity, poll() handles preconditions.

        wm = context.window_manager
        # Check if already running (handled by poll, but double-check)
        if getattr(wm, "comfyui_modal_operator_running", False):
            self.report({"WARNING"}, "Operation already in progress.")
            return {"CANCELLED"}

        # --- Initial Setup & Validation ---
        self._thread_status = "Initializing..."
        self._thread_progress = ""
        self.final_result = None

        prefs = context.preferences.addons[__name__].preferences
        self.server_address = prefs.comfyui_address.strip()
        if not self.server_address or not self.server_address.startswith(
            ("http://", "https://")
        ):
            self.report(
                {"ERROR"},
                "ComfyUI server address not set correctly in preferences.",
            )
            # Automatically open preferences
            bpy.ops.preferences.open_comfyui_addon_prefs('INVOKE_DEFAULT')
            return {"CANCELLED"}

        # --- Get Target GP Object (guaranteed by poll) ---
        self.target_gp_object_name = context.active_object.name # Store name for later retrieval
        print(f"Target Grease Pencil Object: {self.target_gp_object_name}")

        # --- Get Settings from Scene Properties ---
        scene_props = context.scene.comfyui_props
        # Store relevant props on self for the worker thread to access
        self.scene_user_prompt = scene_props.user_prompt
        self.scene_frame_rate = scene_props.frame_rate
        self.scene_depth_strength = scene_props.controlnet_depth_strength
        self.scene_invert_depth = scene_props.invert_depth_input

        frame_mode = scene_props.frame_mode
        frame_start = scene_props.frame_start
        frame_end = scene_props.frame_end

        # Store state
        self.original_frame = context.scene.frame_current
        self.previous_obj = context.active_object # Store the GP object as previous
        self.previous_mode = self.previous_obj.mode if self.previous_obj else "OBJECT"

        # Determine frames
        if frame_mode == "CURRENT":
            self.frames_to_process = [self.original_frame]
        else: # RANGE
            if frame_start > frame_end:
                self.report(
                    {"ERROR"}, "Start frame must be less than or equal to end frame."
                )
                return {"CANCELLED"}
            self.frames_to_process = list(range(frame_start, frame_end + 1))

        if not self.frames_to_process:
            self.report({"WARNING"}, "No frames selected for processing.")
            return {"CANCELLED"}

        # --- Create Temp Directory ---
        try:
            self.temp_dir_obj = None
            self.temp_dir_path = None
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="blender_comfy_gp_") # Updated prefix
            self.temp_dir_path = self.temp_dir_obj.name
            print(f"Using temporary directory: {self.temp_dir_path}")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to create temp directory: {e}")
            self._cleanup_temp_files()
            return {"CANCELLED"}

        # --- SYNCHRONOUS Frame Capture ---
        self._thread_status = "Capturing frames..."
        self._thread_progress = ""
        wm.comfyui_modal_status = self._thread_status
        wm.comfyui_modal_progress = self._thread_progress
        context.workspace.status_text_set(
            f"ComfyUI: {self._thread_status} (This may take time)"
        )

        self.report(
            {"INFO"},
            f"Capturing {len(self.frames_to_process)} frame(s)... (This may take time)",
        )
        wm.progress_begin(0, len(self.frames_to_process))
        self.frame_paths = []
        capture_success = True

        # Ensure Object Mode for capture stability (might affect GP less, but safer)
        mode_switched_for_capture = False
        if self.previous_mode != "OBJECT" and self.previous_obj:
            try:
                context.view_layer.objects.active = self.previous_obj
                bpy.ops.object.mode_set(mode="OBJECT")
                mode_switched_for_capture = True
                print("Switched active object to Object mode for capture.")
            except Exception as e:
                self.report(
                    {"WARNING"}, f"Could not switch active object to Object mode before capture: {e}"
                )

        current_frame_capture_start = time.time()
        for i, frame_num in enumerate(self.frames_to_process):
            frame_start_time = time.time()
            context.scene.frame_set(frame_num)
            self._thread_progress = f"{i+1}/{len(self.frames_to_process)}"
            wm.progress_update(i)
            context.view_layer.update() # Force update

            frame_filename = self.frame_pattern % frame_num
            output_path = os.path.join(self.temp_dir_path, frame_filename)
            try:
                print(f"Capturing frame {frame_num} to {output_path}...")
                capture_viewport(output_path, context)
                self.frame_paths.append(output_path)
                print(
                    f"  Frame {frame_num} captured in {time.time() - frame_start_time:.2f}s"
                )
            except Exception as e:
                self.report({"ERROR"}, f"Failed to capture frame {frame_num}: {e}")
                import traceback
                traceback.print_exc()
                capture_success = False
                break

        wm.progress_end()
        print(f"Total capture time: {time.time() - current_frame_capture_start:.2f}s")

        # Restore mode if changed for capture
        if (
            mode_switched_for_capture
            and self.previous_obj
            and self.previous_obj.name in context.view_layer.objects
        ):
            try:
                print(f"Restoring mode to {self.previous_mode} after capture.")
                context.view_layer.objects.active = self.previous_obj
                # Check validity before setting (including GP modes)
                valid_modes = {"OBJECT", "EDIT", "SCULPT", "PAINT", "WEIGHT",
                               "EDIT_GPENCIL", "SCULPT_GPENCIL", "PAINT_GPENCIL", "WEIGHT_GPENCIL"}
                if self.previous_mode in valid_modes:
                        bpy.ops.object.mode_set(mode=self.previous_mode)
                else:
                    print(f"  Cannot restore invalid original mode '{self.previous_mode}', staying in Object.")
                    # Ensure we are in object mode if the original mode was invalid
                    if context.active_object and context.active_object.mode != 'OBJECT':
                        bpy.ops.object.mode_set(mode='OBJECT')

            except Exception as e:
                print(f"Could not restore mode after capture: {e}")
                # Attempt to force Object mode as a fallback if restoration failed
                try:
                     if context.active_object and context.active_object.mode != 'OBJECT':
                         bpy.ops.object.mode_set(mode='OBJECT')
                except Exception as fallback_e:
                     print(f"Could not force Object mode as fallback: {fallback_e}")


        # --- Check Capture Success BEFORE proceeding ---
        if not capture_success or not self.frame_paths:
            self.report({"ERROR"}, "Frame capture failed. Aborting.")
            self._cleanup_temp_files()
            # Reset WM status on failure
            wm.comfyui_modal_operator_running = False
            wm.comfyui_modal_status = "Idle"
            wm.comfyui_modal_progress = ""
            context.workspace.status_text_set(None)
            return {"CANCELLED"}

        # --- Prepare Workflow ---
        try:
            self.workflow = json.loads(COMFYUI_WORKFLOW_JSON)
            # Basic validation (check if essential nodes exist)
            essential_nodes = [
                self.video_path_node_id,
                self.prompt_node_id,
                self.latent_node_id, # Use the property here
                self.output_node_id,
            ]
            for node_id in essential_nodes:
                if node_id not in self.workflow:
                    raise ValueError(f"Workflow missing essential node ID: {node_id}")
            # Check for 'inputs' where needed (you might refine this)
            if "inputs" not in self.workflow[self.video_path_node_id]:
                 raise ValueError(f"Video node {self.video_path_node_id} missing 'inputs'.")
            if "inputs" not in self.workflow[self.prompt_node_id]:
                 raise ValueError(f"Prompt node {self.prompt_node_id} missing 'inputs'.")
            if "inputs" not in self.workflow[self.latent_node_id]:
                 raise ValueError(f"Latent node {self.latent_node_id} missing 'inputs'.")


            # --- DYNAMICALLY SET LATENT DIMENSIONS ---
            try:
                render = context.scene.render
                scale = render.resolution_percentage / 100.0
                base_width = render.resolution_x
                base_height = render.resolution_y

                # Calculate final dimensions based on render settings
                final_width = int(base_width * scale)
                final_height = int(base_height * scale)

                # --- Ensure dimensions are multiples of 8 (VERY IMPORTANT for Stable Diffusion) ---
                # Round down to the nearest multiple of 8
                latent_width = final_width - (final_width % 8)
                latent_height = final_height - (final_height % 8)

                # Ensure minimum size (optional, but good practice if rounding down could go too low)
                latent_width = max(64, latent_width) # e.g., minimum 64 pixels
                latent_height = max(64, latent_height)

                print(f"Blender Render dimensions: {base_width}x{base_height} @ {render.resolution_percentage}% => Actual: {final_width}x{final_height}")
                print(f"Setting ComfyUI Latent dimensions (Node {self.latent_node_id}) to: {latent_width}x{latent_height} (multiple of 8)")

                # Modify the workflow dictionary
                self.workflow[self.latent_node_id]["inputs"]["width"] = latent_width
                self.workflow[self.latent_node_id]["inputs"]["height"] = latent_height
                print(f"  Successfully updated dimensions in workflow for node {self.latent_node_id}.")

            except KeyError:
                 # This should be caught by the earlier validation, but good to have a specific catch
                 self.report({"ERROR"}, f"Latent Node '{self.latent_node_id}' structure invalid in workflow JSON (missing 'inputs'?).")
                 self._cleanup_temp_files(); wm.comfyui_modal_operator_running = False; wm.comfyui_modal_status = "Idle"; wm.comfyui_modal_progress = ""; context.workspace.status_text_set(None)
                 return {"CANCELLED"}
            except Exception as dim_e:
                self.report({"ERROR"}, f"Failed to calculate/set dynamic latent dimensions: {dim_e}")
                import traceback; traceback.print_exc()
                self._cleanup_temp_files(); wm.comfyui_modal_operator_running = False; wm.comfyui_modal_status = "Idle"; wm.comfyui_modal_progress = ""; context.workspace.status_text_set(None)
                return {"CANCELLED"}
            # --- END DYNAMIC DIMENSIONS ---

        except Exception as e:
            self.report({"ERROR"}, f"Failed to load/validate/prepare workflow JSON: {e}")
            self._cleanup_temp_files(); wm.comfyui_modal_operator_running = False; wm.comfyui_modal_status = "Idle"; wm.comfyui_modal_progress = ""; context.workspace.status_text_set(None)
            return {"CANCELLED"}

        # --- Start Background Thread ---
        self._result_queue = queue.Queue()
        self.client_id = str(uuid.uuid4())
        self._thread = threading.Thread(
            target=self._comfyui_worker_thread, name="ComfyUI_Worker_GP"
        )
        self._thread.daemon = True

        self._thread_status = "Starting ComfyUI request..."
        self._thread_progress = ""
        wm.comfyui_modal_operator_running = True
        wm.comfyui_modal_status = self._thread_status
        wm.comfyui_modal_progress = self._thread_progress
        context.workspace.status_text_set(f"ComfyUI: {self._thread_status}")

        self._thread.start()
        print("Worker thread started.")

        # --- Register Modal Handler ---
        self._timer = context.window_manager.modal_handler_add(self)
        print("Modal handler added.")

        return {"RUNNING_MODAL"}

    # --- Cleanup and State Restoration ---
    def finish_or_cancel(self, context, cancelled=False):
        """Cleans up resources and restores Blender state."""
        print(f"Finishing or cancelling GP operation (Cancelled: {cancelled})")
        wm = context.window_manager

        # --- REMOVED EXPLICIT TIMER REMOVAL ---
        # Let Blender handle timer cleanup when modal returns FINISHED/CANCELLED
        print("Skipping explicit event_timer_remove call.")
        self._timer = None # Still reset our internal variable

        # --- Rest of the function remains the same ---

        # Clear status bar and WM state
        context.workspace.status_text_set(None)
        wm.comfyui_modal_operator_running = False
        if cancelled and wm.comfyui_modal_status != "Finished with Error": # Don't override specific error messages
            wm.comfyui_modal_status = "Cancelled"
            wm.comfyui_modal_progress = ""
        # else: Keep success/error message from execute_finish

        # Ensure thread is finished
        if self._thread and self._thread.is_alive():
            print("Warning: Worker thread still alive during finish/cancel. Attempting join...")
            self._thread.join(timeout=5.0);
            if self._thread.is_alive(): print("ERROR: Worker thread did not terminate cleanly!")
        self._thread = None

        # Cleanup temporary files and directory
        self._cleanup_temp_files()

        # Restore Blender state (frame, selection, mode) - Using previous version logic
        try:
            # Restore frame first
            if hasattr(self, "original_frame") and context.scene.frame_current != self.original_frame:
                if context.scene and context.scene.frame_start <= self.original_frame <= context.scene.frame_end:
                     print(f"Restoring scene frame to {self.original_frame}")
                     context.scene.frame_set(self.original_frame)
                else: print(f"Skipping frame restoration: Original frame {self.original_frame} outside scene range or scene invalid.")

            # Restore selection and mode TO THE ORIGINAL OBJECT (which was GP)
            # Use the stored name to re-acquire object
            original_gp_obj = bpy.data.objects.get(getattr(self, "target_gp_object_name", None))
            prev_mode = getattr(self, "previous_mode", None) # The mode when invoke started

            if original_gp_obj and prev_mode:
                 print(f"Attempting to restore original selection/mode to: {original_gp_obj.name} / {prev_mode}")
                 try:
                     current_mode = context.mode
                     # Ensure we are in object mode before changing selection/active object
                     if current_mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
                     # Deselect all then select/activate the target
                     bpy.ops.object.select_all(action='DESELECT')
                     context.view_layer.objects.active = original_gp_obj
                     original_gp_obj.select_set(True)
                     # Now set the mode if needed and if it's valid
                     if prev_mode != context.active_object.mode:
                           valid_modes = {"OBJECT", "EDIT", "SCULPT", "PAINT", "WEIGHT", "EDIT_GPENCIL", "SCULPT_GPENCIL", "PAINT_GPENCIL", "WEIGHT_GPENCIL"}
                           if prev_mode in valid_modes: bpy.ops.object.mode_set(mode=prev_mode); print(f"  Restored original mode to: {prev_mode}")
                           else: print(f"  Cannot restore invalid original mode '{prev_mode}', staying in Object.")
                 except Exception as restore_e: print(f"  Warning: Could not fully restore original selection/mode: {restore_e}")
            else: print(f"Skipping original selection/mode restoration: Original GP object '{getattr(self, 'target_gp_object_name', 'None')}' not found or prev_mode missing.")

        except Exception as e: self.report({"WARNING"}, f"Error during state restoration: {e}")


        self.report({"INFO"}, f"ComfyUI GP operation {'cancelled' if cancelled else 'finished'}.")
        print("-" * 30)
        return {"CANCELLED"} if cancelled else {"FINISHED"}

    # --- Cleanup Helper (Unchanged) ---
    def _cleanup_temp_files(self):
        """Helper to remove temporary files and directory using shutil."""
        if hasattr(self, "temp_dir_obj") and self.temp_dir_obj:
            print(
                f"Cleaning up temporary directory object: {self.temp_dir_obj.name}..."
            )
            try:
                self.temp_dir_obj.cleanup()
                print("  Temporary directory object cleaned successfully.")
            except Exception as e:
                print(
                    f"  Warning: TemporaryDirectory object cleanup failed: {e}. Attempting manual removal."
                )
                if self.temp_dir_path and os.path.exists(self.temp_dir_path):
                    try:
                        shutil.rmtree(self.temp_dir_path, ignore_errors=True)
                        print(f"  Manual removal of {self.temp_dir_path} attempted.")
                    except Exception as manual_e:
                        print(
                            f"  ERROR: Manual removal of {self.temp_dir_path} also failed: {manual_e}"
                        )
                        self.report(
                            {"WARNING"},
                            f"Manual cleanup of {self.temp_dir_path} failed. Please remove it manually.",
                        )
            finally:
                self.temp_dir_obj = None
                self.temp_dir_path = None
        elif (
            hasattr(self, "temp_dir_path")
            and self.temp_dir_path
            and os.path.exists(self.temp_dir_path)
        ):
            print(f"Attempting manual cleanup of directory: {self.temp_dir_path}...")
            try:
                shutil.rmtree(self.temp_dir_path, ignore_errors=True)
                print(f"  Manual removal of {self.temp_dir_path} attempted.")
            except Exception as manual_e:
                print(
                    f"  ERROR: Manual removal of {self.temp_dir_path} failed: {manual_e}"
                )
                self.report(
                    {"WARNING"},
                    f"Manual cleanup of {self.temp_dir_path} failed. Please remove it manually.",
                )
            finally:
                self.temp_dir_path = None

        # Ensure lists are cleared
        self.frame_paths = []
        self.output_img_paths = []
        self.temp_video_path = None


# -------------------------------------------------------------------
# HELPER OPERATOR: Open Addon Preferences (Unchanged)
# -------------------------------------------------------------------
class PREFERENCES_OT_open_comfyui_addon_prefs(bpy.types.Operator):
    """Opens the preferences specific to this addon"""

    bl_idname = "preferences.open_comfyui_addon_prefs" # More specific ID
    bl_label = "Open ComfyUI Addon Preferences"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return __name__ in context.preferences.addons

    def execute(self, context):
        try:
            bpy.ops.screen.userpref_show("INVOKE_DEFAULT") # Open preferences window
            # Explicitly activate the Add-ons tab and filter
            bpy.context.preferences.active_section = "ADDONS"
            # Find the preferences window/area reliably
            prefs_window = None
            for window in context.window_manager.windows:
                 screen = window.screen
                 if screen.name == "User Preferences": # Default name
                      prefs_window = window
                      break
            if prefs_window:
                 prefs_area = None
                 for area in prefs_window.screen.areas:
                      if area.type == 'PREFERENCES':
                           prefs_area = area
                           break
                 if prefs_area:
                      prefs_area.spaces.active.filter_text = bl_info["name"]
                 else:
                      print("Could not find PREFERENCES area in User Preferences screen.")
            else:
                 print("Could not find User Preferences window.")

            return {"FINISHED"}
        except Exception as e:
            print(f"Error trying to open preferences for '{__name__}': {e}")
            self.report(
                {"ERROR"},
                f"Could not open preferences automatically: {e}. Please go to Edit > Preferences > Add-ons and search for '{bl_info['name']}'.",
            )
            return {"CANCELLED"}

# -------------------------------------------------------------------
# PANEL: UI in the 3D View sidebar
# -------------------------------------------------------------------
class VIEW3D_PT_comfyui_panel(bpy.types.Panel):
    bl_label = "ComfyUI GP Gen" # Updated Label
    bl_idname = "VIEW3D_PT_comfyui_gp_panel" # Unique ID
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ComfyUI" # Tab name

    # Add poll method to check SPA availability for panel visibility
    @classmethod
    def poll(cls, context):
         # Also check if the core function is available
        return SPA_GP_AVAILABLE and hasattr(spa_gp_core, 'import_image_as_gp_reference')

    def draw_header(self, context):
        # Optional: Add an icon or indicator if SPA is available
        if SPA_GP_AVAILABLE:
            self.layout.label(text="", icon='GREASEPENCIL') # Example icon

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        scene_props = context.scene.comfyui_props

        # Get running state
        is_running = getattr(wm, "comfyui_modal_operator_running", False)

        # --- Display Warning if SPA GP not available (handled by panel poll now) ---
        # if not SPA_GP_AVAILABLE:
        #     box = layout.box()
        #     box.label(text="SPA GPencil features not found!", icon='ERROR')
        #     box.label(text="Ensure SPA addon is enabled.")
        #     return # Stop drawing panel contents if dependency missing

        # --- Main Operator Button ---
        row = layout.row()
        # Operator poll method now handles checks for active GP object etc.
        op = row.operator(
            OBJECT_OT_run_comfyui_modal.bl_idname, icon="IMAGE_DATA", text="Run ComfyUI -> GP" # Updated Text
        )
        row.enabled = not is_running

        # --- Status Display ---
        if is_running:
            status = getattr(wm, "comfyui_modal_status", "Running...")
            progress = getattr(wm, "comfyui_modal_progress", "")
            box = layout.box()
            row = box.row()
            row.label(text=f"Status:")
            row.label(text=status)
            if progress:
                row = box.row()
                row.label(text=f"Details:")
                row.label(text=progress)
            # Add Cancel Button (sets the flag for the modal loop)
            row = box.row()
            cancel_op = row.operator("wm.comfyui_cancel_modal", icon='X', text="Cancel")

        # --- Operator Settings ---
        col = layout.column(align=True)
        col.enabled = not is_running

        col.prop(scene_props, "user_prompt")
        col.separator()
        col.prop(scene_props, "frame_mode")

        if scene_props.frame_mode == "RANGE":
            box = col.box()
            row = box.row(align=True)
            row.prop(scene_props, "frame_start", text="Start")
            row.prop(scene_props, "frame_end", text="End")
            scene = context.scene
            box.label(
                text=f"(Scene Range: {scene.frame_start} - {scene.frame_end})",
                icon="INFO",
            )
        col.separator()
        col.prop(scene_props, "frame_rate")

        col.separator()
        box_cn = col.box()
        box_cn.label(text="ControlNet Strengths:")
        box_cn.prop(scene_props, "controlnet_depth_strength")
        box_cn.prop(scene_props, "invert_depth_input")

        layout.separator()
        # --- Preferences Button ---
        layout.operator(
            PREFERENCES_OT_open_comfyui_addon_prefs.bl_idname,
            text="Settings",
            icon="PREFERENCES",
        )

# -------------------------------------------------------------------
# Scene Properties (Unchanged)
# -------------------------------------------------------------------
class ComfyUISceneProperties(bpy.types.PropertyGroup):
    user_prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Positive prompt for the ComfyUI workflow",
        default="cinematic, masterpiece, best quality, 1girl, blonde, wearing a jacket",
    )
    frame_mode: bpy.props.EnumProperty(
        items=[
            ("CURRENT", "Current Frame", "Use only the current frame"),
            ("RANGE", "Frame Range", "Specify a start and end frame"),
        ],
        default="CURRENT",
        name="Frame Mode",
        description="Choose whether to process a single frame or a range.",
    )
    frame_start: bpy.props.IntProperty(
        name="Start Frame",
        description="Starting frame number for range processing",
        default=1,
        min=0,
    )
    frame_end: bpy.props.IntProperty(
        name="End Frame",
        description="Ending frame number for range processing",
        default=10,
        min=0,
    )
    frame_rate: bpy.props.IntProperty(
        name="Video Frame Rate",
        description="Frame rate for the temporary video sent to ComfyUI",
        default=8,
        min=1,
        max=120,
    )
    controlnet_depth_strength: bpy.props.FloatProperty(
        name="Depth Strength",
        description="Strength of the depth controlnet effect",
        default=0.4,
        min=0.0,
        max=1.0,
    )
    invert_depth_input: bpy.props.BoolProperty(
        name="Invert Depth",
        description="Invert the depth input for the controlnet",
        default=False
    )

# -------------------------------------------------------------------
# Simple Operator to Cancel Modal
# -------------------------------------------------------------------
class WM_OT_ComfyUICancelModal(bpy.types.Operator):
    """Operator to signal cancellation to the running modal operator"""
    bl_idname = "wm.comfyui_cancel_modal"
    bl_label = "Cancel ComfyUI Operation"
    bl_description = "Stops the background ComfyUI process"
    bl_options = {'INTERNAL'} # Hide from search

    @classmethod
    def poll(cls, context):
        # Only show if the modal operator is actually running
        return getattr(context.window_manager, "comfyui_modal_operator_running", False)

    def execute(self, context):
        wm = context.window_manager
        if getattr(wm, "comfyui_modal_operator_running", False):
            print("Cancel button pressed. Signaling modal operator to stop.")
            wm.comfyui_modal_operator_running = False # Set the flag
            # The modal loop will check this flag and initiate cleanup
            self.report({'INFO'}, "ComfyUI operation cancellation requested.")
        else:
            self.report({'WARNING'}, "ComfyUI operation is not running.")
        return {'FINISHED'}

# -------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------
classes = (
    ComfyUIAddonPreferences,
    ComfyUISceneProperties,
    OBJECT_OT_run_comfyui_modal,
    PREFERENCES_OT_open_comfyui_addon_prefs,
    VIEW3D_PT_comfyui_panel,
    WM_OT_ComfyUICancelModal, # Register Cancel Operator
)


def register():
    print("-" * 30)
    print(f"Registering {bl_info['name']} Add-on (GPencil version)...")

    # Check essential dependencies first
    if not websocket:
        raise ImportError(
            f"Addon '{bl_info['name']}' requires the 'websocket-client' Python package. "
            f"Please install it. Attempted path: {packages_path}"
        )
    if not SPA_GP_AVAILABLE:
         raise ImportError(
             f"Addon '{bl_info['name']}' requires SPA Studios Grease Pencil features. "
             "Ensure the SPA addon/fork is installed and enabled."
         )

    # Check for ffmpeg (basic check) - Keep as warning
    try:
        startupinfo = None
        if os.name == "nt": # Windows specific: hide console
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            timeout=5,
            startupinfo=startupinfo,
        )
        print("  ffmpeg found.")
    except Exception as e:
        print(
            f"  Warning: ffmpeg check failed: {e}. Ensure ffmpeg is installed and in the system PATH."
        )

    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"  ERROR: Failed to register class {cls.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting to unregister partially registered classes...")
            unregister_on_error()
            raise

    # Add Scene Properties
    bpy.types.Scene.comfyui_props = bpy.props.PointerProperty(
        type=ComfyUISceneProperties
    )

    # Add WindowManager properties
    bpy.types.WindowManager.comfyui_modal_operator_running = bpy.props.BoolProperty(
        default=False
    )
    bpy.types.WindowManager.comfyui_modal_status = bpy.props.StringProperty(
        default="Idle"
    )
    bpy.types.WindowManager.comfyui_modal_progress = bpy.props.StringProperty(
        default=""
    )

    print(f"{bl_info['name']} Add-on registered successfully.")
    print("-" * 30)


# Separate unregister function for error handling (Unchanged)
def unregister_on_error():
    """Attempts to unregister classes without raising further exceptions during error recovery."""
    # Delete WindowManager properties
    try: del bpy.types.WindowManager.comfyui_modal_operator_running
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_status
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_progress
    except AttributeError: pass

    # Delete Scene properties
    try:
        if hasattr(bpy.types.Scene, "comfyui_props"):
            del bpy.types.Scene.comfyui_props
    except Exception as e:
        print(f"  Warning: Error deleting Scene.comfyui_props: {e}")

    # Unregister classes in reverse order
    for cls in reversed(classes):
        if hasattr(cls, "bl_rna"):
             try:
                 bpy.utils.unregister_class(cls)
             except Exception as e:
                 print(f"  Warning: Failed to unregister class {cls.__name__} during error recovery: {e}")

def unregister():
    print("-" * 30)
    print(f"Unregistering {bl_info['name']} Add-on (GPencil version)...")

    # Delete WindowManager properties
    try: del bpy.types.WindowManager.comfyui_modal_operator_running
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_status
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_progress
    except AttributeError: pass

    # Delete Scene properties
    try:
        if hasattr(bpy.types.Scene, "comfyui_props"):
            del bpy.types.Scene.comfyui_props
    except Exception as e:
        print(f"  Warning: Error deleting Scene.comfyui_props during unregister: {e}")

    # Unregister classes in reverse order
    for cls in reversed(classes):
        if hasattr(cls, "bl_rna"):
            try:
                bpy.utils.unregister_class(cls)
            except Exception as e:
                print(f"  Warning: Failed to unregister class {cls.__name__}: {e}")

    print(f"{bl_info['name']} Add-on unregistered.")
    print("-" * 30)


if __name__ == "__main__":
    # Allow running the script directly in Blender Text Editor for testing
    try:
        unregister()
    except Exception:
        pass # Ignore errors if not registered yet
    register()