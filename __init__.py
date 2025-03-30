import sys
import bpy
import os
import tempfile

# import requests # Keep for potential future use, though not used currently
import json
import urllib.request
import urllib.parse
import uuid
import time
import subprocess  # To run ffmpeg
import threading  # For modal operator background task
import queue  # For thread communication
import mathutils  # For Vector math
import math  # For FOV calculations
import shutil  # For robust directory removal

# ------------------------------------------------------------------
# 1) Adjust this path if necessary for websocket-client or other packages:
#    (Ensure websocket-client is installed here or use Blender's pip)
packages_path = "C:\\Users\\ASUS\\AppData\\Roaming\\Python\\Python311\\site-packages"  # MODIFY AS NEEDED
if packages_path not in sys.path:
    sys.path.insert(0, packages_path)

# Try importing websocket-client early to catch missing dependency
try:
    # NOTE: Consider using Blender's built-in pip module if available/preferred:
    # from bpy.utils import previews # (Need previews for ensure_pip usually)
    # import ensure_pip
    # ensure_pip.ensure_pip('websocket-client')
    import websocket  # Need to install: pip install websocket-client
except ImportError:
    print(
        "ERROR: Could not import 'websocket'. Ensure 'websocket-client' is installed "
        f"in the specified packages_path ('{packages_path}') or Blender's Python environment."
    )
    websocket = None  # Set to None to handle checks later
# ------------------------------------------------------------------

bl_info = {
    "name": "ComfyUI AnimateDiff Integration",  # Renamed for clarity
    "author": "Your Name (Modified for ComfyUI Modal)",
    "version": (2, 2),  # Incremented version
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ComfyUI",
    "description": (
        "Captures frames, compiles video, sends to ComfyUI via WebSocket (modal operator), "
        "and creates image planes fitted to camera view for results. Requires ffmpeg in PATH and websocket-client."
    ),
    "category": "Object",
    "warning": "Requires ffmpeg in PATH and 'websocket-client' Python package. Capturing frames may be slow.",
    "doc_url": "",  # Add URL if you have docs
    "tracker_url": "",  # Add URL for bug reports
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
      "text": "1girl, blonde, tying a jacket around her neck",
      "clip": [
        "110",
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
        "110",
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
      "seed": 546052001170593,
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
        "53",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "53": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "width": 1024,
      "height": 576,
      "crop": "disabled",
      "image": [
        "107",
        0
      ]
    },
    "class_type": "ImageScale",
    "_meta": {
      "title": "Upscale Image"
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
        "110",
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
        "53",
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
        "53",
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
      "value": false
    },
    "class_type": "PrimitiveBoolean",
    "_meta": {
      "title": "Invert input for depth"
    }
  }
}
"""


# -------------------------------------------------------------------
# ADD-ON PREFERENCES: ComfyUI Server Address
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
        box.label(text="- Requires 'ffmpeg' installed and in system PATH.")
        box.label(text="- Requires 'websocket-client' Python package.")
        box.label(text=f"  (Attempting to load from: {packages_path})")
        box.label(
            text="- Frame capture uses the scene's render settings (if in camera view) or OpenGL."
        )
        box.label(text="- Rendering frames can be slow and will pause Blender.")


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
    original_engine = (
        render.engine
    )  # Store engine if we change it (not currently changing)
    original_display_mode = None  # To restore viewport shading

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
        render.image_settings.file_format = "PNG"  # Ensure PNG format

        use_opengl = True  # Default to faster OpenGL
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
                use_opengl = False  # Render command was used
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
                with context.temp_override(
                    area=area, region=area.regions[-1]
                ):  # Find appropriate region
                    bpy.ops.render.opengl(write_still=True, view_context=True)
            else:
                print(
                    "Error: Could not find 3D Viewport space for OpenGL render. Cannot capture."
                )
                # Fallback might capture wrong view or fail
                raise RuntimeError(
                    "Cannot perform OpenGL capture without a valid 3D Viewport space."
                )

    except Exception as e:
        print(f"Error during viewport capture: {e}")
        raise  # Re-raise the exception to be caught by the operator
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
# HELPER: Create video from sequence of frames using ffmpeg
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
        "-y",  # Overwrite output without asking
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
            "libx264",  # Video codec
            "-crf",
            "18",  # Quality (lower is better, 18=visually lossless)
            "-preset",
            "fast",  # Encoding speed vs compression (faster is usually fine for temp)
            "-pix_fmt",
            "yuv420p",  # Pixel format for compatibility
            output_video_path,
        ]
    )

    print(f"Running ffmpeg command: {' '.join(command)}")
    try:
        startupinfo = None
        if os.name == "nt":  # Windows specific: hide console
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
            print("ffmpeg stderr:", result.stderr)  # Print stderr for warnings/info
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
# -------------------------------------------------------------------
def get_comfyui_results_from_history(prompt_id, server_address, target_node_id):
    """
    Fetches prompt history via HTTP and extracts image data for a specific node.
    Returns a list of image byte data, or None if history not found/complete
    or node output missing. Returns [] if node found but has no images.
    """
    images_data = None  # Default to None (history fetch failed or not ready)
    try:
        url = f"{server_address}/history/{prompt_id}"
        # print(f"Fetching history from: {url}") # DEBUG
        with urllib.request.urlopen(
            url, timeout=10
        ) as response:  # Add timeout to HTTP request
            if response.status == 200:
                history = json.loads(response.read())
                # History is a dict {prompt_id: {outputs: {node_id: {images: [...]}}}}
                prompt_data = history.get(prompt_id)
                if not prompt_data:
                    print(f"History found, but data for prompt {prompt_id} is missing.")
                    return None  # History exists but is incomplete?

                outputs = prompt_data.get("outputs")
                if not outputs:
                    # This is expected if the prompt hasn't finished generating outputs
                    # print(f"History for prompt {prompt_id} has no 'outputs' section (likely still running/processing).") # DEBUG - Can be noisy
                    return None  # Not finished processing outputs

                node_output = outputs.get(
                    str(target_node_id)
                )  # Node IDs in history keys are strings
                if (
                    node_output is None
                ):  # Check specifically for None, as empty dict is possible
                    print(
                        f"History outputs for prompt {prompt_id} do not contain node ID {target_node_id}."
                    )
                    # Log available nodes for debugging
                    print(f"Available output nodes in history: {list(outputs.keys())}")
                    return (
                        []
                    )  # Return empty list indicating node wasn't found in output

                images_info = node_output.get("images")
                if images_info is None:  # Check specifically for None
                    print(
                        f"Node {target_node_id} output found in history but has no 'images' key."
                    )
                    return []  # Node output exists but no images key
                if not images_info:
                    print(
                        f"Node {target_node_id} output found in history but the 'images' list is empty."
                    )
                    return []  # Node output exists but images list is empty

                print(
                    f"Found {len(images_info)} images for node {target_node_id} in history."
                )
                images_data = []  # Initialize list now that we expect images
                fetch_errors = 0
                fetch_start_time = time.time()
                for i, image_info in enumerate(images_info):
                    filename = image_info.get("filename")
                    subfolder = image_info.get(
                        "subfolder", ""
                    )  # Default to empty string if missing
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
                    images_data  # Return list (might be empty if fetch failed for all)
                )

            elif response.status == 404:
                # Expected if prompt ID doesn't exist or hasn't finished writing history yet
                # print(f"History API returned 404 Not Found for prompt {prompt_id}. Prompt likely hasn't finished or ID is wrong.") # DEBUG - Can be noisy
                return None
            else:
                print(
                    f"Error fetching history for prompt {prompt_id}: HTTP Status {response.status}"
                )
                return None  # Indicate history fetch failed

    except urllib.error.URLError as e:
        # Can happen if server down or history endpoint changes
        print(f"URL Error fetching history {prompt_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error processing history for {prompt_id}: {e}")
        return None
    except TimeoutError:
        print(f"Timeout Error fetching history {prompt_id}.")
        return None  # History check timed out
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
        )  # Increased timeout for queueing
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
        raise ConnectionError(error_message) from e  # Raise a more specific error
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
        ) as response:  # Increased timeout for image fetch
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
        return None  # Indicate failure
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
        images_info = data_source  # History function already returns the images list
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
        ws.connect(ws_url, timeout=20)  # Increased connection timeout
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

    images_data = None  # Use None to indicate not yet retrieved
    # execution_done_via_ws = False # Flag no longer strictly needed with direct assignment
    prompt_execution_finished_signal = (
        False  # Flag for the overall prompt completion signal (node=None)
    )
    output_node_executed_signal = (
        False  # Flag if specific output node finished message received
    )
    retrieved_via_history = False  # Flag to track if history fallback was used

    consecutive_timeouts = 0
    max_consecutive_timeouts_before_warn = (
        5  # How many timeouts before checking history proactively
    )
    max_consecutive_timeouts_overall = (
        20  # Give up after this many timeouts without progress
    )
    overall_timeout_seconds = 900  # Overall safety timeout (15 minutes)
    ws_receive_timeout = 15  # How long to wait for a single message

    start_time = time.time()
    last_message_time = start_time

    try:
        while images_data is None and (
            time.time() - start_time < overall_timeout_seconds
        ):
            try:
                ws.settimeout(ws_receive_timeout)
                out = ws.recv()
                consecutive_timeouts = 0  # Reset timeout counter on successful receive
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
                    if history_result:  # Not None or empty []
                        print("Final history check after max timeouts succeeded.")
                        # --- FIXED: Assign directly, history_result is already list of bytes ---
                        images_data = history_result
                        retrieved_via_history = True
                        break  # Exit loop
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
                    ):  # History fetch succeeded (result might be [] or list of bytes)
                        if history_result:  # Found images (list of bytes) in history
                            print(
                                f"Received {len(history_result)} images from /history API fallback."
                            )
                            # --- FIXED: Assign directly, history_result is already list of bytes ---
                            images_data = history_result
                            retrieved_via_history = True
                            break  # Exit WebSocket loop, we have results
                        else:
                            # History exists, but node/images not found or empty list returned
                            if prompt_execution_finished_signal:
                                print(
                                    f"/history API confirms prompt finished but node {output_node_id} has no image output."
                                )
                                images_data = []  # Mark as finished with no images
                                break
                            else:
                                # Prompt not finished according to history, or node not listed yet
                                # print("History doesn't have the final output images yet. Continuing WS listen.") # DEBUG
                                pass
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
                        if history_result:  # Not None or empty []
                            print("Final history check after WS ping close succeeded.")
                            # --- FIXED: Assign directly ---
                            images_data = history_result
                            retrieved_via_history = True
                        else:
                            images_data = []
                        break  # Exit loop
                    except Exception as ping_e:
                        print(f"Error sending WebSocket ping: {ping_e}")

                continue  # Go to next loop iteration after handling timeout

            except websocket.WebSocketConnectionClosedException:
                print("WebSocket connection closed by server.")
                # --- Try one last history check ---
                history_result = get_comfyui_results_from_history(
                    prompt_id, server_address, output_node_id
                )
                if history_result:  # Not None or empty []
                    print("Final history check after WS close succeeded.")
                    # --- FIXED: Assign directly ---
                    images_data = history_result
                    retrieved_via_history = True
                else:
                    images_data = []
                break  # Exit loop

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
                        node_id_msg == int(output_node_id)
                        and prompt_id_msg == prompt_id
                    ):
                        print(f"Output node {output_node_id} is executing...")
                        if progress_callback:
                            progress_callback(f"Node {output_node_id} running")

                elif msg_type == "executed":
                    prompt_id_msg = data.get("prompt_id")
                    node_id_msg = data.get("node_id")

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
                        break  # Exit loop, we have results via WS

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
            # --- FIXED: Assign directly ---
            images_data = history_result
            retrieved_via_history = True
        else:
            raise  # Re-raise if history also fails

    except TimeoutError as e:  # Catch specific timeout from inner logic
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
            # --- FIXED: Assign directly ---
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
        if history_result is not None:  # Check includes empty list []
            if history_result:  # List has contents (bytes)
                print(
                    f"Final history check successful, received {len(history_result)} images."
                )
                # --- FIXED: Assign directly ---
                images_data = history_result
                retrieved_via_history = True
            else:  # Empty list returned
                print(
                    "Final history check returned empty list (no images found for node)."
                )
                images_data = []
        else:  # History fetch returned None (error)
            print("Final history check failed.")
            images_data = []  # Assume no images if history fails at the end

    # If loop finished because of overall timeout (and not already handled by inner timeout logic)
    elif images_data is None and (time.time() - start_time >= overall_timeout_seconds):
        print(
            f"ERROR: Overall timeout ({overall_timeout_seconds}s) waiting for ComfyUI results for prompt {prompt_id}."
        )
        # Try one last desperate history check if not done by inner timeout logic
        history_result = get_comfyui_results_from_history(
            prompt_id, server_address, output_node_id
        )
        if history_result:  # Not None or empty []
            print(
                f"Final history check after overall timeout succeeded, received {len(history_result)} images."
            )
            # --- FIXED: Assign directly ---
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
# HELPER: Create a plane with image texture fitted to camera view
# -------------------------------------------------------------------
def create_plane_with_image_fit_camera(
    img: bpy.types.Image,
    context: bpy.types.Context,
    location: mathutils.Vector,  # Pass in calculated location
    rotation_euler: mathutils.Euler,  # Pass in calculated rotation
    name="ComfyUI Plane",
    frame_number=None,
    distance=5.0,  # Distance from camera origin where the plane is placed
):
    """
    Creates a plane, sets its transform based on pre-calculated location/rotation,
    and scales it to exactly fit the camera view at the specified distance.
    Applies image, keyframes visibility. Ensures Object mode.
    Returns the created plane object or None on failure.

    Uses FOV (Field of View) for Perspective cameras and ortho_scale for Orthographic.
    """
    if frame_number is None:
        frame_number = context.scene.frame_current

    cam = context.scene.camera
    if not cam or not cam.data or not isinstance(cam.data, bpy.types.Camera):
        print("  create_plane_fit_camera: Error - Valid scene camera not found.")
        return None
    cam_data = cam.data  # Get camera data
    scene = context.scene
    render = scene.render

    # --- Ensure Object Mode ---
    active_obj = context.active_object
    previous_mode = None
    if active_obj and active_obj.mode != "OBJECT":
        previous_mode = active_obj.mode
        try:
            bpy.ops.object.mode_set(mode="OBJECT", toggle=False)
        except Exception as e:
            print(f"  create_plane: Warning - Could not force Object mode: {e}")

    # --- Calculate Scale based on Camera Type and Distance ---
    try:
        # Get the camera's view frame in camera local space
        frame = cam_data.view_frame(scene=context.scene)
        bl, tl, tr, br = frame
        
        # Compute width and height in camera space
        width = (tr - tl).length
        height = (tl - bl).length
        
        # For perspective camera, we need to scale based on distance
        if cam_data.type == "PERSP":
            # Scale proportionally to the distance
            # Divide by camera scale to ensure consistent sizing
            scale_factor = distance
            width *= scale_factor
            height *= scale_factor
        
        # For orthographic camera, the size is consistent regardless of distance
        
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Calculated invalid dimensions ({width}x{height}). Check camera settings."
            )

    except Exception as e:
        print(f"  create_plane_fit_camera: Error calculating plane scale: {e}")
        # [Restore mode if changed]
        if previous_mode and active_obj and context.active_object == active_obj:
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except Exception:
                pass
        return None

    # --- Create Plane and Set Transform ---
    plane_obj = None
    try:
        # Create a plane of size 1x1 initially. We will scale it precisely.
        # Blender's default plane with size=1 runs from -0.5 to 0.5 (total 1 unit).
        bpy.ops.mesh.primitive_plane_add(
            size=1, # Create a 1x1 unit plane
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0), # Create at origin first
            rotation=(0, 0, 0),
        )
        plane_obj = context.active_object
        if not plane_obj:
            raise RuntimeError("Plane creation operator failed.")
        plane_obj.name = name

        # --- Apply Transform in order: Location, Rotation, Scale ---
        plane_obj.location = location
        plane_obj.rotation_euler = rotation_euler

        # Scale the 1x1 plane to match the calculated view dimensions
        # For planes in Blender, scaling by 2x the width/height gives correct results
        # because the default plane is 1x1 (from -0.5 to 0.5 in both X and Y)
        plane_obj.scale = (width, height, 1.0)

        context.view_layer.update() # Ensure updates are flushed
        print(
            f"  Created plane '{plane_obj.name}'. "
            f"Loc: {tuple(round(c, 2) for c in plane_obj.location)}, "
            f"Rot: {tuple(round(math.degrees(a), 1) for a in plane_obj.rotation_euler)}, "
            f"Scale: {tuple(round(s, 2) for s in plane_obj.scale)}"
        )

    except Exception as e:
        print(
            f"  create_plane_fit_camera: ERROR creating/transforming plane '{name}': {e}"
        )
        import traceback
        traceback.print_exc()
        if plane_obj and plane_obj.name in bpy.data.objects:
            bpy.data.objects.remove(plane_obj, do_unlink=True)
            plane_obj = None
        if previous_mode and active_obj and context.active_object == active_obj:
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except Exception:
                pass
        return None

    # --- Create Material ---
    try:
        mat_name = f"ComfyUIImageMat_{name[:30]}"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        # Ensure nodes are cleared ONLY if creating a new material or reusing is safe
        nodes.clear()
        output_node = nodes.new("ShaderNodeOutputMaterial")
        shader_node = nodes.new("ShaderNodeEmission") # Use Emission for exact color match
        tex_node = nodes.new("ShaderNodeTexImage")

        tex_node.image = img
        # Ensure image is packed if desired, or handle paths correctly
        # img.pack() # Optional: pack image into .blend file

        tex_node.interpolation = "Closest" # Use 'Closest' for pixel-perfect (if needed) or 'Linear'
        tex_node.location = (-300, 0)
        shader_node.location = (0, 0)
        output_node.location = (300, 0)

        links.new(tex_node.outputs["Color"], shader_node.inputs["Color"])
        links.new(shader_node.outputs["Emission"], output_node.inputs["Surface"])

        # Assign material to the plane's first slot
        if plane_obj.data.materials:
            plane_obj.data.materials[0] = mat
        else:
            plane_obj.data.materials.append(mat)
    except Exception as e:
        print(
            f"  create_plane_fit_camera: ERROR creating material for {plane_obj.name}: {e}"
        )
        # Optionally remove the plane if material creation fails critically
        # bpy.data.objects.remove(plane_obj, do_unlink=True)
        # return None


    # --- Keyframe Visibility ---
    try:
        # Helper to set interpolation for keyframes
        def set_constant_interpolation(obj, data_path):
            if obj.animation_data and obj.animation_data.action:
                for fcurve in obj.animation_data.action.fcurves:
                    if fcurve.data_path == data_path:
                        for kp in fcurve.keyframe_points:
                            kp.interpolation = "CONSTANT"
                        fcurve.update()
                        return # Found the fcurve

        # Ensure animation data exists
        if not plane_obj.animation_data:
            plane_obj.animation_data_create()
        if not plane_obj.animation_data.action:
            action_name = f"{plane_obj.name}_VisAction"
            plane_obj.animation_data.action = bpy.data.actions.new(name=action_name)

        # Keyframe: Hidden before the frame
        plane_obj.hide_viewport = True
        plane_obj.hide_render = True
        plane_obj.keyframe_insert(data_path="hide_viewport", frame=frame_number - 1)
        plane_obj.keyframe_insert(data_path="hide_render", frame=frame_number - 1)
        set_constant_interpolation(plane_obj, "hide_viewport")
        set_constant_interpolation(plane_obj, "hide_render")

        # Keyframe: Visible on the frame
        plane_obj.hide_viewport = False
        plane_obj.hide_render = False
        plane_obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
        plane_obj.keyframe_insert(data_path="hide_render", frame=frame_number)
        set_constant_interpolation(plane_obj, "hide_viewport")
        set_constant_interpolation(plane_obj, "hide_render")

        # Keyframe: Hidden after the frame
        plane_obj.hide_viewport = True
        plane_obj.hide_render = True
        plane_obj.keyframe_insert(data_path="hide_viewport", frame=frame_number + 1)
        plane_obj.keyframe_insert(data_path="hide_render", frame=frame_number + 1)
        set_constant_interpolation(plane_obj, "hide_viewport")
        set_constant_interpolation(plane_obj, "hide_render")

    except Exception as e:
        print(
            f"  create_plane_fit_camera: ERROR keyframing visibility for {plane_obj.name}: {e}"
        )

    # --- Restore Mode ---
    if previous_mode and active_obj and context.active_object == active_obj:
        try:
            bpy.ops.object.mode_set(mode=previous_mode)
        except Exception as e:
            print(f"  Could not restore previous mode '{previous_mode}': {e}")

    return plane_obj

# -------------------------------------------------------------------
# OPERATOR: Main Modal Operator
# -------------------------------------------------------------------
class OBJECT_OT_run_comfyui_modal(bpy.types.Operator):
    bl_idname = "object.run_comfyui_modal"
    bl_label = "Run ComfyUI Workflow (Modal)"
    bl_description = "Capture frames, create video, send to ComfyUI, create textured planes fitted to camera. Runs in background."
    bl_options = {"REGISTER", "UNDO"}

    # --- Configurable Node IDs ---
    video_path_node_id: bpy.props.StringProperty(
        name="Video Input Node ID", default="107"
    )
    prompt_node_id: bpy.props.StringProperty(name="Prompt Node ID", default="3")
    output_node_id: bpy.props.StringProperty(name="Image Output Node ID", default="12")

    # --- Modal State Variables ---
    _timer = None
    _thread = None
    _result_queue = None
    # --- ADDED: Thread-safe storage for status messages ---
    _thread_status: bpy.props.StringProperty(
        options={"SKIP_SAVE"}
    )  # Use props to ensure they exist
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
    previous_obj = None
    previous_mode = None
    final_result = None
    created_planes = []
    planes_parent = None
    output_img_paths = []
    server_address = ""

    # --- Worker Thread Function ---
    def _comfyui_worker_thread(self):
        """
        Function executed in a separate thread. Performs video creation,
        ComfyUI queuing, and waits for results. Communicates via queue and status vars.
        IMPORTANT: Do NOT interact with bpy data from here except setting simple instance vars.
        """
        try:
            # --- 2) Create Video ---
            # --- Update internal status ---
            self._thread_status = "Creating temporary video..."
            self._thread_progress = ""

            # Determine start number for ffmpeg
            start_num = self.frames_to_process[0] if self.frames_to_process else None
            # Construct video path
            video_filename = f"input_video_{self.frames_to_process[0]}.mp4"
            self.temp_video_path = os.path.join(self.temp_dir_path, video_filename)

            create_video_from_frames(
                self.temp_dir_path,
                self.temp_video_path,
                bpy.context.scene.comfyui_props.frame_rate,  # Get rate from scene props
                self.frame_pattern,
                start_number=start_num,
            )

            # --- 3) Modify and Queue Workflow ---
            abs_video_path = os.path.abspath(self.temp_video_path).replace("\\", "/")
            self.workflow[self.video_path_node_id]["inputs"]["video"] = abs_video_path
            self.workflow[self.prompt_node_id]["inputs"][
                "text"
            ] = bpy.context.scene.comfyui_props.user_prompt # Get from scene props

            # --- *** Get and Apply ControlNet Strengths *** ---
            # Read values from Scene Properties (safe to access bpy.context from thread?)
            # It's generally discouraged, but reading simple properties *might* be okay.
            # A safer alternative would be to pass these values from invoke/modal.
            # Let's stick to reading directly for now, but be aware.
            depth_strength = bpy.context.scene.comfyui_props.controlnet_depth_strength
            invert_depth = bpy.context.scene.comfyui_props.invert_depth_input

            # Node IDs identified from your workflow JSON:
            depth_apply_node_id = "114"
            depth_invert_node_id = "123"

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

            # --- *** END CONTROLNET STRENGTH UPDATE *** ---

            # --- Update internal status ---
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
            # --- Update internal status ---
            self._thread_status = (
                f"Waiting for ComfyUI results (Prompt: {prompt_id[:8]}...)"
            )
            self._thread_progress = ""

            # Define a callback for progress updates - THIS updates the internal status vars
            def progress_update(message):
                # Update the instance variables, which the modal loop will read
                self._thread_status = (
                    f"Waiting for ComfyUI results (Prompt: {prompt_id[:8]}...)"
                )
                self._thread_progress = message

            output_images_data = get_comfyui_images_ws(
                prompt_id,
                self.server_address,
                self.client_id,
                self.output_node_id,
                progress_callback=progress_update,  # Pass the callback
            )

            # --- Success ---
            # --- Update internal status ---
            self._thread_status = "Received results from ComfyUI."
            self._thread_progress = ""
            self._result_queue.put(output_images_data)  # Put list of image data

        except Exception as e:
            # --- Failure ---
            error_short = str(e).splitlines()[0]
            # --- Update internal status ---
            self._thread_status = "Error during ComfyUI interaction."
            self._thread_progress = f"{type(e).__name__}: {error_short}"
            print(f"Error in worker thread: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging
            self._result_queue.put(e)  # Put the exception object in the queue

    # --- Main Execution Logic (Called by Modal on Completion) ---
    def execute_finish(self, context):
        """
        Processes the results received from the worker thread. Creates image planes.
        Assumes it's called from the modal method when results are ready (main thread).
        """
        print("Entering Execute Finish method...")  # DEBUG
        wm = context.window_manager

        if isinstance(self.final_result, Exception):
            error_short = str(self.final_result).splitlines()[0]
            self.report(
                {"ERROR"},
                f"ComfyUI processing failed in worker thread: {type(self.final_result).__name__}: {error_short}",
            )
            # Update final status
            wm.comfyui_modal_status = "Finished with Error"
            wm.comfyui_modal_progress = (
                f"{type(self.final_result).__name__}: {error_short}"
            )
            return {"CANCELLED"}  # Error state
        elif (
            self.final_result is None
        ):  # Check specifically for None if history failed badly
            self.report({"WARNING"}, "No valid response received from ComfyUI.")
            wm.comfyui_modal_status = "Finished (No Response)"
            wm.comfyui_modal_progress = ""
            return {"FINISHED"}
        elif not isinstance(self.final_result, list):
            self.report(
                {"ERROR"},
                f"Received unexpected result type from worker: {type(self.final_result)}",
            )
            wm.comfyui_modal_status = "Finished with Error"
            wm.comfyui_modal_progress = (
                f"Internal error: Unexpected result type {type(self.final_result)}"
            )
            return {"CANCELLED"}
        elif not self.final_result:
            self.report({"INFO"}, "No images were generated or retrieved by ComfyUI.")
            wm.comfyui_modal_status = "Finished (No Images)"
            wm.comfyui_modal_progress = ""
            return {"FINISHED"}  # Finished, but with no results

        # --- Process images and create planes ---
        print(f"Execute Finish received {len(self.final_result)} image(s).")  # DEBUG

        self.report(
            {"INFO"}, f"Received {len(self.final_result)} image(s). Creating planes..."
        )
        wm.progress_begin(0, len(self.final_result))

        self.created_planes = []  # Reset list for this run
        self.output_img_paths = []  # Reset list for this run

        # --- Ensure Blender is in Object mode ---
        original_active = context.active_object
        original_mode = original_active.mode if original_active else "OBJECT"
        needs_mode_change = original_mode != "OBJECT"
        if needs_mode_change and original_active:
            print("Switching to Object mode for plane creation...")  # DEBUG
            try:
                context.view_layer.objects.active = (
                    original_active  # Ensure it's active
                )
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError as e:
                self.report(
                    {"ERROR"}, f"Could not set Object mode: {e}. Plane creation failed."
                )
                wm.progress_end()
                wm.comfyui_modal_status = "Finished with Error"
                wm.comfyui_modal_progress = f"Failed to set Object mode: {e}"
                return {"CANCELLED"}  # Cannot proceed without object mode

        # --- Create Parent Empty ---
        self.planes_parent = None
        try:
            parent_name = f"ComfyUI_Output_{self.client_id[:6]}"
            # Ensure we are in object mode before adding empty
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            bpy.ops.object.empty_add(
                type="PLAIN_AXES", location=context.scene.cursor.location
            )
            self.planes_parent = context.active_object
            self.planes_parent.name = parent_name
            print(f"Created parent empty: {parent_name}")  # DEBUG
        except Exception as e:
            print(f"Warning: Could not create parent empty: {e}")
            self.planes_parent = None

        # --- Get Scene Camera ONCE before the loop ---
        cam = context.scene.camera
        if not cam:
            self.report(
                {"ERROR"}, "No active scene camera found. Cannot create planes."
            )
            wm.progress_end()
            wm.comfyui_modal_status = "Finished with Error"
            wm.comfyui_modal_progress = "No scene camera"
            # Need to clean up potentially created parent empty? Unlikely but possible
            if self.planes_parent and self.planes_parent.name in bpy.data.objects:
                bpy.data.objects.remove(self.planes_parent, do_unlink=True)
            return {"CANCELLED"}

        # Define plane distance (needed for location AND scaling now)
        plane_distance = 5.0  # Or make this a user setting

        # --- Create Planes Loop ---
        success_count = 0
        num_results = len(self.final_result)
        num_frames_requested = len(self.frames_to_process)

        if num_results != num_frames_requested:
            print(
                f"Warning: Received {num_results} images, but requested {num_frames_requested} frames. "
                "Planes will be created based on received images, frame matching might be approximate."
            )

        original_scene_frame = context.scene.frame_current  # Store original frame

        for i, img_data in enumerate(self.final_result):
            # --- Update status for this stage ---
            current_status = f"Processing output {i+1}/{num_results}"
            current_progress = ""  # Reset progress for this step
            wm.comfyui_modal_status = current_status
            wm.comfyui_modal_progress = current_progress
            # Force redraw might be needed if status bar doesn't update quickly enough
            context.workspace.status_text_set(f"ComfyUI: {current_status}")

            if not isinstance(img_data, bytes) or not img_data:
                self.report({"WARNING"}, f"Skipping invalid image data at index {i}.")
                wm.progress_update(i + 1)
                continue

            # --- Determine frame number ---
            if i < num_frames_requested:
                frame_num = self.frames_to_process[i]
            else:
                frame_num = self.original_frame + i  # Fallback frame number
                print(f"  Using fallback frame number {frame_num} for image index {i}")

            wm.comfyui_modal_progress = f"Frame {frame_num}"  # Update progress detail
            context.workspace.status_text_set(
                f"ComfyUI: {current_status} (Frame {frame_num})"
            )

            # --- Save and Load Image ---
            output_img_filename = f"comfy_out_{frame_num:04d}.png"
            if not self.temp_dir_path or not os.path.isdir(self.temp_dir_path):
                self.report(
                    {"ERROR"},
                    f"Temporary directory missing during image saving ({self.temp_dir_path}). Aborting plane creation.",
                )
                wm.comfyui_modal_status = "Finished with Error"
                wm.comfyui_modal_progress = "Temporary directory lost"
                break  # Stop processing further images

            output_img_path = os.path.join(self.temp_dir_path, output_img_filename)
            print(f"\nProcessing image {i+1}/{num_results} for frame {frame_num}...")

            try:
                print(f"  Saving image data to: {output_img_path}")
                with open(output_img_path, "wb") as f:
                    f.write(img_data)
                if (
                    not os.path.exists(output_img_path)
                    or os.path.getsize(output_img_path) == 0
                ):
                    raise IOError(
                        "Temporary image file not found or empty after writing."
                    )
                print(f"  Image data saved successfully.")
                self.output_img_paths.append(output_img_path)

                print(f"  Loading image into Blender: {output_img_path}")
                processed_image = bpy.data.images.load(
                    output_img_path, check_existing=True
                )
                if not processed_image:
                    self.report(
                        {"ERROR"},
                        f"CRITICAL: Failed to load temporary image into Blender: {output_img_path}",
                    )
                    if os.path.exists(output_img_path):
                        try:
                            os.remove(output_img_path)
                        except OSError as rem_e:
                            print(f"    Could not remove failed temp image: {rem_e}")
                    if output_img_path in self.output_img_paths:
                        self.output_img_paths.remove(output_img_path)
                    wm.progress_update(i + 1)
                    continue
                processed_image.name = output_img_filename
                print(
                    f"  Image '{processed_image.name}' loaded (Size: {processed_image.size[0]}x{processed_image.size[1]})."
                )

                # --- *** GET EVALUATED CAMERA TRANSFORM FOR THIS FRAME *** ---
                context.scene.frame_set(frame_num)  # Set frame for evaluation
                depsgraph = context.evaluated_depsgraph_get()
                evaluated_cam = cam.evaluated_get(depsgraph)
                cam_matrix = evaluated_cam.matrix_world

                # Calculate target location and rotation from the evaluated matrix
                target_location = cam_matrix @ mathutils.Vector((0, 0, (-plane_distance)-2))
                target_rotation = cam_matrix.to_euler(
                    "XYZ"
                )  # Use 'XYZ' or preferred order
                # --- *** END EVALUATED TRANSFORM GET *** ---

                # --- Create plane ---
                plane_name = f"ComfyUI Plane (Frame {frame_num})"
                print(
                    f"  Calling create_plane_with_image_fit_camera for '{plane_name}'..."
                )

                # --- Call the updated function (WITH distance arg) ---
                new_plane = create_plane_with_image_fit_camera(
                    processed_image,
                    context,
                    location=target_location,  # Pass calculated location
                    rotation_euler=target_rotation,  # Pass calculated rotation
                    name=plane_name,
                    frame_number=frame_num,
                    distance=plane_distance,  # Pass distance for scaling
                )

                if new_plane:
                    print(f"  Successfully created plane '{new_plane.name}'.")
                    self.created_planes.append(new_plane)
                    success_count += 1
                    # --- Parent to empty ---
                    if (
                        self.planes_parent
                        and self.planes_parent.name in context.view_layer.objects
                    ):
                        print(
                            f"    Parenting '{new_plane.name}' to '{self.planes_parent.name}'."
                        )
                        context.view_layer.update()  # Ensure matrices are up-to-date
                        original_matrix = new_plane.matrix_world.copy()
                        new_plane.parent = self.planes_parent
                        new_plane.matrix_world = original_matrix  # Keep original world transform after parenting
                else:
                    self.report(
                        {"WARNING"},
                        f"Function create_plane_with_image_fit_camera failed for frame {frame_num}",
                    )

                wm.progress_update(i + 1)

            except Exception as e:
                self.report(
                    {"ERROR"},
                    f"Failed processing output image {i} (frame {frame_num}): {e}",
                )
                import traceback

                traceback.print_exc()
                wm.progress_update(i + 1)
                # Continue with the next image

        # --- Restore original frame ---
        # Check if scene frame exists before setting
        if context.scene.frame_start <= original_scene_frame <= context.scene.frame_end:
            context.scene.frame_set(original_scene_frame)
        else:
            # Set to start frame if original is invalid? Or just leave it? Leave it for now.
            print(
                f"  Original scene frame {original_scene_frame} no longer valid, not restoring."
            )

        wm.progress_end()

        # --- Restore original mode ---
        if (
            needs_mode_change
            and original_active
            and original_active.name in context.view_layer.objects
        ):
            print(
                f"Restoring original mode ({original_mode}) for object '{original_active.name}'..."
            )
            try:
                if context.mode == "OBJECT":
                    context.view_layer.objects.active = original_active
                    # Check if mode is valid before setting
                    valid_modes = {
                        "MESH": {
                            "OBJECT",
                            "EDIT",
                            "VERTEX_PAINT",
                            "WEIGHT_PAINT",
                            "TEXTURE_PAINT",
                            "SCULPT",
                        }
                    }  # Add other types if needed
                    if (
                        original_active.type in valid_modes
                        and original_mode in valid_modes[original_active.type]
                    ):
                        bpy.ops.object.mode_set(mode=original_mode)
                        print(f"  Restored mode to {original_mode}")
                    else:
                        print(
                            f"  Cannot restore mode: '{original_mode}' is not valid for object type '{original_active.type}'. Staying in Object mode."
                        )
                else:
                    print(
                        f"  Cannot restore mode - not currently in Object mode (current: {context.mode})."
                    )
            except Exception as e:
                print(f"Could not restore original mode {original_mode}: {e}")
        elif needs_mode_change:
            print(
                f"Skipping mode restoration: Original object '{original_active.name if original_active else 'None'}' not found or wasn't active."
            )

        final_op_status = (
            f"Successfully created {success_count} / {num_results} planes."
        )
        self.report({"INFO"}, final_op_status)
        wm.comfyui_modal_status = "Finished"
        wm.comfyui_modal_progress = final_op_status
        print("Exiting Execute Finish method.")
        return {"FINISHED"}

    # --- Modal Method ---
    def modal(self, context, event):
        wm = context.window_manager

        # Force UI update (panel status)
        if context.area:
            context.area.tag_redraw()

        # Handle cancellation
        if event.type == "ESC" or not wm.comfyui_modal_operator_running:
            print("ESC pressed or operator cancelled externally.")
            # Ensure cleanup runs even if thread is still going (thread join happens in finish_or_cancel)
            return self.finish_or_cancel(context, cancelled=True)

        # Process timer events
        if event.type == "TIMER":
            # --- Update UI Status from internal thread variables ---
            # Read the latest status from the variables set by the worker thread/callback
            current_status = self._thread_status
            current_progress = self._thread_progress

            # Update WindowManager properties (which the panel reads)
            if wm.comfyui_modal_status != current_status:
                wm.comfyui_modal_status = current_status
            if wm.comfyui_modal_progress != current_progress:
                wm.comfyui_modal_progress = current_progress

            # Update status bar at bottom of Blender window
            status_bar_text = f"ComfyUI: {current_status}"
            if current_progress:
                status_bar_text += f" ({current_progress})"
            context.workspace.status_text_set(status_bar_text)
            # --- End UI Update ---

            # --- Check if thread has finished ---
            if self._thread and not self._thread.is_alive():
                print("Worker thread finished.")
                try:
                    self.final_result = self._result_queue.get_nowait()
                    print(f"Result from queue: {type(self.final_result)}")
                except queue.Empty:
                    self.report({"ERROR"}, "Thread finished but result queue is empty.")
                    self.final_result = RuntimeError("Thread finished but queue empty.")
                    # Update status to reflect error
                    wm.comfyui_modal_status = "Finished with Error"
                    wm.comfyui_modal_progress = "Internal error: Queue empty"
                except Exception as e:
                    self.report({"ERROR"}, f"Error retrieving result from queue: {e}")
                    self.final_result = e
                    # Update status to reflect error
                    wm.comfyui_modal_status = "Finished with Error"
                    wm.comfyui_modal_progress = (
                        f"Internal error: Queue retrieval failed {type(e).__name__}"
                    )

                # Call the main execution logic (now runs in main thread)
                final_status = self.execute_finish(context)
                # Finish the modal operator (finish_or_cancel handles cleanup)
                return self.finish_or_cancel(
                    context, cancelled=("CANCELLED" in final_status)
                )

            # If thread still running, just continue modal loop
            elif self._thread:
                pass  # Status updated above

        # Allow other events (like navigation) to pass through
        return {"PASS_THROUGH"}

    # --- Invoke Method (Starts the process) ---
    def invoke(self, context, event):
        wm = context.window_manager
        if wm.comfyui_modal_operator_running:
            self.report({"WARNING"}, "Operation already in progress.")
            return {"CANCELLED"}

        if not websocket:
            self.report(
                {"ERROR"},
                "websocket-client package not found or failed to import. Please install it (see console/preferences).",
            )
            return {"CANCELLED"}

        # --- Initial Setup & Validation ---
        # Initialize internal status variables
        self._thread_status = "Initializing..."
        self._thread_progress = ""

        prefs = context.preferences.addons[__name__].preferences
        self.server_address = prefs.comfyui_address.strip()
        if not self.server_address or not self.server_address.startswith(
            ("http://", "https://")
        ):
            self.report(
                {"ERROR"},
                "ComfyUI server address is not set correctly in add-on preferences.",
            )
            bpy.ops.preferences.open_comfyui_addon_prefs("INVOKE_DEFAULT")
            return {"CANCELLED"}

        # --- Get Settings from Scene Properties ---
        scene_props = context.scene.comfyui_props
        # We read these directly in the worker thread now, no need to copy here
        # self.user_prompt = scene_props.user_prompt
        frame_mode = scene_props.frame_mode
        frame_start = scene_props.frame_start
        frame_end = scene_props.frame_end
        # frame_rate = scene_props.frame_rate # Read in worker thread

        # Store state
        self.original_frame = context.scene.frame_current
        self.previous_obj = context.active_object
        self.previous_mode = self.previous_obj.mode if self.previous_obj else "OBJECT"

        # Determine frames
        if frame_mode == "CURRENT":
            self.frames_to_process = [self.original_frame]
        else:  # RANGE
            if frame_start > frame_end:
                self.report(
                    {"ERROR"}, "Start frame must be less than or equal to end frame."
                )
                return {"CANCELLED"}
            self.frames_to_process = list(range(frame_start, frame_end + 1))

        if not self.frames_to_process:
            self.report({"WARNING"}, "No frames selected for processing.")
            return {"CANCELLED"}

        # --- Create Temp Directory Safely ---
        try:
            self.temp_dir_obj = None
            self.temp_dir_path = None
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="blender_comfy_")
            self.temp_dir_path = self.temp_dir_obj.name
            print(f"Using temporary directory: {self.temp_dir_path}")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to create temp directory: {e}")
            self._cleanup_temp_files()
            return {"CANCELLED"}

        # --- SYNCHRONOUS Frame Capture ---
        self._thread_status = "Capturing frames..."  # Update status before blocking
        self._thread_progress = ""
        wm.comfyui_modal_status = self._thread_status  # Set initial WM status
        wm.comfyui_modal_progress = self._thread_progress
        context.workspace.status_text_set(
            f"ComfyUI: {self._thread_status} (This may take time)"
        )  # Update status bar

        self.report(
            {"INFO"},
            f"Capturing {len(self.frames_to_process)} frame(s)... (This may take time)",
        )
        wm.progress_begin(0, len(self.frames_to_process))
        self.frame_paths = []
        capture_success = True

        # Ensure Object Mode for capture stability
        mode_switched_for_capture = False
        if self.previous_mode != "OBJECT" and self.previous_obj:
            try:
                context.view_layer.objects.active = self.previous_obj
                bpy.ops.object.mode_set(mode="OBJECT")
                mode_switched_for_capture = True
                print("Switched to Object mode for capture.")
            except Exception as e:
                self.report(
                    {"WARNING"}, f"Could not switch to Object mode before capture: {e}"
                )

        current_frame_capture_start = time.time()
        for i, frame_num in enumerate(self.frames_to_process):
            frame_start_time = time.time()
            context.scene.frame_set(frame_num)
            # Update progress detail in internal var (modal loop will pick it up later)
            self._thread_progress = f"{i+1}/{len(self.frames_to_process)}"
            wm.progress_update(i)  # Update Blender's progress bar immediately

            # Force viewport update
            context.view_layer.update()

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
                capture_success = False
                break  # Stop capturing on first error

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
                bpy.ops.object.mode_set(mode=self.previous_mode)
            except Exception as e:
                print(f"Could not restore mode after capture: {e}")

        if not capture_success or not self.frame_paths:
            self.report({"ERROR"}, "Frame capture failed. Aborting.")
            self._cleanup_temp_files()
            # Reset WM status on failure
            wm.comfyui_modal_operator_running = False
            wm.comfyui_modal_status = "Idle"
            wm.comfyui_modal_progress = ""
            context.workspace.status_text_set(None)
            return {"CANCELLED"}
        # --- End Synchronous Capture ---

        # --- Prepare Workflow ---
        try:
            self.workflow = json.loads(COMFYUI_WORKFLOW_JSON)
            if self.video_path_node_id not in self.workflow:
                raise ValueError(
                    f"Workflow missing video input node ID: {self.video_path_node_id}"
                )
            if self.prompt_node_id not in self.workflow:
                raise ValueError(
                    f"Workflow missing prompt node ID: {self.prompt_node_id}"
                )
            if self.output_node_id not in self.workflow:
                raise ValueError(
                    f"Workflow missing expected output node ID: {self.output_node_id}"
                )
            if (
                "inputs" not in self.workflow[self.video_path_node_id]
                or "inputs" not in self.workflow[self.prompt_node_id]
            ):
                raise ValueError(
                    "Specified video or prompt node is missing 'inputs' dictionary."
                )

        except Exception as e:
            self.report({"ERROR"}, f"Failed to load/validate workflow JSON: {e}")
            self._cleanup_temp_files()
            wm.comfyui_modal_operator_running = False  # Reset state
            wm.comfyui_modal_status = "Idle"
            wm.comfyui_modal_progress = ""
            context.workspace.status_text_set(None)
            return {"CANCELLED"}

        # --- Start Background Thread ---
        self._result_queue = queue.Queue()
        self.client_id = str(uuid.uuid4())
        self._thread = threading.Thread(
            target=self._comfyui_worker_thread, name="ComfyUI_Worker"
        )
        self._thread.daemon = True

        # Set initial status via WindowManager & internal vars
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

        self.bl_options.add("RUNNING_MODAL")
        return {"RUNNING_MODAL"}

    # --- Cleanup and State Restoration ---
    # [finish_or_cancel method remains mostly the same]
    def finish_or_cancel(self, context, cancelled=False):
        """Cleans up resources and restores Blender state."""
        print(f"Finishing or cancelling operation (Cancelled: {cancelled})")
        wm = context.window_manager

        # Remove modal timer only if it exists
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
                print("Modal timer removed.")  # DEBUG
            except Exception as timer_e:  # Catch potential errors during removal
                print(f"Warning: Error removing modal timer: {timer_e}")  # DEBUG
            finally:
                self._timer = None  # Ensure timer is set to None even if removal failed

        # Clear status bar and WM state (do this early)
        context.workspace.status_text_set(None)
        wm.comfyui_modal_operator_running = False
        # Don't reset status/progress immediately if finished successfully, keep final message
        if cancelled:
            wm.comfyui_modal_status = "Cancelled"
            wm.comfyui_modal_progress = ""
        # If not cancelled, execute_finish should have set the final status

        # Ensure thread is finished (give it a moment)
        if self._thread and self._thread.is_alive():
            print(
                "Warning: Worker thread still alive during finish/cancel. Attempting join..."
            )  # DEBUG
            self._thread.join(timeout=5.0)  # Wait max 5 seconds
            if self._thread.is_alive():
                print("ERROR: Worker thread did not terminate cleanly!")  # DEBUG
        self._thread = None

        # Cleanup temporary files and directory
        self._cleanup_temp_files()  # Use the helper

        # Restore Blender state (frame, selection, mode)
        try:
            # --- Restore frame first ---
            if (
                hasattr(self, "original_frame")
                and context.scene.frame_current != self.original_frame
            ):
                if (
                    context.scene.frame_start
                    <= self.original_frame
                    <= context.scene.frame_end
                ):
                    print(f"Restoring scene frame to {self.original_frame}")  # DEBUG
                    context.scene.frame_set(self.original_frame)
                else:
                    print(
                        f"Skipping frame restoration: Original frame {self.original_frame} outside scene range {context.scene.frame_start}-{context.scene.frame_end}."
                    )

            # --- Restore selection and mode ---
            prev_obj = getattr(self, "previous_obj", None)
            prev_mode = getattr(self, "previous_mode", None)
            obj_exists = prev_obj and prev_obj.name in context.view_layer.objects

            if obj_exists:
                print(
                    f"Attempting to restore selection to: {prev_obj.name} and mode to {prev_mode}"
                )  # DEBUG
                try:
                    if context.mode != "OBJECT":
                        bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    context.view_layer.objects.active = prev_obj
                    prev_obj.select_set(True)
                    if prev_mode and prev_mode != context.active_object.mode:
                        valid_modes = {
                            "MESH": {
                                "OBJECT",
                                "EDIT",
                                "VERTEX_PAINT",
                                "WEIGHT_PAINT",
                                "TEXTURE_PAINT",
                                "SCULPT",
                            }
                        }
                        if (
                            prev_obj.type in valid_modes
                            and prev_mode in valid_modes[prev_obj.type]
                        ):
                            bpy.ops.object.mode_set(mode=prev_mode)
                            print(f"  Restored mode to: {prev_mode}")  # DEBUG
                        else:
                            print(
                                f"  Cannot restore mode: '{prev_mode}' is not valid for object type '{prev_obj.type}'."
                            )
                except Exception as restore_e:
                    print(
                        f"  Warning: Could not fully restore selection/mode: {restore_e}"
                    )

            elif (
                self.planes_parent
                and self.planes_parent.name in context.view_layer.objects
            ):
                print(
                    "Previous object invalid or gone, selecting created parent empty instead."
                )  # DEBUG
                try:
                    if context.mode != "OBJECT":
                        bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    context.view_layer.objects.active = self.planes_parent
                    self.planes_parent.select_set(True)
                    print(f"  Selected parent empty '{self.planes_parent.name}'.")
                except Exception as select_e:
                    print(f"  Warning: Could not select parent empty: {select_e}")

        except Exception as e:
            self.report({"WARNING"}, f"Error during state restoration: {e}")

        # Remove RUNNING_MODAL from options if it was added
        if "RUNNING_MODAL" in self.bl_options:
            self.bl_options.remove("RUNNING_MODAL")

        self.report(
            {"INFO"}, f"ComfyUI operation {'cancelled' if cancelled else 'finished'}."
        )
        print("-" * 30)  # Separator for logs
        return {"CANCELLED"} if cancelled else {"FINISHED"}

    # [_cleanup_temp_files method remains the same]
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

        self.frame_paths = []
        self.output_img_paths = []
        self.temp_video_path = None


# -------------------------------------------------------------------
# HELPER OPERATOR: Open this addon's preferences
# -------------------------------------------------------------------
class PREFERENCES_OT_open_comfyui_addon_prefs(bpy.types.Operator):
    """Opens the preferences specific to this addon"""

    bl_idname = "preferences.open_comfyui_addon_prefs"  # More specific ID
    bl_label = "Open ComfyUI Addon Preferences"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return __name__ in context.preferences.addons

    def execute(self, context):
        try:
            bpy.ops.screen.userpref_show("INVOKE_DEFAULT")  # Open preferences window
            # Explicitly activate the Add-ons tab and filter
            # Need slight delay for window manager to update? Sometimes needed.
            # Use addon name from bl_info for filtering
            bpy.context.preferences.active_section = "ADDONS"
            context.window_manager.windows[-1].screen.areas[
                -1
            ].spaces.active.filter_text = bl_info["name"]
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
    bl_label = "ComfyUI Gen"
    bl_idname = "VIEW3D_PT_comfyui_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ComfyUI"  # Tab name in the sidebar

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        scene_props = context.scene.comfyui_props

        # Get running state from WindowManager
        is_running = getattr(wm, "comfyui_modal_operator_running", False)

        # --- Main Operator Button ---
        row = layout.row()
        # Pass scene properties to the operator when invoked
        op = row.operator(
            OBJECT_OT_run_comfyui_modal.bl_idname, icon="IMAGE_DATA", text="Run ComfyUI"
        )
        # Operator will read properties from scene_props during its invoke

        # Disable button if already running
        row.enabled = not is_running

        # --- Display Status if Running ---
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
            # Add a cancel button - This requires the operator to check a cancel flag
            # layout.operator("object.cancel_comfyui_modal", icon='X', text="Cancel") # Need separate cancel operator

        # --- Operator Settings (Only editable when not running) ---
        col = layout.column(align=True)
        col.enabled = not is_running

        # Use Scene properties directly in the panel for persistent UI state
        col.prop(scene_props, "user_prompt")
        col.separator()
        col.prop(scene_props, "frame_mode")

        if scene_props.frame_mode == "RANGE":
            box = col.box()
            row = box.row(align=True)
            row.prop(scene_props, "frame_start", text="Start")
            row.prop(scene_props, "frame_end", text="End")
            # Display current scene frame range for reference
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
# Scene Properties (for persistent UI settings)
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
        default=8,  # Match default in VHS_VideoCombine if relevant?
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
# Registration
# -------------------------------------------------------------------
classes = (
    ComfyUIAddonPreferences,
    ComfyUISceneProperties,  # Register PropertyGroup
    OBJECT_OT_run_comfyui_modal,
    PREFERENCES_OT_open_comfyui_addon_prefs,
    VIEW3D_PT_comfyui_panel,
)


def register():
    print("-" * 30)
    print(f"Registering {bl_info['name']} Add-on...")

    # Check dependencies
    if not websocket:
        # Raise error during registration to prevent activation
        raise ImportError(
            f"Addon '{bl_info['name']}' requires the 'websocket-client' Python package. "
            "Please install it (e.g., 'pip install websocket-client') in the Python environment Blender uses, "
            f"or configure the 'packages_path' in the script. Attempted path: {packages_path}"
        )

    # Check for ffmpeg (basic check)
    try:
        startupinfo = None
        if os.name == "nt":  # Windows specific: hide console
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
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        print(
            f"  Warning: ffmpeg check failed: {e}. Ensure ffmpeg is installed and in the system PATH."
        )
        # Don't raise error, allow addon but warn user via bl_info and preferences

    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            # print(f"  Registered class: {cls.__name__}") # DEBUG
        except Exception as e:
            print(f"  ERROR: Failed to register class {cls.__name__}: {e}")
            # --- ADDED: Print detailed traceback for registration errors ---
            import traceback

            traceback.print_exc()
            # Attempt to unregister already registered classes on failure
            # Call unregister directly without catching exceptions here to see if it reveals more
            print("Attempting to unregister partially registered classes...")
            unregister_on_error()  # Use a separate function to avoid recursive error loops
            raise  # Re-raise the original exception to indicate failure

    # Add Scene Properties to bpy.types.Scene
    bpy.types.Scene.comfyui_props = bpy.props.PointerProperty(
        type=ComfyUISceneProperties
    )

    # Add WindowManager properties for modal state communication
    # Use annotations for default values
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


# --- ADDED: Separate unregister function for error handling ---
def unregister_on_error():
    """Attempts to unregister classes without raising further exceptions during error recovery."""
    # Delete WindowManager properties
    try:
        del bpy.types.WindowManager.comfyui_modal_operator_running
    except AttributeError:
        pass
    try:
        del bpy.types.WindowManager.comfyui_modal_status
    except AttributeError:
        pass
    try:
        del bpy.types.WindowManager.comfyui_modal_progress
    except AttributeError:
        pass

    # Delete Scene properties
    try:
        if hasattr(bpy.types.Scene, "comfyui_props"):  # Check before deleting
            del bpy.types.Scene.comfyui_props
    except Exception as e:
        print(f"  Warning: Error deleting Scene.comfyui_props: {e}")

    # Unregister classes in reverse order
    for cls in reversed(classes):
        # Check if class is actually registered before attempting unregister
        if hasattr(bpy.types, cls.__name__) or hasattr(
            bpy, cls.__name__
        ):  # Basic check
            try:
                bpy.utils.unregister_class(cls)
                # print(f"  Unregistered class during error recovery: {cls.__name__}") # DEBUG
            except Exception as e:
                print(
                    f"  Warning: Failed to unregister class {cls.__name__} during error recovery: {e}"
                )


def unregister():
    print("-" * 30)
    print(f"Unregistering {bl_info['name']} Add-on...")

    # Delete WindowManager properties
    try:
        del bpy.types.WindowManager.comfyui_modal_operator_running
    except AttributeError:
        pass
    try:
        del bpy.types.WindowManager.comfyui_modal_status
    except AttributeError:
        pass
    try:
        del bpy.types.WindowManager.comfyui_modal_progress
    except AttributeError:
        pass

    # Delete Scene properties
    try:
        # Check before deleting to avoid errors if registration failed partially
        if hasattr(bpy.types.Scene, "comfyui_props"):
            del bpy.types.Scene.comfyui_props
    except Exception as e:
        print(f"  Warning: Error deleting Scene.comfyui_props during unregister: {e}")

    # Unregister classes in reverse order
    for cls in reversed(classes):
        # Check if class is actually registered before attempting unregister
        # A more robust check using bl_rna
        if hasattr(cls, "bl_rna"):
            try:
                bpy.utils.unregister_class(cls)
                # print(f"  Unregistered class: {cls.__name__}") # DEBUG
            except Exception as e:
                print(f"  Warning: Failed to unregister class {cls.__name__}: {e}")
        # else:
        #      print(f"  Skipping unregister for {cls.__name__}, likely wasn't registered.") # DEBUG

    print(f"{bl_info['name']} Add-on unregistered.")
    print("-" * 30)


if __name__ == "__main__":
    # Allow running the script directly in Blender Text Editor for testing
    # Unregister first if script was reloaded
    try:
        unregister()
    except Exception:
        pass  # Ignore errors if not registered yet
    register()
