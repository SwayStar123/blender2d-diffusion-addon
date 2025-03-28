# -*- coding: utf-8 -*-

import sys
import bpy
import os
import tempfile
import requests # Keep for potential future use, though not used currently
import json
import urllib.request
import urllib.parse
import uuid
import time
import subprocess # To run ffmpeg
import threading # For modal operator background task
import queue     # For thread communication
import mathutils # For Vector math
# ------------------------------------------------------------------
# 1) Adjust this path if necessary for websocket-client or other packages:
#    (Ensure websocket-client is installed here)
packages_path = "C:\\Users\\ASUS\\AppData\\Roaming\\Python\\Python311\\site-packages"
if packages_path not in sys.path:
    sys.path.insert(0, packages_path)

# Try importing websocket-client early to catch missing dependency
try:
    import websocket # Need to install: pip install websocket-client
except ImportError:
    # This error will likely be raised properly during registration,
    # but we can make a note here.
    print("ERROR: Could not import 'websocket'. Ensure 'websocket-client' is installed in the specified packages_path or Blender's Python environment.")
    websocket = None # Set to None to handle checks later
# ------------------------------------------------------------------

bl_info = {
    "name": "ComfyUI AnimateDiff Integration", # Renamed for clarity
    "author": "Your Name (Modified for ComfyUI Modal)",
    "version": (2, 1), # Incremented version
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > ComfyUI",
    "description": (
        "Captures frames, compiles video, sends to ComfyUI via WebSocket (modal operator), "
        "and creates image planes for results. Requires ffmpeg in PATH and websocket-client."
    ),
    "category": "Object",
    "warning": "Requires ffmpeg in PATH and 'websocket-client' Python package. Capturing frames may be slow.",
}

# -------------------------------------------------------------------
# ComfyUI Workflow JSON (Embed the specific workflow here)
# -------------------------------------------------------------------
# MAKE SURE this JSON is valid and correctly represents your desired workflow.
# Node IDs are crucial. We'll modify "107" (VHS_LoadVideoPath), "3" (Positive Prompt),
# and fetch results based on "12" (SaveImage).
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
      "text": "PLACEHOLDER_PROMPT",
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
      "text": "ugly, deformed, bad lighting, blurry, text, watermark, extra hands, bad quality, deformed hands, deformed fingers, nostalgic, drawing, painting, bad anatomy, worst quality, blurry, blurred, normal quality, bad focus, tripod, three legs, weird legs, short legs",
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
      "seed": 380620886430907,
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
        "117",
        0
      ],
      "negative": [
        "117",
        1
      ],
      "latent_image": [
        "56",
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
  "56": {
    "inputs": {
      "pixels": [
        "53",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
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
        "102",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "97": {
    "inputs": {
      "control_net_name": "control_v11p_sd15_openpose.pth"
    },
    "class_type": "ControlNetLoaderAdvanced",
    "_meta": {
      "title": "Load Advanced ControlNet Model 🛂🅐🅒🅝"
    }
  },
  "100": {
    "inputs": {
      "detect_hand": "enable",
      "detect_body": "enable",
      "detect_face": "enable",
      "resolution": 512,
      "bbox_detector": "yolox_l.onnx",
      "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
      "scale_stick_for_xinsr_cn": "disable",
      "image": [
        "53",
        0
      ]
    },
    "class_type": "DWPreprocessor",
    "_meta": {
      "title": "DWPose Estimator"
    }
  },
  "102": {
    "inputs": {
      "a": 6.283185307179586,
      "bg_threshold": 0.1,
      "resolution": 512,
      "image": [
        "53",
        0
      ]
    },
    "class_type": "MiDaS-DepthMapPreprocessor",
    "_meta": {
      "title": "MiDaS Depth Map"
    }
  },
  "103": {
    "inputs": {
      "images": [
        "100",
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
        "102",
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
      "model_name": "v3_sd15_mm.ckpt"
    },
    "class_type": "ADE_LoadAnimateDiffModel",
    "_meta": {
      "title": "Load AnimateDiff Model 🎭🅐🅓②"
    }
  },
  "117": {
    "inputs": {
      "strength": 1,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "114",
        0
      ],
      "negative": [
        "114",
        1
      ],
      "control_net": [
        "97",
        0
      ],
      "image": [
        "100",
        0
      ]
    },
    "class_type": "ACN_AdvancedControlNetApply",
    "_meta": {
      "title": "Apply Advanced ControlNet 🛂🅐🅒🅝"
    }
  }
}
"""

# -------------------------------------------------------------------
# ADD-ON PREFERENCES: ComfyUI Server Address
# -------------------------------------------------------------------
class ComfyUIAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__ # This will be 'sdxl' if file is in sdxl folder

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
        box.label(text="Dependencies & Notes:", icon='INFO')
        box.label(text="- Requires 'ffmpeg' installed and in system PATH.")
        box.label(text="- Requires 'websocket-client' Python package.")
        box.label(text=f"  (Attempting to load from: {packages_path})")
        box.label(text="- Frame capture uses the scene's render settings or OpenGL.")
        box.label(text="- Rendering frames can be slow and will pause Blender.")


# -------------------------------------------------------------------
# HELPER: Capture the current view
# -------------------------------------------------------------------
def capture_viewport(output_path):
    """
    Renders the current view to the specified PNG file path.
    Uses camera render if available and active, otherwise OpenGL viewport render.
    NOTE: bpy.ops.render.render() can be very slow depending on scene/engine.
    """
    # Store current settings
    scene = bpy.context.scene
    render = scene.render
    original_filepath = render.filepath
    original_engine = render.engine # Store engine if we change it (not currently changing)
    original_display_mode = None # To restore viewport shading

    # Find 3D view area and space
    area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
    space = next((s for s in area.spaces if s.type == 'VIEW_3D'), None) if area else None

    try:
        render.filepath = output_path

        use_opengl = True # Default to faster OpenGL
        if scene.camera:
             # Check if we are actually in camera view in the 3D viewport
             is_camera_view = space and space.region_3d.view_perspective == 'CAMERA'
             if is_camera_view:
                 print("Capturing using Render Engine (might be slow).")
                 # Ensure correct render settings (resolution etc. are used from scene)
                 bpy.ops.render.render(write_still=True)
                 use_opengl = False # Render command was used
             else:
                 print("Camera exists, but not in camera view. Using OpenGL viewport capture.")
        else:
            print("No scene camera found, using OpenGL viewport capture.")

        if use_opengl:
            if space:
                 # Optional: Set solid mode for consistency? Or keep current shading?
                 # original_display_mode = space.shading.type
                 # space.shading.type = 'SOLID'
                 # space.shading.color_type = 'TEXTURE' # Show textures if in solid mode

                 # Use opengl render which respects viewport settings
                 bpy.ops.render.opengl(write_still=True, view_context=True)
            else:
                 print("Warning: Could not find 3D Viewport space for OpenGL render.")
                 # Fallback attempt? Might not capture correctly.
                 bpy.ops.render.opengl(write_still=True)


    except Exception as e:
         print(f"Error during viewport capture: {e}")
         raise # Re-raise the exception to be caught by the operator
    finally:
        # Restore original settings
        render.filepath = original_filepath
        # render.engine = original_engine # Restore if changed
        # if space and original_display_mode:
        #      space.shading.type = original_display_mode # Restore viewport shading

    if not os.path.exists(output_path):
        raise RuntimeError(f"Output image file not found after capture attempt: {output_path}")

    return output_path


# -------------------------------------------------------------------
# HELPER: Create video from sequence of frames using ffmpeg
# -------------------------------------------------------------------
def create_video_from_frames(frame_dir, output_video_path, frame_rate=24, frame_pattern="frame_%04d.png", start_number=None):
    """
    Uses ffmpeg to create a video from a sequence of PNG frames.
    Requires ffmpeg to be in the system PATH.
    frame_pattern example: 'frame_%04d.png'
    start_number: Optional first frame number if pattern needs it explicitly.
    """
    input_pattern = os.path.join(frame_dir, frame_pattern)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    command = [
        'ffmpeg',
        '-y', # Overwrite output without asking
        '-framerate', str(frame_rate),
    ]
    # If frame numbers don't start at 0 or 1, ffmpeg might need -start_number
    if start_number is not None:
         command.extend(['-start_number', str(start_number)])

    command.extend([
        '-i', input_pattern,
        '-c:v', 'libx264',   # Video codec
        '-crf', '23',        # Quality (lower is better, 18-28 reasonable)
        '-pix_fmt', 'yuv420p',# Pixel format for compatibility
        output_video_path
    ])

    print(f"Running ffmpeg command: {' '.join(command)}")
    try:
        startupinfo = None
        if os.name == 'nt': # Windows specific: hide console
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(command, check=True, capture_output=True, text=True, startupinfo=startupinfo, encoding='utf-8')
        # print("ffmpeg stdout:", result.stdout) # Can be noisy
        if result.stderr:
             print("ffmpeg stderr:", result.stderr) # Print stderr for warnings/info
        print(f"Video created successfully: {output_video_path}")
        return output_video_path
    except FileNotFoundError:
        print("ERROR: ffmpeg command not found. Make sure ffmpeg is installed and in your system's PATH.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed with exit code {e.returncode}")
        print("ffmpeg stdout:", e.stdout)
        print("ffmpeg stderr:", e.stderr)
        raise
    except Exception as e:
        print(f"An unexpected error occurred during video creation: {e}")
        raise


# -------------------------------------------------------------------
# HELPER: ComfyUI API Interaction (Unchanged)
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
        print(f"Fetching history from: {url}")
        with urllib.request.urlopen(url, timeout=10) as response: # Add timeout to HTTP request
            if response.status == 200:
                history = json.loads(response.read())
                # History is a dict {prompt_id: {outputs: {node_id: {images: [...]}}}}
                prompt_data = history.get(prompt_id)
                if not prompt_data:
                    print(f"History found, but data for prompt {prompt_id} is missing.")
                    return None # History exists but is incomplete?

                outputs = prompt_data.get('outputs')
                if not outputs:
                     print(f"History for prompt {prompt_id} has no 'outputs' section (likely still running/processing).")
                     return None # Not finished processing outputs

                node_output = outputs.get(str(target_node_id)) # Node IDs in history keys are strings
                if node_output is None: # Check specifically for None, as empty dict is possible
                    print(f"History outputs for prompt {prompt_id} do not contain node ID {target_node_id}.")
                    # Log available nodes for debugging
                    print(f"Available output nodes in history: {list(outputs.keys())}")
                    return [] # Return empty list indicating node wasn't found in output

                images_info = node_output.get('images')
                if images_info is None: # Check specifically for None
                    print(f"Node {target_node_id} output found in history but has no 'images' key.")
                    return [] # Node output exists but no images key
                if not images_info:
                    print(f"Node {target_node_id} output found in history but the 'images' list is empty.")
                    return [] # Node output exists but images list is empty


                print(f"Found {len(images_info)} images for node {target_node_id} in history.")
                images_data = [] # Initialize list now that we expect images
                fetch_errors = 0
                fetch_start_time = time.time()
                for i, image_info in enumerate(images_info):
                    filename = image_info.get('filename')
                    subfolder = image_info.get('subfolder')
                    img_type = image_info.get('type')
                    if filename and img_type:
                        print(f"  Fetching image {i+1}/{len(images_info)} from history ref: {filename}")
                        # Use the existing helper to fetch actual image data
                        img_data = get_comfyui_image_data(filename, subfolder, img_type, server_address)
                        if img_data:
                            images_data.append(img_data)
                        else:
                            print(f"  ERROR: Failed to retrieve image data for {filename} from /view API (referenced in history).")
                            fetch_errors += 1
                    else:
                        print(f"  Warning: Incomplete image info in history output: {image_info}")
                        fetch_errors += 1

                print(f"  History image fetching took {time.time()-fetch_start_time:.2f}s.")
                if fetch_errors > 0:
                     print(f"Warning: Failed to fetch {fetch_errors} image(s) referenced in history.")
                     # Return partial list if some succeeded, or empty if all failed

                return images_data # Return list (might be empty if fetch failed for all)

            elif response.status == 404:
                 print(f"History API returned 404 Not Found for prompt {prompt_id}. Prompt likely hasn't finished or ID is wrong.")
                 return None
            else:
                print(f"Error fetching history for prompt {prompt_id}: HTTP Status {response.status}")
                return None # Indicate history fetch failed

    except urllib.error.URLError as e:
        print(f"URL Error fetching history {prompt_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error processing history for {prompt_id}: {e}")
        return None
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
    data = json.dumps(p).encode('utf-8')
    url = f"{server_address}/prompt"
    print(f"Queueing prompt to {url}")
    req = urllib.request.Request(url, data=data)
    try:
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except urllib.error.URLError as e:
        print(f"Error queueing prompt: {e}")
        error_message = f"Error queueing prompt: {e}"
        if hasattr(e, 'reason'): error_message += f" Reason: {e.reason}"
        if hasattr(e, 'read'):
            try: error_message += f" Server Response: {e.read().decode()}"
            except Exception as decode_e: error_message += f" Server Response (decode error): {decode_e}"
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
    print(f"Fetching image from: {url}")
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                return response.read()
            else:
                print(f"Error fetching image {filename}: Status code {response.status}")
                return None
    except urllib.error.URLError as e:
        print(f"Error fetching image {filename}: {e}")
        if hasattr(e, 'read'):
            try: print(f"Server response: {e.read().decode()}")
            except: pass
        return None # Indicate failure
    except Exception as e:
         print(f"Unexpected error fetching image {filename}: {e}")
         return None


def get_comfyui_images_ws(prompt_id, server_address, client_id, output_node_id="12", progress_callback=None):
    """
    Connects via WebSocket, waits for results from a specific node.
    Includes timeout, progress reporting, and fallback to /history API
    if the final node 'executed' message is missed.
    Raises exceptions on critical errors (connection, timeout without completion).
    """
    if not websocket:
        raise RuntimeError("websocket-client package not loaded.")

    ws_url = f"ws://{server_address.split('//')[1]}/ws?clientId={client_id}"
    print(f"Connecting to WebSocket: {ws_url}")
    ws = websocket.WebSocket()
    retrieved_via_history = False # Flag to track if history fallback was used

    try:
        ws.connect(ws_url, timeout=20) # Increased connection timeout
        print("WebSocket connected.")
    except websocket.WebSocketException as e:
         print(f"WebSocket connection failed: {e}")
         raise ConnectionError(f"WebSocket connection failed: {e}") from e
    except Exception as e:
         print(f"WebSocket connection failed unexpectedly: {e}")
         raise ConnectionError(f"WebSocket connection failed unexpectedly: {e}") from e

    images_data = []
    execution_done_via_ws = False # Flag for getting images via WebSocket message
    prompt_execution_finished_signal = False # Flag for the overall prompt completion signal (node=None)
    consecutive_timeouts = 0
    max_consecutive_timeouts_before_history = 8
    overall_timeout_seconds = 600 # Keep overall timeout (10 minutes)
    start_time = time.time()

    try:
        while not execution_done_via_ws and (time.time() - start_time < overall_timeout_seconds):
            try:
                ws.settimeout(15) # Check every 15 seconds
                out = ws.recv()
                consecutive_timeouts = 0 # Reset timeout counter on successful receive
            except websocket.WebSocketTimeoutException:
                consecutive_timeouts += 1
                print(f"WebSocket receive timed out ({consecutive_timeouts}/{max_consecutive_timeouts_before_history}). Checking state...")

                if prompt_execution_finished_signal and consecutive_timeouts >= max_consecutive_timeouts_before_history:
                    print("Prompt finished signal received, but output message potentially missed. Trying /history API fallback...")
                    history_result = get_comfyui_results_from_history(prompt_id, server_address, output_node_id)

                    if history_result is not None:
                        print(f"Received {len(history_result)} images from /history API fallback.")
                        images_data = history_result
                        execution_done_via_ws = True # Mark as done (even if via history)
                        retrieved_via_history = True # Set flag
                        break # Exit WebSocket loop
                    else:
                         print("Warning: /history API did not provide results yet or failed. Will continue listening to WebSocket briefly.")

                elif consecutive_timeouts < overall_timeout_seconds / ws.gettimeout():
                    try:
                        ws.ping()
                        print("  Connection ping OK.")
                    except websocket.WebSocketConnectionClosedException:
                        print("WebSocket connection closed while waiting for data (detected by ping).")
                        if prompt_execution_finished_signal:
                             print("Connection closed after prompt finished. Final /history check...")
                             history_result = get_comfyui_results_from_history(prompt_id, server_address, output_node_id)
                             if history_result is not None:
                                  images_data = history_result
                                  retrieved_via_history = True # Set flag
                             execution_done_via_ws = True
                             break
                        else:
                             raise ConnectionAbortedError("WebSocket connection closed by server before prompt completion signal.")
                else:
                     pass

                continue # Go to next loop iteration after handling timeout

            if isinstance(out, str):
                try:
                    message = json.loads(out)
                except json.JSONDecodeError as e:
                    print(f"Error decoding WebSocket message: {e}\nReceived raw: {out[:200]}...")
                    continue

                msg_type = message.get('type')
                data = message.get('data', {})

                if msg_type == 'status':
                    # ... (status handling as before) ...
                    status = data.get('status', {})
                    exec_info = status.get('exec_info', {})
                    queue_remaining = exec_info.get('queue_remaining', 0)
                    if progress_callback: progress_callback(f"Queue: {queue_remaining} left")

                elif msg_type == 'progress':
                    # ... (progress handling as before) ...
                     value = data.get('value', 0)
                     max_val = data.get('max', 0)
                     if max_val > 0 and progress_callback:
                          percent = int((value / max_val) * 100)
                          progress_callback(f"Executing: {percent}% ({value}/{max_val})")

                elif msg_type == 'executing':
                    prompt_id_msg = data.get('prompt_id')
                    node_id_msg = data.get('node')

                    if node_id_msg is None and prompt_id_msg == prompt_id:
                        print("Execution finished signal received (node=None).")
                        prompt_execution_finished_signal = True

                    elif node_id_msg == output_node_id and prompt_id_msg == prompt_id:
                        print(f"Output node {output_node_id} is executing...")
                        if progress_callback: progress_callback(f"Node {output_node_id} running")

                elif msg_type == 'executed':
                     prompt_id_msg = data.get('prompt_id')
                     node_id_msg = data.get('node_id')

                     if node_id_msg == int(output_node_id) and prompt_id_msg == prompt_id:
                         print(f"Output node {output_node_id} finished execution (message received).")
                         outputs = data.get('outputs', {})
                         if 'images' in outputs:
                             print(f"Found {len(outputs['images'])} images in node {output_node_id} output via WebSocket.")
                             img_fetch_start_time = time.time()
                             images_data_ws = []
                             fetch_errors = 0
                             for i, image_info in enumerate(outputs['images']):
                                 filename = image_info.get('filename')
                                 subfolder = image_info.get('subfolder')
                                 img_type = image_info.get('type')
                                 if filename and img_type:
                                     if progress_callback: progress_callback(f"Fetching image {i+1}/{len(outputs['images'])}")
                                     print(f"  Retrieving image via WS ref: {filename}")
                                     img_data = get_comfyui_image_data(filename, subfolder, img_type, server_address)
                                     if img_data:
                                         images_data_ws.append(img_data)
                                     else:
                                         print(f"  ERROR: Failed to retrieve image data for {filename} (from WS msg).")
                                         fetch_errors += 1
                                 else:
                                     print(f"  Warning: Incomplete image info in WS output: {image_info}")
                                     fetch_errors += 1
                             print(f"  WS Image fetching took {time.time() - img_fetch_start_time:.2f}s.")
                             if fetch_errors > 0: print(f"Warning: Failed to fetch {fetch_errors} image(s) referenced in WS message.")

                             images_data = images_data_ws
                             execution_done_via_ws = True
                             retrieved_via_history = False # Explicitly set false
                             break # Exit loop
                         else:
                              print(f"Warning: Output node {output_node_id} executed but no 'images' found in outputs (WS msg).")
                              images_data = []
                              execution_done_via_ws = True
                              retrieved_via_history = False # Explicitly set false
                              break

    except ConnectionAbortedError as e:
         print(f"WebSocket Error: {e}")
         raise
    except Exception as e:
        print(f"Error processing WebSocket messages: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error processing WebSocket messages: {e}") from e
    finally:
        ws.close()
        print("WebSocket closed.")

    # --- Post-Loop Checks ---
    if not execution_done_via_ws and prompt_execution_finished_signal:
         print("WebSocket loop finished without results, but prompt finished signal was seen. Final /history check.")
         history_result = get_comfyui_results_from_history(prompt_id, server_address, output_node_id)
         if history_result is not None:
              print(f"Final history check successful, received {len(history_result)} images.")
              images_data = history_result
              retrieved_via_history = True # Set flag
         else:
              print("Final history check failed or returned no results.")
              if images_data is None: images_data = [] # Ensure it's a list

    elif not execution_done_via_ws and (time.time() - start_time >= overall_timeout_seconds):
        print(f"ERROR: Overall timeout ({overall_timeout_seconds}s) waiting for ComfyUI results for prompt {prompt_id}.")
        raise TimeoutError(f"Overall timeout ({overall_timeout_seconds}s) waiting for ComfyUI results for prompt {prompt_id}.")

    # --- Final Logging and Return ---
    if images_data is None: images_data = []

    # *** Corrected Log Messages ***
    log_source = "History API fallback" if retrieved_via_history else "WebSocket"
    if not images_data:
         print(f"Warning: Operation finished, but no images were retrieved via {log_source}.")
    else:
         print(f"Info: Images retrieved via {log_source} ({len(images_data)} images).")

    return images_data

# -------------------------------------------------------------------
# HELPER: Create a plane with image texture and keyframed visibility
# (Mostly unchanged, minor robustness)
# -------------------------------------------------------------------
def create_plane_with_image(img: bpy.types.Image, name="ComfyUI Plane", frame_number=None):
    """
    Creates a plane, aligns to camera/view, applies image, keyframes visibility.
    Ensures operations happen in Object mode. Includes more logging.
    Returns the created plane object or None on failure.
    """
    print(f"  create_plane_with_image: Called for '{name}', frame {frame_number}") # DEBUG
    if frame_number is None:
        frame_number = bpy.context.scene.frame_current

    # --- Ensure Object Mode ---
    active_obj = bpy.context.active_object
    previous_mode = None
    if active_obj and active_obj.mode != 'OBJECT':
        previous_mode = active_obj.mode
        print(f"    create_plane: Forcing Object mode from {previous_mode}") # DEBUG
        try:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        except Exception as e:
             print(f"    create_plane: Warning - Could not force Object mode: {e}") # DEBUG

    # --- Create and Align Plane ---
    plane_obj = None
    new_plane_name = "" # To track name even if creation fails midway
    align_method = "Unknown"
    try:
        cam = bpy.context.scene.camera
        if cam:
            align_method = "Camera Alignment"
            print(f"    create_plane: Attempting {align_method}") # DEBUG
            cam_matrix = cam.matrix_world
            # *** FIXED: Use mathutils.Vector ***
            location = cam_matrix.translation + cam_matrix.to_quaternion() @ mathutils.Vector((0, 0, -5.0)) # Offset
            rotation = cam_matrix.to_euler()

            bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=rotation)
            plane_obj = bpy.context.active_object # Assumes operator makes it active
            if not plane_obj: raise RuntimeError("Plane creation operator failed to return an active object.")
            new_plane_name = plane_obj.name # Store the temp name
            print(f"    create_plane: Created primitive plane '{new_plane_name}' at {location}") # DEBUG
            plane_obj.name = name

            # Adjust scale based on image aspect ratio relative to default plane size (2x2)
            base_dimension = 5.0 # Desired largest dimension in Blender units
            img_w, img_h = img.size
            if img_w == 0 or img_h == 0:
                 scale_x, scale_y = base_dimension / 2.0, base_dimension / 2.0
            elif img_w >= img_h:
                 scale_x = base_dimension / 2.0
                 scale_y = (base_dimension * (img_h / img_w)) / 2.0
            else: # img_h > img_w
                 scale_y = base_dimension / 2.0
                 scale_x = (base_dimension * (img_w / img_h)) / 2.0
            plane_obj.scale = (scale_x, scale_y, 1)
            print(f"    create_plane: Applied scale ({scale_x:.2f}, {scale_y:.2f}, 1)") # DEBUG

        else: # Fallback if no camera: Align to view
            align_method = "View Alignment"
            print(f"    create_plane: No camera found. Attempting {align_method}") # DEBUG
            cursor_loc = bpy.context.scene.cursor.location
            bpy.ops.mesh.primitive_plane_add(size=1, align='VIEW', location=cursor_loc)
            plane_obj = bpy.context.active_object
            if not plane_obj: raise RuntimeError("Plane creation operator failed to return an active object.")
            new_plane_name = plane_obj.name # Store the temp name
            print(f"    create_plane: Created primitive plane '{new_plane_name}' at cursor {cursor_loc}") # DEBUG
            plane_obj.name = name

            # Adjust scale based on image aspect ratio
            base_size = 5.0
            img_w, img_h = img.size
            if img_w == 0 or img_h == 0:
                 scale_x, scale_y = base_size / 2.0, base_size / 2.0
            elif img_w >= img_h:
                 scale_x = base_size / 2.0
                 scale_y = (base_size * (img_h / img_w)) / 2.0
            else:
                 scale_x = (base_size * (img_w / img_h)) / 2.0
                 scale_y = base_size / 2.0
            plane_obj.scale = (scale_x, scale_y, 1)
            print(f"    create_plane: Applied scale ({scale_x:.2f}, {scale_y:.2f}, 1)") # DEBUG

    except Exception as e:
         print(f"    create_plane: ERROR creating/aligning plane '{new_plane_name}' using {align_method}: {e}") # DEBUG
         # If plane_obj exists but its name is not in data.objects, it means deletion happened or error was severe
         if plane_obj and plane_obj.name not in bpy.data.objects: plane_obj = None
         if not plane_obj: return None # Exit if creation failed

    # --- Create Material ---
    try:
        print(f"    create_plane: Creating material for '{plane_obj.name}'") # DEBUG
        mat_name = f"ComfyUIImageMat_{plane_obj.name}"
        # Check if material already exists to prevent potential name conflicts if run multiple times
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
             mat = bpy.data.materials.new(name=mat_name)
             print(f"    create_plane: New material '{mat_name}' created.") # DEBUG
        else:
             print(f"    create_plane: Reusing existing material '{mat_name}'.") # DEBUG

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        output_node = nodes.new("ShaderNodeOutputMaterial")
        shader_node = nodes.new("ShaderNodeEmission") # Use Emission for unlit
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.interpolation = 'Closest'

        tex_node.location = (-300, 0)
        shader_node.location = (0, 0)
        output_node.location = (300, 0)

        links.new(tex_node.outputs["Color"], shader_node.inputs["Color"])
        links.new(shader_node.outputs["Emission"], output_node.inputs["Surface"])
        print(f"    create_plane: Material nodes created and linked.") # DEBUG

        if plane_obj.data.materials:
            plane_obj.data.materials[0] = mat
        else:
            plane_obj.data.materials.append(mat)
        print(f"    create_plane: Material assigned.") # DEBUG
    except Exception as e:
         print(f"    create_plane: ERROR creating material for {plane_obj.name}: {e}") # DEBUG

    # --- Keyframe Visibility ---
    try:
        print(f"    create_plane: Keyframing visibility for frame {frame_number} (+/- 1)") # DEBUG
        def set_constant_interpolation(obj, data_path):
            if obj.animation_data and obj.animation_data.action:
                for fcurve in obj.animation_data.action.fcurves:
                    if fcurve.data_path == data_path:
                        for kp in fcurve.keyframe_points:
                            kp.interpolation = 'CONSTANT'
                        fcurve.update()
                        return

        if not plane_obj.animation_data: plane_obj.animation_data_create()
        if not plane_obj.animation_data.action:
             action_name = f"{plane_obj.name}_VisAction"
             plane_obj.animation_data.action = bpy.data.actions.new(name=action_name)

        # Keyframing logic remains the same...
        plane_obj.hide_viewport = True
        plane_obj.hide_render = True
        plane_obj.keyframe_insert(data_path='hide_viewport', frame=frame_number - 1)
        plane_obj.keyframe_insert(data_path='hide_render', frame=frame_number - 1)
        set_constant_interpolation(plane_obj, 'hide_viewport')
        set_constant_interpolation(plane_obj, 'hide_render')

        plane_obj.hide_viewport = False
        plane_obj.hide_render = False
        plane_obj.keyframe_insert(data_path='hide_viewport', frame=frame_number)
        plane_obj.keyframe_insert(data_path='hide_render', frame=frame_number)
        set_constant_interpolation(plane_obj, 'hide_viewport')
        set_constant_interpolation(plane_obj, 'hide_render')

        plane_obj.hide_viewport = True
        plane_obj.hide_render = True
        plane_obj.keyframe_insert(data_path='hide_viewport', frame=frame_number + 1)
        plane_obj.keyframe_insert(data_path='hide_render', frame=frame_number + 1)
        set_constant_interpolation(plane_obj, 'hide_viewport')
        set_constant_interpolation(plane_obj, 'hide_render')
        print(f"    create_plane: Visibility keyframed successfully.") # DEBUG

    except Exception as e:
         print(f"    create_plane: ERROR keyframing visibility for {plane_obj.name}: {e}") # DEBUG

    # --- Restore Mode if changed (Better handled outside this function) ---
    # Commenting out mode restoration here, should be done in execute/finish_or_cancel
    # if previous_mode and active_obj and active_obj.name in bpy.context.view_layer.objects:
    #     if bpy.context.view_layer.objects.active != active_obj:
    #          bpy.context.view_layer.objects.active = active_obj
    #     try:
    #         bpy.ops.object.mode_set(mode=previous_mode)
    #         print(f"    create_plane: Restored mode to {previous_mode}") # DEBUG
    #     except Exception as e:
    #         print(f"    create_plane: Could not restore previous mode '{previous_mode}': {e}") # DEBUG

    print(f"  create_plane_with_image: Finished for '{plane_obj.name if plane_obj else 'Failed Plane'}'.") # DEBUG
    return plane_obj # Return the created object (or None if failed)

# -------------------------------------------------------------------
# OPERATOR: Main Modal Operator
# -------------------------------------------------------------------
class OBJECT_OT_run_comfyui_modal(bpy.types.Operator):
    bl_idname = "object.run_comfyui_modal" # Changed ID
    bl_label = "Run ComfyUI Workflow (Modal)"
    bl_description = (
        "Capture frames, create video, send to ComfyUI, create textured planes. Runs in background."
    )
    bl_options = {'REGISTER', 'UNDO'} # Removed RUNNING_MODAL here, add in invoke

    # --- Operator Properties ---
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
        description="Choose whether to process a single frame or a range."
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
        name="Frame Rate for Video",
        description="Frame rate for the temporary video sent to ComfyUI",
        default=8, # Match default in VHS_VideoCombine if relevant?
        min=1,
    )

    # --- Modal State Variables ---
    _timer = None
    _thread = None
    _result_queue = None
    _status_message = "Idle"
    _progress_message = "" # Specific progress like % or step
    _is_running = False # Flag to prevent double execution

    # --- Stored State for Thread & Cleanup ---
    temp_dir = None
    frames_to_process = []
    frame_paths = []
    frame_pattern = "frame_%04d.png"
    workflow = None
    client_id = None
    original_frame = 0
    previous_obj = None
    previous_mode = None
    final_result = None # Stores result or exception from thread
    created_planes = []
    output_img_paths = []
    server_address = ""

    # --- Configurable Node IDs ---
    video_path_node_id = "107"
    prompt_node_id = "3"
    output_node_id = "12"

    # --- Worker Thread Function ---
    def _comfyui_worker_thread(self):
        """
        Function executed in a separate thread.
        Performs video creation, ComfyUI queuing, and waits for results.
        Communicates progress and results via self._result_queue and setting _status_message/_progress_message.
        IMPORTANT: Do NOT interact with bpy data (bpy.data, bpy.context, most bpy.ops) from here.
        """
        try:
            # --- 2) Create Video ---
            self._status_message = "Creating temporary video..."
            self._progress_message = ""
            # Determine start number for ffmpeg if needed (e.g., if frames don't start near 0/1)
            start_num = self.frames_to_process[0] if self.frames_to_process else None

            # Construct video path using stored temp_dir and frame number
            video_filename = f"input_video_{self.frames_to_process[0]}.mp4"
            temp_video_path = os.path.join(self.temp_dir, video_filename)

            create_video_from_frames(
                self.temp_dir,
                temp_video_path,
                self.frame_rate,
                self.frame_pattern,
                start_number=start_num
            )

            # --- 3) Modify and Queue Workflow ---
             # Ensure path uses forward slashes or escaped backslashes for JSON
            abs_video_path = os.path.abspath(temp_video_path).replace('\\', '/')
            self.workflow[self.video_path_node_id]["inputs"]["video"] = abs_video_path
            self.workflow[self.prompt_node_id]["inputs"]["text"] = self.user_prompt # Get from self

            self._status_message = f"Queueing ComfyUI prompt (Client: {self.client_id[:8]}...)"
            queue_response = queue_comfyui_prompt(self.workflow, self.server_address, self.client_id)
            prompt_id = queue_response.get('prompt_id')
            if not prompt_id:
                raise RuntimeError(f"Did not receive prompt_id. Response: {queue_response}")
            print(f"ComfyUI Prompt ID: {prompt_id}")

            # --- 4) Wait for Results via WebSocket ---
            self._status_message = f"Waiting for ComfyUI results (Prompt: {prompt_id[:8]}...)"

            # Define a callback for progress updates from the websocket function
            def progress_update(message):
                 self._progress_message = message

            output_images_data = get_comfyui_images_ws(
                prompt_id,
                self.server_address,
                self.client_id,
                self.output_node_id,
                progress_callback=progress_update # Pass the callback
            )

            # --- Success ---
            self._status_message = "Received results from ComfyUI."
            self._progress_message = ""
            self._result_queue.put(output_images_data) # Put list of image data

        except Exception as e:
            # --- Failure ---
            self._status_message = f"Error during ComfyUI interaction."
            self._progress_message = str(e)
            print(f"Error in worker thread: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            self._result_queue.put(e) # Put the exception object in the queue

    # --- Main Execution Logic (Called by Modal on Completion) ---
    def execute(self, context):
        """
        Processes the results received from the worker thread.
        Creates image planes. Assumes it's called from the modal method
        when results are ready (runs in main thread).
        """
        print("Entering Execute method...") # DEBUG
        if isinstance(self.final_result, Exception):
            self.report({'ERROR'}, f"ComfyUI processing failed in worker thread: {self.final_result}")
            return {'CANCELLED'} # Error state
        elif self.final_result is None: # Check specifically for None if history failed
             self.report({'WARNING'}, "No response or failed response received from ComfyUI.")
             return {'FINISHED'}
        elif not isinstance(self.final_result, list):
            self.report({'ERROR'}, f"Received unexpected result type from worker: {type(self.final_result)}")
            return {'CANCELLED'}
        elif not self.final_result:
            self.report({'INFO'}, "No images were generated or retrieved by ComfyUI.")
            return {'FINISHED'} # Finished, but with no results

        # --- Process images and create planes ---
        print(f"Execute method received result type: {type(self.final_result)}, length: {len(self.final_result)}") # DEBUG
        if self.final_result:
             print(f"  First element type: {type(self.final_result[0])}") # DEBUG (should be bytes)

        self.report({'INFO'}, f"Received {len(self.final_result)} image(s). Creating planes...")
        self.created_planes = [] # Reset list for this run
        self.output_img_paths = [] # Reset list for this run

        # Ensure Blender is in Object mode before creating planes
        original_active = context.active_object
        original_mode = original_active.mode if original_active else 'OBJECT'
        needs_mode_change = original_mode != 'OBJECT'
        if needs_mode_change:
            print("Switching to Object mode for plane creation...") # DEBUG
            try: bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError as e:
                 self.report({'ERROR'}, f"Could not set Object mode: {e}. Plane creation failed.")
                 return {'CANCELLED'} # Cannot proceed without object mode

        context.window_manager.progress_begin(0, len(self.final_result))
        success_count = 0
        planes_parent = None # Optional: Parent all planes to an empty

        # Create parent empty (optional)
        try:
            parent_name = f"ComfyUI_Output_{self.client_id[:6]}"
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=context.scene.cursor.location)
            planes_parent = context.active_object
            planes_parent.name = parent_name
            print(f"Created parent empty: {parent_name}") # DEBUG
        except Exception as e:
             print(f"Warning: Could not create parent empty: {e}")
             planes_parent = None


        for i, img_data in enumerate(self.final_result):
            if not isinstance(img_data, bytes) or not img_data:
                self.report({'WARNING'}, f"Skipping invalid image data at index {i}.")
                continue

            frame_num = self.frames_to_process[i] if i < len(self.frames_to_process) else self.original_frame + i

            output_img_filename = f"comfy_out_{frame_num:04d}.png" # Use padding
            output_img_path = os.path.join(self.temp_dir, output_img_filename)
            print(f"\nProcessing image {i+1}/{len(self.final_result)} for frame {frame_num}...") # DEBUG

            try:
                # Save image data to file
                print(f"  Saving image data to: {output_img_path}") # DEBUG
                with open(output_img_path, "wb") as f:
                    f.write(img_data)
                if not os.path.exists(output_img_path):
                     raise IOError("Temporary image file not found after writing.")
                print(f"  Image data saved successfully.") # DEBUG
                self.output_img_paths.append(output_img_path) # Track for cleanup only if save succeeds

                # Load image into Blender
                print(f"  Loading image into Blender: {output_img_path}") # DEBUG
                processed_image = bpy.data.images.load(output_img_path)
                if not processed_image:
                    # This is critical - if loading fails, nothing else will work
                    self.report({'ERROR'}, f"CRITICAL: Failed to load temporary image into Blender: {output_img_path}")
                    # Attempt to remove the problematic temp file
                    if os.path.exists(output_img_path):
                         try: os.remove(output_img_path)
                         except OSError as rem_e: print(f"    Could not remove failed temp image: {rem_e}")
                    # Remove path from tracking if load failed
                    if output_img_path in self.output_img_paths: self.output_img_paths.remove(output_img_path)
                    context.window_manager.progress_update(i + 1) # Update progress even on fail
                    continue # Skip creating plane for this failed image load

                processed_image.name = output_img_filename
                print(f"  Image '{processed_image.name}' loaded into Blender.") # DEBUG
                # processed_image.pack() # Optional

                # Create plane - Ensure correct frame context for keyframing
                print(f"  Setting scene frame to {frame_num} for keyframing.") # DEBUG
                context.scene.frame_set(frame_num)
                plane_name = f"ComfyUI Plane (Frame {frame_num})"
                print(f"  Calling create_plane_with_image for '{plane_name}'...") # DEBUG
                new_plane = create_plane_with_image(processed_image, name=plane_name, frame_number=frame_num)

                if new_plane:
                    print(f"  Successfully created plane '{new_plane.name}'.") # DEBUG
                    self.created_planes.append(new_plane)
                    success_count += 1
                    # Parent to empty (optional)
                    if planes_parent:
                         print(f"    Parenting '{new_plane.name}' to '{planes_parent.name}'.") # DEBUG
                         # Ensure matrices are updated before parenting
                         context.view_layer.update()
                         original_matrix = new_plane.matrix_world.copy()
                         new_plane.parent = planes_parent
                         new_plane.matrix_world = original_matrix # Keep original world transform
                else:
                     self.report({'WARNING'}, f"Function create_plane_with_image failed for frame {frame_num}")

                context.window_manager.progress_update(i + 1)

            except Exception as e:
                self.report({'ERROR'}, f"Failed processing output image {i} (frame {frame_num}): {e}")
                import traceback
                traceback.print_exc() # Print full traceback for this error
                context.window_manager.progress_update(i + 1) # Update progress even on fail
                # Continue with the next image

        context.window_manager.progress_end()

        # Restore original mode if it was changed
        if needs_mode_change and context.active_object:
             print("Restoring original mode...") # DEBUG
             if context.active_object.mode == 'OBJECT': # Only switch back if still in object mode
                try:
                    if original_active and original_active.name in context.view_layer.objects:
                         context.view_layer.objects.active = original_active
                         bpy.ops.object.mode_set(mode=original_mode)
                         print(f"  Restored mode to {original_mode}") # DEBUG
                    else: # Original object gone, stay in object mode
                         print(f"  Original object '{original_active.name if original_active else 'None'}' not found, staying in Object mode.") # DEBUG
                         pass
                except Exception as e:
                    print(f"Could not restore original mode {original_mode}: {e}")

        self.report({'INFO'}, f"Successfully created {success_count} / {len(self.final_result)} planes.")
        print("Exiting Execute method.") # DEBUG
        return {'FINISHED'}


    # --- Modal Method ---
    def modal(self, context, event):
        # Force UI update
        if context.area:
            context.area.tag_redraw()

        # Handle cancellation
        if event.type == 'ESC' or not self._is_running:
            return self.finish_or_cancel(context, cancelled=True)

        # Process timer events
        if event.type == 'TIMER':
            # Check if thread has finished
            if self._thread and not self._thread.is_alive():
                print("Worker thread finished.")
                try:
                    # Get result/error from queue (non-blocking)
                    self.final_result = self._result_queue.get_nowait()
                    print(f"Result from queue: {type(self.final_result)}")
                except queue.Empty:
                    # This shouldn't happen if thread is done, indicates potential issue
                    self.report({'ERROR'}, "Thread finished but result queue is empty.")
                    self.final_result = RuntimeError("Thread finished but queue empty.")
                except Exception as e:
                     self.report({'ERROR'}, f"Error retrieving result from queue: {e}")
                     self.final_result = e

                # Call the main execution logic (now runs in main thread)
                final_status = self.execute(context)
                # Finish the modal operator
                return self.finish_or_cancel(context, cancelled=('CANCELLED' in final_status))

            # Update status if thread is still running
            elif self._thread:
                 status_text = f"ComfyUI: {self._status_message}"
                 if self._progress_message:
                      status_text += f" ({self._progress_message})"
                 context.workspace.status_text_set(status_text)

        # Allow other events (like navigation) to pass through
        return {'PASS_THROUGH'}


    # --- Invoke Method (Starts the process) ---
    def invoke(self, context, event):
        if self._is_running:
             self.report({'WARNING'}, "Operation already in progress.")
             return {'CANCELLED'}

        if not websocket:
             self.report({'ERROR'}, "websocket-client package not found or failed to import. Please install it.")
             return {'CANCELLED'}

        # --- Initial Setup & Validation ---
        prefs = context.preferences.addons[__name__].preferences
        self.server_address = prefs.comfyui_address.strip()
        if not self.server_address:
            self.report({'ERROR'}, "ComfyUI server address is not set in add-on preferences.")
            return {'CANCELLED'}

        # Store state
        self.original_frame = context.scene.frame_current
        self.previous_obj = context.active_object
        self.previous_mode = self.previous_obj.mode if self.previous_obj else 'OBJECT' # Default to Object mode if no object

        # Determine frames
        if self.frame_mode == "CURRENT":
            self.frames_to_process = [self.original_frame]
        else: # RANGE
            if self.frame_start > self.frame_end:
                 self.report({'ERROR'}, "Start frame must be less than or equal to end frame.")
                 return {'CANCELLED'}
            self.frames_to_process = list(range(self.frame_start, self.frame_end + 1))

        if not self.frames_to_process:
             self.report({'WARNING'}, "No frames selected for processing.")
             return {'CANCELLED'}

        # --- Create Temp Directory ---
        try:
            # Clear old temp dir path if it exists from a previous run
            self.temp_dir = None
            self.temp_dir = tempfile.mkdtemp(prefix="blender_comfy_")
            print(f"Using temporary directory: {self.temp_dir}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create temp directory: {e}")
            return {'CANCELLED'}

        # --- SYNCHRONOUS Frame Capture ---
        # This part will still block Blender's UI!
        self.report({'INFO'}, f"Capturing {len(self.frames_to_process)} frame(s)... (This may take time)")
        context.window_manager.progress_begin(0, len(self.frames_to_process))
        self.frame_paths = [] # Reset list
        capture_success = True

        # Ensure Object Mode for capture stability
        if self.previous_mode != 'OBJECT' and self.previous_obj:
             try: bpy.ops.object.mode_set(mode='OBJECT')
             except Exception as e:
                  self.report({'WARNING'}, f"Could not switch to Object mode before capture: {e}")

        current_frame_capture_start = time.time()
        for i, frame_num in enumerate(self.frames_to_process):
             frame_start_time = time.time()
             context.scene.frame_set(frame_num)
             # Force viewport update - essential before capture
             context.view_layer.update()
             # Optional small delay - sometimes helps ensure buffers are ready
             # time.sleep(0.05)

             frame_filename = self.frame_pattern % frame_num # Use frame number in filename
             output_path = os.path.join(self.temp_dir, frame_filename)
             try:
                 print(f"Capturing frame {frame_num} to {output_path}...")
                 capture_viewport(output_path)
                 self.frame_paths.append(output_path)
                 context.window_manager.progress_update(i + 1)
                 print(f"  Frame {frame_num} captured in {time.time() - frame_start_time:.2f}s")
             except Exception as e:
                 self.report({'ERROR'}, f"Failed to capture frame {frame_num}: {e}")
                 capture_success = False
                 break # Stop capturing on first error

        context.window_manager.progress_end()
        print(f"Total capture time: {time.time() - current_frame_capture_start:.2f}s")

        # Restore mode if changed for capture
        if self.previous_mode != 'OBJECT' and self.previous_obj and self.previous_obj.mode == 'OBJECT':
             try:
                  # Ensure original object is active
                  if context.view_layer.objects.active != self.previous_obj and self.previous_obj.name in context.view_layer.objects:
                     context.view_layer.objects.active = self.previous_obj
                  bpy.ops.object.mode_set(mode=self.previous_mode)
             except Exception as e:
                  print(f"Could not restore mode after capture: {e}")


        if not capture_success or not self.frame_paths:
            self.report({'ERROR'}, "Frame capture failed. Aborting.")
            self._cleanup_temp_files() # Clean up partially created files/dir
            return {'CANCELLED'}
        # --- End Synchronous Capture ---

        # --- Prepare Workflow ---
        try:
            self.workflow = json.loads(COMFYUI_WORKFLOW_JSON) # Load fresh copy
            # Basic validation
            if self.video_path_node_id not in self.workflow:
                 raise ValueError(f"Workflow missing video input node ID: {self.video_path_node_id}")
            if self.prompt_node_id not in self.workflow:
                 raise ValueError(f"Workflow missing prompt node ID: {self.prompt_node_id}")
            if self.output_node_id not in self.workflow:
                 raise ValueError(f"Workflow missing expected output node ID: {self.output_node_id}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load/validate workflow JSON: {e}")
            self._cleanup_temp_files()
            return {'CANCELLED'}

        # --- Start Background Thread ---
        self._result_queue = queue.Queue()
        self.client_id = str(uuid.uuid4()) # Generate unique ID for this run
        self._thread = threading.Thread(target=self._comfyui_worker_thread)
        self._is_running = True # Set running flag
        self._thread.start()
        print("Worker thread started.")

        # --- Register Modal Handler ---
        self._timer = context.window_manager.modal_handler_add(self)
        self._status_message = "Started ComfyUI request..."
        context.workspace.status_text_set(self._status_message)
        print("Modal handler added.")

        # Add RUNNING_MODAL to options for invoke return
        self.bl_options.add('RUNNING_MODAL')
        return {'RUNNING_MODAL'}


    # --- Cleanup and State Restoration ---
    def finish_or_cancel(self, context, cancelled=False):
        """Cleans up resources and restores Blender state."""
        print(f"Finishing or cancelling operation (Cancelled: {cancelled})")
        self._is_running = False # Clear running flag

        # Remove modal timer only if it exists
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
                print("Modal timer removed.") # DEBUG
            except Exception as timer_e: # Catch potential errors during removal
                 print(f"Warning: Error removing modal timer: {timer_e}") # DEBUG
            finally:
                 self._timer = None # Ensure timer is set to None even if removal failed
        else:
             print("Finish/Cancel: No active timer found to remove.") # DEBUG


        # Clear status bar
        context.workspace.status_text_set(None)

        # Ensure thread is finished
        if self._thread and self._thread.is_alive():
            print("Warning: Worker thread still alive during finish/cancel. Attempting join...") # DEBUG
            self._thread.join(timeout=3.0) # Wait max 3 seconds
            if self._thread.is_alive():
                 print("ERROR: Worker thread did not terminate cleanly!") # DEBUG
            self._thread = None

        # Cleanup temporary files and directory
        self._cleanup_temp_files() # Assumes this helper exists and works

        # Restore Blender state (frame, selection, mode)
        try:
            # Restore frame first
            if hasattr(self, 'original_frame'):
                 print(f"Restoring scene frame to {self.original_frame}") # DEBUG
                 context.scene.frame_set(self.original_frame)

            # Restore selection and mode (using stored values)
            prev_obj = getattr(self, 'previous_obj', None)
            prev_mode = getattr(self, 'previous_mode', None)

            if prev_obj and prev_obj.name in context.view_layer.objects:
                 print(f"Attempting to restore selection to: {prev_obj.name} and mode to {prev_mode}") # DEBUG
                 try:
                      bpy.ops.object.select_all(action='DESELECT')
                      context.view_layer.objects.active = prev_obj
                      prev_obj.select_set(True)
                      if prev_mode and prev_mode != context.active_object.mode:
                          bpy.ops.object.mode_set(mode=prev_mode)
                          print(f"  Restored mode to: {prev_mode}") # DEBUG
                      else:
                          print(f"  Mode already correct ({context.active_object.mode}) or no previous mode stored.") #DEBUG
                 except Exception as restore_e:
                       print(f"  Warning: Could not fully restore selection/mode: {restore_e}") # DEBUG

            # If previous object not valid, select the newly created planes (if any)
            elif hasattr(self, 'created_planes') and self.created_planes:
                 print("Previous object invalid, selecting created planes instead.") # DEBUG
                 try:
                     bpy.ops.object.select_all(action='DESELECT')
                     active_set = False
                     for plane in self.created_planes:
                          if plane and plane.name in context.view_layer.objects:
                               plane.select_set(True)
                               if not active_set: # Set the last valid plane as active
                                    context.view_layer.objects.active = plane
                                    active_set = True
                     if active_set: print(f"  Selected {len(self.created_planes)} created planes.") # DEBUG
                 except Exception as select_e:
                      print(f"  Warning: Could not select created planes: {select_e}") #DEBUG

        except Exception as e:
            self.report({'WARNING'}, f"Error during state restoration: {e}")


        # Remove RUNNING_MODAL from options if it was added
        # Check if bl_options exists and is a set before modifying
        if hasattr(self, 'bl_options') and isinstance(self.bl_options, set) and 'RUNNING_MODAL' in self.bl_options:
            self.bl_options.remove('RUNNING_MODAL')

        self.report({'INFO'}, f"ComfyUI operation {'cancelled' if cancelled else 'finished'}.")
        print("-" * 30) # Separator for logs
        return {'CANCELLED'} if cancelled else {'FINISHED'}

    def _cleanup_temp_files(self):
        """Helper to remove temporary files and directory."""
        if not self.temp_dir or not os.path.exists(self.temp_dir):
            print("Cleanup: No temp directory to clean.")
            return

        print(f"Cleaning up temporary files in {self.temp_dir}...")
        cleaned_files = 0
        try:
            # Remove known image/video files first
            files_to_remove = []
            if hasattr(self, 'frame_paths'): files_to_remove.extend(self.frame_paths)
            if hasattr(self, 'output_img_paths'): files_to_remove.extend(self.output_img_paths)
            # Add video file path
            if hasattr(self, 'frames_to_process') and self.frames_to_process:
                video_filename = f"input_video_{self.frames_to_process[0]}.mp4"
                files_to_remove.append(os.path.join(self.temp_dir, video_filename))

            for file_path in files_to_remove:
                 if file_path and os.path.exists(file_path):
                      try:
                           os.remove(file_path)
                           cleaned_files += 1
                      except OSError as e:
                           print(f"  Warning: Could not remove file {file_path}: {e}")

            # Attempt to remove the directory itself
            try:
                os.rmdir(self.temp_dir)
                print(f"Cleanup: Removed {cleaned_files} file(s) and directory {self.temp_dir}.")
            except OSError as e:
                # If rmdir fails, list remaining items
                try:
                    remaining = os.listdir(self.temp_dir)
                    if remaining:
                        print(f"  Warning: Could not remove temp dir {self.temp_dir}. Remaining items: {remaining}. Error: {e}")
                    else:
                         # If empty but still fails, might be permissions or OS handle
                         print(f"  Warning: Could not remove empty temp dir {self.temp_dir}. Error: {e}")
                except Exception as list_e:
                    print(f"  Warning: Could not remove temp dir {self.temp_dir} (Error: {e}). Also failed to list remaining files: {list_e}.")

        except Exception as e:
            self.report({'WARNING'}, f"Error during cleanup: {e}. Manual cleanup of {self.temp_dir} may be needed.")
        finally:
            # Clear the variable so cleanup isn't attempted again on this instance
            self.temp_dir = None


# -------------------------------------------------------------------
# HELPER OPERATOR: Open this addon's preferences
# -------------------------------------------------------------------
class PREFERENCES_OT_open_comfyui_addon_prefs(bpy.types.Operator):
    """Opens the preferences specific to this addon"""
    bl_idname = "preferences.open_comfyui_addon_prefs" # More specific ID
    bl_label = "Open ComfyUI Addon Preferences"
    bl_options = {'REGISTER', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        # Ensure the addon name used in preferences is correct
        return __name__ in context.preferences.addons

    def execute(self, context):
        try:
            # Use __name__ which should evaluate to the addon's registered name (e.g., 'sdxl')
            bpy.ops.screen.userpref_show('INVOKE_DEFAULT') # Open preferences window
            # Explicitly activate the Add-ons tab and filter for this addon
            context.window_manager.windows[-1].screen.areas[-1].spaces.active.filter_text = bl_info["name"] # Filter by addon name
            return {'FINISHED'}
        except Exception as e:
            print(f"Error trying to open preferences for '{__name__}': {e}")
            self.report({'ERROR'}, f"Could not open preferences automatically: {e}. Please go to Edit > Preferences > Add-ons and search for '{bl_info['name']}'.")
            return {'CANCELLED'}

# -------------------------------------------------------------------
# PANEL: UI in the 3D View sidebar
# -------------------------------------------------------------------
class VIEW3D_PT_comfyui_panel(bpy.types.Panel):
    bl_label = "ComfyUI Gen"
    bl_idname = "VIEW3D_PT_comfyui_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ComfyUI" # Tab name in the sidebar

    def draw(self, context):
        layout = self.layout
        ws = context.window_manager # Access window state if needed

        # Check if the operator is currently running
        # This is tricky without a global state or checking operator registry easily.
        # We can use a property on the WindowManager as a simple flag.
        is_running = getattr(ws, "comfyui_modal_operator_running", False)

        # Main Operator Button
        row = layout.row()
        op = row.operator(OBJECT_OT_run_comfyui_modal.bl_idname, icon='IMAGE_DATA', text="Run ComfyUI")
        op.user_prompt = context.scene.comfyui_props.user_prompt # Link to scene prop
        # Link other properties similarly if you store them in Scene properties

        row.enabled = not is_running # Disable button if already running


        # Display Status if Running
        if is_running:
            # Access status messages stored perhaps on WindowManager or a Scene property group
            status = getattr(ws, "comfyui_modal_status", "Running...")
            progress = getattr(ws, "comfyui_modal_progress", "")
            box = layout.box()
            box.label(text=f"Status: {status}")
            if progress:
                box.label(text=f"Details: {progress}")
            # Add a cancel button maybe? Requires operator implementing cancel check
            # layout.operator("object.cancel_comfyui_modal", icon='X', text="Cancel")


        # Operator Properties (Only editable when not running)
        col = layout.column(align=True)
        col.enabled = not is_running

        # Use Scene properties instead of operator properties directly in panel
        # This makes the UI state persistent and independent of the operator instance
        scene_props = context.scene.comfyui_props
        col.prop(scene_props, "user_prompt")
        col.prop(scene_props, "frame_mode")

        if scene_props.frame_mode == 'RANGE':
            row = col.row(align=True)
            row.prop(scene_props, "frame_start", text="Start")
            row.prop(scene_props, "frame_end", text="End")
            # Display current scene frame range for reference
            scene = context.scene
            col.label(text=f"Scene Range: {scene.frame_start} - {scene.frame_end}")

        col.prop(scene_props, "frame_rate")

        layout.separator()
        # Preferences Button
        layout.operator(PREFERENCES_OT_open_comfyui_addon_prefs.bl_idname, text="Settings", icon='PREFERENCES')

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
        description="Choose whether to process a single frame or a range."
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
        name="Frame Rate for Video",
        description="Frame rate for the temporary video sent to ComfyUI",
        default=8, # Match default in VHS_VideoCombine if relevant?
        min=1,
    )

# -------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------
classes = (
    ComfyUIAddonPreferences,
    ComfyUISceneProperties, # Register PropertyGroup
    OBJECT_OT_run_comfyui_modal, # Use modal operator
    PREFERENCES_OT_open_comfyui_addon_prefs, # Use new preferences operator ID
    VIEW3D_PT_comfyui_panel,
)

def register():
    # Check dependencies
    if not websocket:
        # Raise error during registration to prevent activation
        raise ImportError("Addon requires the 'websocket-client' Python package. Please install it.")

    # Check for ffmpeg (basic check, might not be foolproof)
    try:
        startupinfo = None
        if os.name == 'nt': # Windows specific: hide console
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, startupinfo=startupinfo)
        print("ffmpeg found.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: ffmpeg check failed: {e}. Ensure ffmpeg is installed and in PATH.")
        # Don't raise error, allow addon but warn user via bl_info and preferences

    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception as e:
            print(f"Failed to register class {cls.__name__}: {e}")
            # Attempt to unregister already registered classes on failure
            unregister()
            raise # Re-raise the exception to indicate failure

    # Add Scene Properties to bpy.types.Scene
    bpy.types.Scene.comfyui_props = bpy.props.PointerProperty(type=ComfyUISceneProperties)

    # Add WindowManager properties for modal state communication (simple approach)
    bpy.types.WindowManager.comfyui_modal_operator_running = bpy.props.BoolProperty(default=False)
    bpy.types.WindowManager.comfyui_modal_status = bpy.props.StringProperty(default="Idle")
    bpy.types.WindowManager.comfyui_modal_progress = bpy.props.StringProperty(default="")


    print(f"{bl_info['name']} Add-on registered.")


def unregister():
     # Delete WindowManager properties
    try: del bpy.types.WindowManager.comfyui_modal_operator_running
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_status
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_progress
    except AttributeError: pass

    # Delete Scene properties
    try:
        del bpy.types.Scene.comfyui_props
    except AttributeError:
        pass # Ignore if not found

    # Unregister classes in reverse order
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"Failed to unregister class {cls.__name__}: {e}")

    print(f"{bl_info['name']} Add-on unregistered.")


if __name__ == "__main__":
    # Allow running the script directly in Blender Text Editor for testing
    try:
        unregister() # Unregister first if script was reloaded
    except Exception:
        pass # Ignore errors if not registered
    register()