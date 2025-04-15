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
import shutil      # For robust directory removal

# ------------------------------------------------------------------
#    Adjust this path if necessary for websocket-client or other packages:
#    Consider using Blender's built-in python/lib/site-packages or installing
#    via pip into Blender's Python environment if possible.
packages_path = "C:\\Users\\ASUS\\AppData\\Roaming\\Python\\Python311\\site-packages" # EXAMPLE PATH
if packages_path not in sys.path:
    # Check if the path exists before adding
    if os.path.isdir(packages_path):
        sys.path.insert(0, packages_path)
        print(f"Added package path: {packages_path}")
    else:
        print(f"Package path does not exist, skipping: {packages_path}")

# Try importing websocket-client
try:
    import websocket # Need to install: pip install websocket-client
except ImportError:
    print(
        "ERROR: Could not import 'websocket'. Ensure 'websocket-client' is installed "
        f"in a path known to Blender's Python (e.g., '{packages_path}' if valid, or Blender's site-packages)."
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
    "name": "ComfyUI GPencil Integration v3", # Renamed and versioned
    "author": "Your Name (Modified for Dynamic Workflows + GPencil)",
    "version": (3, 1, 0),
    "blender": (3, 3, 0),
    "location": "View3D > Sidebar > ComfyUI",
    "description": (
        "Captures frames, sends to ComfyUI using dynamically selected workflows, "
        "and adds results as Grease Pencil references fitted to camera view. "
        "Requires ACTIVE GP OBJECT, SPA addon, ffmpeg, websocket-client."
    ),
    "category": "Object",
    "warning": "Requires ACTIVE GP OBJECT, SPA Addon, ffmpeg in PATH, and 'websocket-client'.", # Updated warning
    "doc_url": "",
    "tracker_url": "",
}

# -------------------------------------------------------------------
# Workflow File Names and Node ID Mappings
# -------------------------------------------------------------------
WORKFLOW_DIR = os.path.join(os.path.dirname(__file__), "workflows")
WORKFLOW_FILES = {
    "SINGLE": "singleframe.json",
    "MULTI_REF": "multiframe_depth+reference.json",
    "MULTI_CONTEXT": "multiframe_depth+context.json",
}

# Define KEY node IDs for each workflow type. Use logical names.
# IDs based on the provided JSON files. VERIFY THESE CAREFULLY.
WORKFLOW_NODE_IDS = {
    "SINGLE": {
        "PROMPT": "4",              # CLIPTextEncode (Positive)
        "NEGATIVE_PROMPT": "13",    # CLIPTextEncode (Negative) - Optional if needed elsewhere
        "LORA_LOADER": "3",         # LoraLoader (for setting lora)
        "INPUT_IMAGE": "40",        # VHS_LoadImagePath (for the single frame)
        "CNET_STRENGTH": "6",       # ControlNetApplyAdvanced (contains strength input)
        "INVERT_BOOL": "37",        # PrimitiveBoolean (for input image inversion)
        "OUTPUT_DECODE": "19",      # PreviewImage (after upscale) - Target this for results
        "SAVE_IMAGE": "35",         # SaveImage (final output node, use DECODE for WS)
    },
    "MULTI_REF": {
        "PROMPT": "168",            # WanVideoTextEncode (Contains positive/negative)
        "INPUT_VIDEO": "189",       # VHS_LoadVideoPath (main frame sequence)
        "REFERENCE_IMAGE": "188",   # VHS_LoadImagePath (single reference image)
        "INVERT_BOOL": "185",       # PrimitiveBoolean (for input VIDEO inversion)
        "OUTPUT_DECODE": "190",     # SaveImage (produces final images)
        # "VIDEO_COMBINE": "165",     # VHS_VideoCombine (use DECODE for WS)
    },
    "MULTI_CONTEXT": {
        "PROMPT": "16",             # WanVideoTextEncode
        "INPUT_VIDEO": "184",       # VHS_LoadVideo (main frame sequence)
        "PRECEDING_IMAGE": "203",   # VHS_LoadImagePath (for frame N-1)
        "SUCCEEDING_IMAGE": "204",  # VHS_LoadImagePath (for frame N+1)
        "USE_PRECEDING_BOOL": "191",# PrimitiveBoolean (controls use of start frame)
        "USE_SUCCEEDING_BOOL": "192",# PrimitiveBoolean (controls use of end frame)
        "INVERT_BOOL": "186",       # PrimitiveBoolean (for input VIDEO inversion)
        "OUTPUT_DECODE": "205",     # SaveImage (produces final images)
        # "VIDEO_COMBINE": "139",     # VHS_VideoCombine (use DECODE for WS)
        "VACE_STARTEND_1": "111",   # Need to set num_frames here
        "VACE_STARTEND_2": "194",   # Need to set num_frames here
        "VACE_STARTEND_3": "195",   # Need to set num_frames here

    }
}

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

    lora_directory: bpy.props.StringProperty(
        name="LoRA Models Directory",
        description="Directory containing LoRA (.safetensors, .pt) files",
        subtype='DIR_PATH',
        default=r"D:\MyStuff\StableDiffusion\StabilityMatrix\Data\Models\Lora"
    )

    workflow_dir_info: bpy.props.StringProperty(
        name="Workflow Directory Info",
        description="Location where the addon expects workflow JSON files",
        default=f"Expected in: {WORKFLOW_DIR}",
        options={'HIDDEN'} # Hide from direct editing, just informational
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="ComfyUI Server Settings:")
        layout.prop(self, "comfyui_address")
        layout.separator()

        layout.label(text="Model Paths:")
        layout.prop(self, "lora_directory")
        layout.separator()

        box = layout.box()
        box.label(text="Dependencies & Notes:", icon="INFO")
        box.label(text="- Requires an ACTIVE Grease Pencil object selected.")
        box.label(text="- Requires SPA Studios Blender fork/addon features.")
        box.label(text="- Requires 'ffmpeg' installed and in system PATH.")
        box.label(text="- Requires 'websocket-client' Python package.")
        box.label(text=f"  (Attempting websocket load from relevant paths including: {packages_path})")
        box.label(text=f"- Requires workflow JSON files in: {WORKFLOW_DIR}")
        if not os.path.isdir(WORKFLOW_DIR):
             box.label(text=f"  WARNING: Workflow directory NOT FOUND!", icon='ERROR')
        else:
             missing_files = []
             for wf_type, wf_file in WORKFLOW_FILES.items():
                 if not os.path.exists(os.path.join(WORKFLOW_DIR, wf_file)):
                     missing_files.append(wf_file)
             if missing_files:
                 box.label(text=f"  WARNING: Missing workflow files: {', '.join(missing_files)}", icon='ERROR')
             else:
                 box.label(text=f"  All required workflow files found.", icon='CHECKMARK')


# -------------------------------------------------------------------
# HELPER: Capture the current view
# -------------------------------------------------------------------
def capture_viewport(output_path, context):
    """
    Renders the current view to the specified PNG file path.
    Uses camera render if available and active, otherwise OpenGL viewport render.
    """
    scene = context.scene
    render = scene.render
    original_filepath = render.filepath
    original_format = render.image_settings.file_format

    area = next((a for a in context.screen.areas if a.type == "VIEW_3D"), None)
    space = (
        next((s for s in area.spaces if s.type == "VIEW_3D"), None) if area else None
    )

    if not space:
        print("Warning: Could not find active 3D Viewport space. Capture might fail or be incorrect.")

    try:
        render.filepath = output_path
        render.image_settings.file_format = "PNG" # Ensure PNG format

        use_opengl = True
        if scene.camera:
            is_camera_view = (
                space
                and space.region_3d
                and space.region_3d.view_perspective == "CAMERA"
            )

            if is_camera_view:
                print("Capturing using Render Engine settings.")
                # *** NOTE: This will use scene render resolution, NOT the 480x360 assumed by workflow ***
                # Consider warning the user or adding an option to force OpenGL with fixed size?
                # For now, proceed as requested.
                bpy.ops.render.render(write_still=True)
                use_opengl = False
            else:
                print("Camera exists, but not in camera view. Using OpenGL viewport capture.")
        else:
            print("No scene camera found, using OpenGL viewport capture.")

        if use_opengl:
            if space:
                print("Capturing using OpenGL render.")
                render_region = None
                for region in area.regions:
                    if region.type == 'WINDOW':
                         render_region = region
                         break
                if not render_region:
                    raise RuntimeError("Cannot perform OpenGL capture without a valid WINDOW region.")

                with context.temp_override(area=area, region=render_region):
                    # *** NOTE: OpenGL also uses viewport resolution, may not be 480x360 ***
                    bpy.ops.render.opengl(write_still=True, view_context=True)
            else:
                 raise RuntimeError("Cannot perform OpenGL capture without a valid 3D Viewport space.")

    except Exception as e:
        print(f"Error during viewport capture: {e}")
        raise
    finally:
        render.filepath = original_filepath
        render.image_settings.file_format = original_format

    if not os.path.exists(output_path):
        raise RuntimeError(f"Output image file not found after capture attempt: {output_path}")
    elif os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Output image file is empty after capture attempt: {output_path}")

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
    """Uses ffmpeg to create a video from PNG frames."""
    input_pattern = os.path.join(frame_dir, frame_pattern)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    command = ["ffmpeg", "-y", "-framerate", str(frame_rate)]
    if start_number is not None:
        command.extend(["-start_number", str(start_number)])
    command.extend(
        [
            "-i", input_pattern, "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-pix_fmt", "yuv420p", output_video_path,
        ]
    )

    print(f"Running ffmpeg command: {' '.join(command)}")
    try:
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        result = subprocess.run(
            command, check=True, capture_output=True, text=True,
            startupinfo=startupinfo, encoding="utf-8", errors='replace'
        )
        if result.stderr: print("ffmpeg stderr:", result.stderr)
        print(f"Video created successfully: {output_video_path}")
        return output_video_path
    except FileNotFoundError:
        print("\nERROR: ffmpeg command not found. Ensure ffmpeg is installed and in system PATH.")
        raise
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed with exit code {e.returncode}")
        print("ffmpeg stdout:", e.stdout)
        print("ffmpeg stderr:", e.stderr)
        print(f"Input pattern used: {input_pattern}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during video creation: {e}")
        raise

# -------------------------------------------------------------------
# HELPER: ComfyUI API Interaction
# -------------------------------------------------------------------

def get_comfyui_results_from_history(prompt_id, server_address, target_node_id):
    """Fetches prompt history via HTTP and extracts image data for a specific node."""
    images_data = None
    try:
        url = f"{server_address}/history/{prompt_id}"
        with urllib.request.urlopen(url, timeout=15) as response:
            if response.status == 200:
                history = json.loads(response.read())
                prompt_data = history.get(prompt_id)
                if not prompt_data: return None
                outputs = prompt_data.get("outputs")
                if not outputs: return None

                # Ensure target_node_id is string for dictionary lookup
                node_output = outputs.get(str(target_node_id))
                if node_output is None:
                    print(f"History outputs for prompt {prompt_id} do not contain node ID {target_node_id}.")
                    print(f"Available output nodes in history: {list(outputs.keys())}")
                    return [] # Node not found

                images_info = node_output.get("images")
                if images_info is None: return [] # No images key
                if not images_info: return [] # Empty images list

                print(f"Found {len(images_info)} images for node {target_node_id} in history.")
                images_data = []
                fetch_errors = 0
                for i, image_info in enumerate(images_info):
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    img_type = image_info.get("type")
                    if filename and img_type:
                        img_data = get_comfyui_image_data(filename, subfolder, img_type, server_address)
                        if img_data:
                            images_data.append(img_data)
                        else:
                            print(f"  ERROR: Failed retrieving image data for {filename} from history ref.")
                            fetch_errors += 1
                    else:
                        print(f"  Warning: Incomplete image info in history: {image_info}")
                        fetch_errors += 1
                if fetch_errors > 0:
                    print(f"Warning: Failed to fetch {fetch_errors} image(s) referenced in history.")
                return images_data

            elif response.status == 404: return None # Not finished or wrong ID
            else: print(f"Error fetching history {prompt_id}: HTTP Status {response.status}"); return None
    except urllib.error.URLError as e: print(f"URL Error fetching history {prompt_id}: {e}"); return None
    except json.JSONDecodeError as e: print(f"JSON Decode Error processing history {prompt_id}: {e}"); return None
    except TimeoutError: print(f"Timeout Error fetching history {prompt_id}."); return None
    except Exception as e: print(f"Unexpected error fetching/processing history {prompt_id}: {e}"); return None
    return images_data # Should be unreachable

def queue_comfyui_prompt(prompt_workflow, server_address, client_id):
    """Sends the workflow to the ComfyUI queue."""
    p = {"prompt": prompt_workflow, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    url = f"{server_address}/prompt"
    print(f"Queueing prompt to {url}")
    req = urllib.request.Request(url, data=data)
    try:
        response = urllib.request.urlopen(req, timeout=30) # Increased timeout
        return json.loads(response.read())
    except urllib.error.URLError as e:
        error_message = f"Error queueing prompt: {e}"
        if hasattr(e, "reason"): error_message += f" Reason: {e.reason}"
        if hasattr(e, "read"):
            try: error_message += f" Server Response: {e.read().decode()}"
            except Exception: pass
        print(error_message)
        raise ConnectionError(error_message) from e
    except json.JSONDecodeError as e: print(f"Error decoding queue response: {e}"); raise
    except TimeoutError: print("Timeout Error during prompt queueing."); raise
    except Exception as e: print(f"Unexpected error queueing prompt: {e}"); raise

def get_comfyui_image_data(filename, subfolder, image_type, server_address):
    """Fetches image data from ComfyUI's /view endpoint."""
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    url = f"{server_address}/view?{url_values}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response: # Increased timeout
            if response.status == 200: return response.read()
            else: print(f"Error fetching image {filename}: Status {response.status}"); return None
    except urllib.error.URLError as e: print(f"Error fetching image {filename}: {e}"); return None
    except TimeoutError: print(f"Timeout error fetching image {filename}."); return None
    except Exception as e: print(f"Unexpected error fetching image {filename}: {e}"); return None

def fetch_images_from_ws_or_history_data(
    data_source, source_type, prompt_id, server_address, target_node_id, progress_callback=None
):
    """Helper function to fetch image data given WS 'executed' data or History API data."""
    images_data = []
    fetch_errors = 0
    start_time = time.time()

    if source_type == "ws":
        node_outputs = data_source.get("outputs", {})
        images_info = node_outputs.get("images")
        source_name = "WebSocket message"
    elif source_type == "history":
        images_info = data_source # History function returns the images list
        source_name = "History API"
    else:
        print("Error: Invalid source_type for image fetching."); return []

    if not images_info:
        # This can be normal if the node didn't produce images
        # print(f"No 'images' key or empty list found in data from {source_name}.")
        return []

    print(f"Found {len(images_info)} image refs in {source_name}. Fetching data...")
    for i, image_info in enumerate(images_info):
        filename = image_info.get("filename")
        subfolder = image_info.get("subfolder", "")
        img_type = image_info.get("type")
        if filename and img_type:
            if progress_callback: progress_callback(f"Fetching image {i+1}/{len(images_info)}")
            img_data = get_comfyui_image_data(filename, subfolder, img_type, server_address)
            if img_data: images_data.append(img_data)
            else: print(f"  ERROR: Failed retrieving image data for {filename} (from {source_name})."); fetch_errors += 1
        else: print(f"  Warning: Incomplete image info in {source_name}: {image_info}"); fetch_errors += 1

    print(f"  Image fetching from {source_name} took {time.time() - start_time:.2f}s.")
    if fetch_errors > 0: print(f"Warning: Failed to fetch {fetch_errors} image(s) referenced in {source_name}.")
    return images_data

def get_comfyui_images_ws(
    prompt_id, server_address, client_id, target_node_id, progress_callback=None
):
    """
    Connects via WebSocket, waits for results from a specific node ID.
    Includes timeout, progress reporting, and fallback to /history API.
    """
    if not websocket: raise RuntimeError("websocket-client package not loaded.")

    # Ensure target_node_id is an integer for comparison with WebSocket message data
    try:
        target_node_id_int = int(target_node_id)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid target_node_id for WebSocket: '{target_node_id}'. Must be convertible to int.")

    ws_url = f"ws://{server_address.split('//')[1]}/ws?clientId={client_id}"
    print(f"Connecting to WebSocket: {ws_url}")
    ws = websocket.WebSocket()

    try: ws.connect(ws_url, timeout=20)
    except websocket.WebSocketTimeoutException: raise ConnectionError(f"WebSocket connection timed out: {ws_url}") from None
    except ConnectionRefusedError: raise ConnectionError(f"WebSocket connection refused: {ws_url}") from None
    except websocket.WebSocketException as e: raise ConnectionError(f"WebSocket connection failed: {e}") from e
    except Exception as e: raise ConnectionError(f"WebSocket connection failed unexpectedly: {e}") from e
    print("WebSocket connected.")

    images_data = None
    prompt_execution_finished_signal = False
    output_node_executed_signal = False
    retrieved_via_history = False
    consecutive_timeouts = 0
    max_consecutive_timeouts_before_warn = 5
    max_consecutive_timeouts_overall = 20
    overall_timeout_seconds = 1200 # Increased overall timeout (20 minutes)
    ws_receive_timeout = 15
    start_time = time.time()
    last_message_time = start_time

    try:
        while images_data is None and (time.time() - start_time < overall_timeout_seconds):
            try:
                ws.settimeout(ws_receive_timeout)
                out = ws.recv()
                consecutive_timeouts = 0
                last_message_time = time.time()
            except websocket.WebSocketTimeoutException:
                consecutive_timeouts += 1
                if consecutive_timeouts >= max_consecutive_timeouts_overall:
                    print(f"Exceeded max consecutive WebSocket timeouts ({max_consecutive_timeouts_overall}). Trying history...")
                    history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
                    if history_result:
                        print("Final history check after max timeouts succeeded.")
                        images_data = history_result; retrieved_via_history = True; break
                    else:
                        raise TimeoutError(f"WebSocket stopped receiving messages. Final history check failed.")

                should_check_history = False
                if prompt_execution_finished_signal and not output_node_executed_signal: should_check_history = True
                elif consecutive_timeouts >= max_consecutive_timeouts_before_warn: should_check_history = True

                if should_check_history:
                    history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
                    if history_result is not None:
                        if history_result:
                            print(f"Received {len(history_result)} images from /history API fallback.")
                            images_data = history_result; retrieved_via_history = True; break
                        else:
                            if prompt_execution_finished_signal:
                                print(f"/history confirms finished but node {target_node_id} has no image output.")
                                images_data = []; break
                            # else: History exists but node/images not found yet, continue WS

                if time.time() - last_message_time > 60:
                    try: ws.ping(); last_message_time = time.time()
                    except websocket.WebSocketConnectionClosedException:
                        print("WebSocket closed while idle (ping detected). Trying history...")
                        history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
                        images_data = history_result if history_result else []
                        if history_result: retrieved_via_history = True
                        break
                    except Exception as ping_e: print(f"Error sending WebSocket ping: {ping_e}")
                continue # Go to next loop iteration after handling timeout

            except websocket.WebSocketConnectionClosedException:
                print("WebSocket connection closed by server. Trying history...")
                history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
                images_data = history_result if history_result else []
                if history_result: retrieved_via_history = True
                break

            # --- Process Received Message ---
            if isinstance(out, str):
                try: message = json.loads(out)
                except json.JSONDecodeError: print(f"Error decoding WS message: {out[:200]}..."); continue

                msg_type = message.get("type")
                data = message.get("data", {})

                if msg_type == "status":
                    queue_remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining", 0)
                    if progress_callback: progress_callback(f"Queue: {queue_remaining}")
                elif msg_type == "progress":
                    value = data.get("value", 0); max_val = data.get("max", 0)
                    if max_val > 0 and progress_callback:
                        percent = int((value / max_val) * 100)
                        progress_callback(f"Executing: {percent}% ({value}/{max_val})")
                elif msg_type == "executing":
                    node_id_msg = data.get("node") # Is None for final signal
                    prompt_id_msg = data.get("prompt_id")
                    if node_id_msg is None and prompt_id_msg == prompt_id:
                        print("Execution finished signal received (node=None).")
                        prompt_execution_finished_signal = True
                    elif node_id_msg == target_node_id_int and prompt_id_msg == prompt_id:
                        print(f"Target node {target_node_id} is executing...")
                        if progress_callback: progress_callback(f"Node {target_node_id} running")
                elif msg_type == "executed":
                    node_id_msg = data.get("node_id") # Note: 'node_id' here, vs 'node' above
                    prompt_id_msg = data.get("prompt_id")
                    if node_id_msg == target_node_id_int and prompt_id_msg == prompt_id:
                        print(f"Target node {target_node_id} finished execution (message received).")
                        output_node_executed_signal = True
                        images_data = fetch_images_from_ws_or_history_data(
                            data, "ws", prompt_id, server_address, target_node_id, progress_callback)
                        retrieved_via_history = False
                        break # Exit loop, have results via WS

    except ConnectionAbortedError as e:
        print(f"WebSocket Error: {e}. Trying history...")
        history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
        if history_result: print("History check after WS ConnectionAbortedError succeeded."); images_data = history_result; retrieved_via_history = True
        else: raise # Re-raise if history also fails
    except TimeoutError as e: # Catch specific timeout from inner logic
        print(f"Operation Timeout Error: {e}"); raise # History check already done inside timeout logic
    except Exception as e:
        print(f"Unexpected error during WebSocket processing: {e}"); import traceback; traceback.print_exc()
        history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
        if history_result: print("Final history check after unexpected WS error succeeded."); images_data = history_result; retrieved_via_history = True
        else: raise RuntimeError(f"Unexpected WS error & history fallback failed: {e}") from e
    finally:
        if ws and ws.connected: ws.close(); print("WebSocket closed.")

    # --- Post-Loop / Final Checks ---
    if images_data is None and (time.time() - start_time < overall_timeout_seconds):
        print("WebSocket loop finished without direct results. Final /history check.")
        history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
        if history_result is not None:
            images_data = history_result
            if history_result: print(f"Final history check successful ({len(history_result)} images)."); retrieved_via_history = True
            else: print("Final history check returned empty list.")
        else: print("Final history check failed."); images_data = []
    elif images_data is None and (time.time() - start_time >= overall_timeout_seconds):
        print(f"ERROR: Overall timeout ({overall_timeout_seconds}s) waiting for prompt {prompt_id}.")
        history_result = get_comfyui_results_from_history(prompt_id, server_address, target_node_id)
        if history_result: print(f"Final history check after overall timeout succeeded ({len(history_result)} images)."); images_data = history_result; retrieved_via_history = True
        else: raise TimeoutError(f"Overall timeout for prompt {prompt_id}. Final history check failed.")

    if images_data is None: images_data = []
    log_source = "History API fallback" if retrieved_via_history else "WebSocket"
    if not images_data: print(f"Warning: Operation finished, no images retrieved via {log_source}.")
    else: print(f"Info: Images retrieved via {log_source} ({len(images_data)} images).")
    return images_data

# -------------------------------------------------------------------
# OPERATOR: Main Modal Operator
# -------------------------------------------------------------------
class OBJECT_OT_run_comfyui_modal(bpy.types.Operator):
    bl_idname = "object.run_comfyui_modal"
    bl_label = "Run ComfyUI GPencil Workflow" # Simplified Label
    bl_description = (
        "Capture frames, run selected ComfyUI workflow, create GP references. "
        "Requires ACTIVE GP OBJECT and SPA addon features. Runs in background."
    )
    bl_options = {"REGISTER", "UNDO"}

    # --- Modal State Variables ---
    _timer = None
    _thread = None
    _result_queue = None
    _thread_status: bpy.props.StringProperty(options={"SKIP_SAVE"})
    _thread_progress: bpy.props.StringProperty(options={"SKIP_SAVE"})

    # --- Stored State for Thread & Cleanup ---
    temp_dir_obj = None
    temp_dir_path = None
    frames_to_process = []
    frame_paths = {} # Store paths by frame number: {1: "path/f_0001.png", ...}
    frame_pattern = "frame_%04d.png"
    temp_video_path = None
    workflow = None # Loaded workflow dictionary
    workflow_type = None # 'SINGLE', 'MULTI_REF', 'MULTI_CONTEXT'
    node_ids = {} # Node ID mapping for the loaded workflow
    client_id = None
    original_frame = 0
    previous_obj = None
    previous_mode = None
    final_result = None
    output_img_paths = [] # Store paths of images SAVED by the addon locally
    selected_lora_name: bpy.props.StringProperty(options={"SKIP_SAVE"})
    selected_lora_strength_model: bpy.props.FloatProperty(options={"SKIP_SAVE"}) 
    selected_lora_strength_clip: bpy.props.FloatProperty(options={"SKIP_SAVE"})
    server_address = ""
    target_gp_object_name: bpy.props.StringProperty(options={"SKIP_SAVE"})

    # Store settings needed by thread
    scene_user_prompt: bpy.props.StringProperty(options={"SKIP_SAVE"})
    scene_frame_rate: bpy.props.IntProperty(options={"SKIP_SAVE"})
    scene_depth_strength: bpy.props.FloatProperty(options={"SKIP_SAVE"}) # Still needed? Check workflows
    scene_invert_depth: bpy.props.BoolProperty(options={"SKIP_SAVE"})
    scene_reference_image_path: bpy.props.StringProperty(options={"SKIP_SAVE"})
    scene_use_preceding: bpy.props.BoolProperty(options={"SKIP_SAVE"})
    scene_use_succeeding: bpy.props.BoolProperty(options={"SKIP_SAVE"})
    preceding_frame_path: bpy.props.StringProperty(options={"SKIP_SAVE"})
    succeeding_frame_path: bpy.props.StringProperty(options={"SKIP_SAVE"})

    def _get_first_lora_fallback(self, context):
        """Gets the filename of the first available LoRA as a fallback."""
        items = get_lora_items(self, context) # Call the dynamic enum function
        for identifier, name, description in items:
            if identifier not in ["NONE", "INVALID_PATH", "SCAN_ERROR"]:
                return identifier # Return the first valid filename
        return None # No valid LoRAs found

    @classmethod
    def poll(cls, context):
        # Base checks
        if not isinstance(context.region.data, bpy.types.RegionView3D): cls.poll_message_set("Requires a 3D Viewport"); return False
        if not websocket: cls.poll_message_set("Missing 'websocket-client' package"); return False
        if not SPA_GP_AVAILABLE: cls.poll_message_set("Requires SPA Studios Grease Pencil addon"); return False

        # Active GP Object check
        active_obj = context.active_object
        if not active_obj or active_obj.type != 'GPENCIL': cls.poll_message_set("Requires an active Grease Pencil object"); return False
        if not active_obj.data: cls.poll_message_set("Active GP object has no data"); return False

        # Check if running
        wm = context.window_manager
        if getattr(wm, "comfyui_modal_operator_running", False): return False

        # Check workflow files exist (basic check)
        if not os.path.isdir(WORKFLOW_DIR): cls.poll_message_set(f"Workflow dir missing: {WORKFLOW_DIR}"); return False
        # Could add check for specific needed file based on current UI state, but might be overkill for poll

        scene_props = context.scene.comfyui_props
        if scene_props.frame_mode == "CURRENT":
            prefs = context.preferences.addons[__name__].preferences
            lora_dir = prefs.lora_directory
            if not lora_dir or not os.path.isdir(lora_dir):
                cls.poll_message_set("LoRA directory invalid/not set in preferences")
                return False
            # Check if selected lora is an error indicator
            if scene_props.selected_lora in ["INVALID_PATH", "SCAN_ERROR"]:
                 cls.poll_message_set(f"LoRA Selection Error: {scene_props.selected_lora}")
                 return False

        return True


    def _comfyui_worker_thread(self):
        """
        Worker thread: Creates video (if needed), modifies workflow, queues, waits for results.
        """
        try:
            # --- 1) Prepare Inputs (Video/Image Paths) ---
            if self.workflow_type == "SINGLE":
                # Single frame: Input is the captured image path
                if not self.frames_to_process: raise ValueError("No frame captured for single frame mode.")
                frame_num = self.frames_to_process[0]
                input_image_path = self.frame_paths.get(frame_num)
                if not input_image_path or not os.path.exists(input_image_path):
                     raise ValueError(f"Input image path not found for frame {frame_num}.")
                abs_input_path = os.path.abspath(input_image_path).replace("\\", "/")
                print(f"Using single frame input: {abs_input_path}")
                self.workflow[self.node_ids["INPUT_IMAGE"]]["inputs"]["image"] = abs_input_path
            else:
                # Multi frame: Input is a video
                self._thread_status = "Creating temporary video..."
                self._thread_progress = ""
                start_num = self.frames_to_process[0] if self.frames_to_process else 0
                video_filename = f"input_video_{start_num}.mp4"
                self.temp_video_path = os.path.join(self.temp_dir_path, video_filename)

                create_video_from_frames(
                    self.temp_dir_path, self.temp_video_path,
                    self.scene_frame_rate, self.frame_pattern, start_number=start_num
                )
                abs_video_path = os.path.abspath(self.temp_video_path).replace("\\", "/")
                print(f"Using video input: {abs_video_path}")
                # Set video path in workflow
                self.workflow[self.node_ids["INPUT_VIDEO"]]["inputs"]["video"] = abs_video_path
                # Set frame load cap/skip based on original frames? (Workflows have defaults)
                # Example: Could set frame_load_cap = len(self.frames_to_process) if node supports it
                if "frame_load_cap" in self.workflow[self.node_ids["INPUT_VIDEO"]]["inputs"]:
                     self.workflow[self.node_ids["INPUT_VIDEO"]]["inputs"]["frame_load_cap"] = len(self.frames_to_process)
                     print(f"  Set frame_load_cap to {len(self.frames_to_process)}")
                # else: node doesn't support it or isn't VHS Load Video Path type

            # --- 2) Set Common Parameters (Prompt, Invert, etc.) ---
            # Prompt (adjust based on workflow structure)
            if "PROMPT" in self.node_ids:
                prompt_node_id = self.node_ids["PROMPT"]
                if prompt_node_id in self.workflow:
                    # Single frame workflow has separate positive prompt node
                    if self.workflow_type == "SINGLE" and "text" in self.workflow[prompt_node_id]["inputs"]:
                        self.workflow[prompt_node_id]["inputs"]["text"] = self.scene_user_prompt
                        print(f"  Set Prompt (Node {prompt_node_id}): {self.scene_user_prompt[:30]}...")
                    # Multi frame workflows use WanVideoTextEncode with positive/negative
                    elif self.workflow_type in ["MULTI_REF", "MULTI_CONTEXT"] and "positive_prompt" in self.workflow[prompt_node_id]["inputs"]:
                         self.workflow[prompt_node_id]["inputs"]["positive_prompt"] = self.scene_user_prompt
                         # You might want a separate negative prompt UI element later
                         # self.workflow[prompt_node_id]["inputs"]["negative_prompt"] = self.scene_negative_prompt
                         print(f"  Set Positive Prompt (Node {prompt_node_id}): {self.scene_user_prompt[:30]}...")
                else: print(f"  Warning: Prompt Node ID '{prompt_node_id}' not found in workflow.")

            # Invert Input Boolean
            if "INVERT_BOOL" in self.node_ids:
                 invert_node_id = self.node_ids["INVERT_BOOL"]
                 if invert_node_id in self.workflow and "inputs" in self.workflow[invert_node_id]:
                      self.workflow[invert_node_id]["inputs"]["value"] = self.scene_invert_depth
                      print(f"  Set Invert Input Bool (Node {invert_node_id}): {self.scene_invert_depth}")
                 else: print(f"  Warning: Invert Bool Node ID '{invert_node_id}' not found/invalid.")

            # ControlNet Strength (Only for Single Frame workflow in this setup)
            if self.workflow_type == "SINGLE" and "CNET_STRENGTH" in self.node_ids:
                 cn_strength_node_id = self.node_ids["CNET_STRENGTH"]
                 if cn_strength_node_id in self.workflow and "inputs" in self.workflow[cn_strength_node_id]:
                      # The node is ControlNetApplyAdvanced, strength is direct input
                      self.workflow[cn_strength_node_id]["inputs"]["strength"] = self.scene_depth_strength # Assuming scene_depth_strength maps here
                      print(f"  Set ControlNet Strength (Node {cn_strength_node_id}): {self.scene_depth_strength}")
                 else: print(f"  Warning: CNet Strength Node ID '{cn_strength_node_id}' not found/invalid.")

            # --- Handle LoRA for SINGLE frame ---
            if self.workflow_type == "SINGLE":
                lora_node_id = self.node_ids.get("LORA_LOADER")
                if lora_node_id and lora_node_id in self.workflow:
                    lora_inputs = self.workflow[lora_node_id]["inputs"]
                    # Check if a valid LoRA was selected (not NONE or an error indicator)
                    if self.selected_lora_name and self.selected_lora_name != "NONE":
                        lora_inputs["lora_name"] = self.selected_lora_name
                        lora_inputs["strength_model"] = self.selected_lora_strength_model
                        lora_inputs["strength_clip"] = self.selected_lora_strength_clip
                        print(f"  Using LoRA (Node {lora_node_id}): {self.selected_lora_name} (Str M:{self.selected_lora_strength_model:.2f} C:{self.selected_lora_strength_clip:.2f})")
                    else:
                        # No LoRA selected or an error occurred - Disable LoRA effect
                        lora_inputs["strength_model"] = 0.0
                        lora_inputs["strength_clip"] = 0.0
                        # Keep a default lora_name, ComfyUI might require *something*
                        # If the original workflow had a default, it will be used.
                        # Otherwise, keep whatever was loaded. We could try setting a fallback if needed.
                        print(f"  LoRA Disabled (Node {lora_node_id}): Setting strength to 0.0")
                        # Optional: Add a fallback name if lora_name might be missing
                        # if "lora_name" not in lora_inputs or not lora_inputs["lora_name"]:
                        #     fallback_lora = self._get_first_lora_fallback(bpy.context) # Need context here? Maybe pass it or get during invoke
                        #     if fallback_lora:
                        #         lora_inputs["lora_name"] = fallback_lora
                        #         print(f"    Set fallback LoRA name: {fallback_lora}")
                        #     else:
                        #         print(f"    Warning: Could not set a fallback LoRA name.")

                else:
                    print(f"  Warning: LoRA Loader Node ID '{lora_node_id}' not found in workflow.")

            # --- 3) Set Workflow-Specific Parameters ---
            if self.workflow_type == "MULTI_REF":
                if "REFERENCE_IMAGE" in self.node_ids:
                    ref_img_node_id = self.node_ids["REFERENCE_IMAGE"]
                    if not self.scene_reference_image_path or not os.path.exists(self.scene_reference_image_path):
                        # Default to a black image if path is invalid? Or raise error?
                        # For now, warn and maybe let ComfyUI handle missing input.
                        print(f"  Warning: Reference image path invalid or not set: {self.scene_reference_image_path}")
                        # Optionally: Set a dummy path or remove the input link if possible
                        # self.workflow[ref_img_node_id]["inputs"]["image"] = "path/to/dummy/black.png"
                    else:
                        abs_ref_path = os.path.abspath(self.scene_reference_image_path).replace("\\", "/")
                        self.workflow[ref_img_node_id]["inputs"]["image"] = abs_ref_path
                        print(f"  Set Reference Image (Node {ref_img_node_id}): {abs_ref_path}")
                else: print(f"  Warning: Reference Image Node ID not defined for MULTI_REF workflow.")

            elif self.workflow_type == "MULTI_CONTEXT":
                # Set Preceding/Succeeding Image Paths (even if not used by bools)
                if "PRECEDING_IMAGE" in self.node_ids:
                     prec_img_node_id = self.node_ids["PRECEDING_IMAGE"]
                     if self.preceding_frame_path and os.path.exists(self.preceding_frame_path):
                          abs_prec_path = os.path.abspath(self.preceding_frame_path).replace("\\", "/")
                          self.workflow[prec_img_node_id]["inputs"]["image"] = abs_prec_path
                          print(f"  Set Preceding Image (Node {prec_img_node_id}): {abs_prec_path}")
                     else: print(f"  Warning: Preceding frame path invalid/missing for Node {prec_img_node_id}.") # Workflow might require it
                else: print(f"  Warning: Preceding Image Node ID not defined.")

                if "SUCCEEDING_IMAGE" in self.node_ids:
                     succ_img_node_id = self.node_ids["SUCCEEDING_IMAGE"]
                     if self.succeeding_frame_path and os.path.exists(self.succeeding_frame_path):
                          abs_succ_path = os.path.abspath(self.succeeding_frame_path).replace("\\", "/")
                          self.workflow[succ_img_node_id]["inputs"]["image"] = abs_succ_path
                          print(f"  Set Succeeding Image (Node {succ_img_node_id}): {abs_succ_path}")
                     else: print(f"  Warning: Succeeding frame path invalid/missing for Node {succ_img_node_id}.")
                else: print(f"  Warning: Succeeding Image Node ID not defined.")

                # Set Boolean Flags for Context Usage
                if "USE_PRECEDING_BOOL" in self.node_ids:
                     use_prec_node_id = self.node_ids["USE_PRECEDING_BOOL"]
                     if use_prec_node_id in self.workflow and "inputs" in self.workflow[use_prec_node_id]:
                          self.workflow[use_prec_node_id]["inputs"]["value"] = self.scene_use_preceding
                          print(f"  Set Use Preceding Bool (Node {use_prec_node_id}): {self.scene_use_preceding}")
                     else: print(f"  Warning: Use Preceding Bool Node ID '{use_prec_node_id}' not found/invalid.")
                else: print(f"  Warning: Use Preceding Bool Node ID not defined.")

                if "USE_SUCCEEDING_BOOL" in self.node_ids:
                     use_succ_node_id = self.node_ids["USE_SUCCEEDING_BOOL"]
                     if use_succ_node_id in self.workflow and "inputs" in self.workflow[use_succ_node_id]:
                          self.workflow[use_succ_node_id]["inputs"]["value"] = self.scene_use_succeeding
                          print(f"  Set Use Succeeding Bool (Node {use_succ_node_id}): {self.scene_use_succeeding}")
                     else: print(f"  Warning: Use Succeeding Bool Node ID '{use_succ_node_id}' not found/invalid.")
                else: print(f"  Warning: Use Succeeding Bool Node ID not defined.")

                if "VACE_STARTEND_1" in self.node_ids and "VACE_STARTEND_2" in self.node_ids and "VACE_STARTEND_3" in self.node_ids:
                    num_frames = len(self.frames_to_process) + int(self.scene_use_preceding) + int(self.scene_use_succeeding)
                    self.workflow[self.node_ids["VACE_STARTEND_1"]]["inputs"]["num_frames"] = num_frames
                    self.workflow[self.node_ids["VACE_STARTEND_2"]]["inputs"]["num_frames"] = num_frames
                    self.workflow[self.node_ids["VACE_STARTEND_3"]]["inputs"]["num_frames"] = num_frames
                else: print(f"  Warning: VACE_STARTEND Node IDs not defined")

            # --- 4) Queue and Wait ---
            self._thread_status = f"Queueing ComfyUI ({self.workflow_type})"
            self._thread_progress = ""
            print("\nFinal Workflow (first level inputs):")
            for node_id, node_data in self.workflow.items():
                if "inputs" in node_data:
                    print(f"  Node {node_id} ({node_data.get('class_type', 'Unknown')}): {node_data['inputs']}")
                else:
                    print(f"  Node {node_id} ({node_data.get('class_type', 'Unknown')}): No inputs key")
            print("-" * 20)


            queue_response = queue_comfyui_prompt(
                self.workflow, self.server_address, self.client_id
            )
            prompt_id = queue_response.get("prompt_id")
            if not prompt_id: raise RuntimeError(f"Did not receive prompt_id. Response: {queue_response}")
            print(f"ComfyUI Prompt ID: {prompt_id}")

            self._thread_status = f"Waiting for results (Prompt: {prompt_id[:8]}...)"
            self._thread_progress = ""

            def progress_update(message):
                # Update shared status variables for modal UI
                self._thread_status = f"Waiting (Prompt: {prompt_id[:8]}...)" # Keep status simple
                self._thread_progress = message # Show detailed progress

            # Get the specific output node ID for the current workflow
            output_node_id = self.node_ids.get("OUTPUT_DECODE") # Use the decode node
            if not output_node_id:
                 output_node_id = self.node_ids.get("SAVE_IMAGE") # Fallback for single?
            if not output_node_id:
                 raise ValueError(f"Could not determine output node ID for workflow type {self.workflow_type}")
            print(f"Waiting for results from Node ID: {output_node_id}")

            output_images_data = get_comfyui_images_ws(
                prompt_id, self.server_address, self.client_id,
                output_node_id, # Pass the dynamically determined output node
                progress_callback=progress_update
            )

            # --- Success ---
            self._thread_status = "Received results from ComfyUI."
            self._thread_progress = f"{len(output_images_data)} image(s)"
            self._result_queue.put(output_images_data) # Put list of image data bytes

        except Exception as e:
            # --- Failure ---
            error_short = str(e).splitlines()[0] if str(e) else type(e).__name__
            self._thread_status = "Error during ComfyUI interaction."
            self._thread_progress = f"{type(e).__name__}: {error_short}"
            print(f"Error in worker thread: {e}")
            import traceback
            traceback.print_exc()
            self._result_queue.put(e) # Put the exception object

    # --- Main Execution Logic (Called by Modal on Completion) ---
    def execute_finish(self, context):
        """ Processes results, creates GPencil References, handles context workflow output. """
        print("Entering Execute Finish method (GPencil)...")
        wm = context.window_manager

        # --- Check Target GP Object ---
        gp_object = bpy.data.objects.get(self.target_gp_object_name)
        if not gp_object or gp_object.type != 'GPENCIL' or not gp_object.data:
            self.report({"ERROR"}, f"Target GP object '{self.target_gp_object_name}' not found/invalid.")
            wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = "Target GP object lost"
            return {"CANCELLED"}
        gpd = gp_object.data

        # --- Handle Final Result (Error or Image List) ---
        if isinstance(self.final_result, Exception):
            error_short = str(self.final_result).splitlines()[0] if str(self.final_result) else type(self.final_result).__name__
            self.report({"ERROR"}, f"Worker thread error: {type(self.final_result).__name__}: {error_short}")
            wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = f"{type(self.final_result).__name__}: {error_short}"
            return {"CANCELLED"}
        elif self.final_result is None:
             self.report({"WARNING"}, "No valid response from ComfyUI."); wm.comfyui_modal_status = "Finished (No Response)"; wm.comfyui_modal_progress = ""
             return {"FINISHED"}
        elif not isinstance(self.final_result, list):
             self.report({"ERROR"}, f"Unexpected result type: {type(self.final_result)}"); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = f"Internal error: Bad result type"
             return {"CANCELLED"}
        # Empty list is now handled below after slicing

        # --- Prepare for GP Reference Creation ---
        print(f"Execute Finish received {len(self.final_result)} raw image(s) for workflow {self.workflow_type}.")
        self.output_img_paths = []

        # --- Adjust Image List based on Context Workflow ---
        images_to_process = self.final_result
        expected_frame_count = len(self.frames_to_process)
        frames_for_gp = list(self.frames_to_process) # Copy the list

        if self.workflow_type == "MULTI_CONTEXT":
            print("Context workflow detected, adjusting output image list...")
            start_index = 0
            end_index = len(self.final_result)

            if self.scene_use_preceding:
                if len(images_to_process) > 0:
                    print("  Skipping first image (preceding frame context).")
                    start_index = 1
                else: print("  Warning: Expected preceding frame in output, but received no images.")

            if self.scene_use_succeeding:
                 if len(images_to_process) > start_index : # Check if there's at least one image left to potentially remove
                    print("  Skipping last image (succeeding frame context).")
                    end_index -= 1
                 else: print("  Warning: Expected succeeding frame in output, but not enough images received.")

            # Ensure start_index is not greater than end_index
            if start_index >= end_index:
                 print(f"  Warning: After slicing for context frames, no images remain (start: {start_index}, end: {end_index}).")
                 images_to_process = []
            else:
                 images_to_process = self.final_result[start_index:end_index]

            print(f"  Sliced images: {len(images_to_process)} image(s) remain for GP references.")

            # --- Sanity Check: Match sliced images count with original frame request count ---
            if len(images_to_process) != expected_frame_count:
                 print(f"  WARNING: Number of images after slicing ({len(images_to_process)}) "
                       f"does not match the number of requested frames ({expected_frame_count}). "
                       f"GP frame assignment might be incorrect.")
                 # Attempt to truncate frames_for_gp if too long, or warn if too short
                 if len(frames_for_gp) > len(images_to_process):
                      frames_for_gp = frames_for_gp[:len(images_to_process)]
                      print(f"  Adjusted frames for GP to {len(frames_for_gp)} based on available images.")

        # --- Handle Empty Process List ---
        if not images_to_process:
             self.report({"INFO"}, "No images generated/retrieved or remaining after context slicing."); wm.comfyui_modal_status = "Finished (No Images)"; wm.comfyui_modal_progress = ""
             return {"FINISHED"}

        self.report({"INFO"}, f"Processing {len(images_to_process)} image(s). Creating GP references...")
        wm.progress_begin(0, len(images_to_process))

        # --- Find a suitable 3D Viewport context ---
        view3d_area = None; view3d_region = None
        # (Context finding logic - unchanged from previous)
        if context.area and context.area.type == 'VIEW_3D':
            view3d_area = context.area; view3d_region = next((r for r in view3d_area.regions if r.type == 'WINDOW'), None)
        if not (view3d_area and view3d_region):
            for area in context.screen.areas:
                 if area.type == 'VIEW_3D':
                     view3d_area = area; view3d_region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                     if view3d_region: break
        if not (view3d_area and view3d_region):
             self.report({"ERROR"}, "Could not find a suitable 3D Viewport context."); wm.progress_end(); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = "Context Error (No 3D View)"
             return {"CANCELLED"}

        original_active_at_start = context.active_object
        original_mode_at_start = original_active_at_start.mode if original_active_at_start else "OBJECT"
        original_scene_frame = context.scene.frame_current
        success_count = 0
        num_to_process = len(images_to_process)

        print(f"Using context override: Area='{view3d_area.type}', Region='{view3d_region.type}'")
        try: # Ensure mode is reset
            with context.temp_override(window=context.window, area=view3d_area, region=view3d_region):
                # --- Ensure correct context and GP Edit mode ---
                print("Setting active object and GP Edit mode...")
                context.view_layer.objects.active = gp_object
                gp_object.select_set(True)
                try:
                    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
                    if context.object.mode != 'EDIT_GPENCIL': raise RuntimeError("Failed to enter GP Edit Mode.")
                    print(f"  Mode set to: {context.object.mode}")
                except Exception as mode_e:
                     self.report({"ERROR"}, f"Could not set GP Edit mode: {mode_e}. Aborting."); wm.progress_end(); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = f"Mode Error: {mode_e}"
                     try: bpy.ops.object.mode_set(mode='OBJECT')
                     except: pass
                     return {"CANCELLED"}

                # --- Get or Create Target GP Layer ---
                target_layer_name = "ComfyUI Output"
                target_layer = gpd.layers.get(target_layer_name)
                if not target_layer: target_layer = gpd.layers.new(target_layer_name, set_active=True); print(f"Created GP layer: {target_layer_name}")
                else: gpd.layers.active = target_layer; print(f"Using existing GP layer: {target_layer_name}")

                # --- Loop through images to process ---
                for i, img_data in enumerate(images_to_process):
                    # Determine frame number carefully
                    if i < len(frames_for_gp):
                        frame_num = frames_for_gp[i]
                    else:
                        # Fallback if frame list doesn't match image list (shouldn't happen with checks above)
                        frame_num = self.original_frame + i
                        print(f"  Warning: Using fallback frame num {frame_num} for image index {i}.")

                    current_status = f"Processing output {i+1}/{num_to_process}"; current_progress = f"Frame {frame_num}"
                    wm.comfyui_modal_status = current_status; wm.comfyui_modal_progress = current_progress
                    try: context.workspace.status_text_set(f"ComfyUI: {current_status} ({current_progress})")
                    except: pass

                    if not isinstance(img_data, bytes) or not img_data:
                        self.report({"WARNING"}, f"Skipping invalid image data at index {i} (for frame {frame_num})."); wm.progress_update(i + 1); continue

                    # Save image temporarily
                    output_img_filename = f"comfy_out_{frame_num:04d}.png"
                    if not self.temp_dir_path or not os.path.isdir(self.temp_dir_path):
                         self.report({"ERROR"}, f"Temp directory missing. Aborting."); wm.comfyui_modal_status = "Finished with Error"; wm.comfyui_modal_progress = "Temp dir lost"; break
                    output_img_path = os.path.join(self.temp_dir_path, output_img_filename)
                    print(f"\nProcessing image {i+1}/{num_to_process} for frame {frame_num} -> GP Ref...")

                    try:
                        print(f"  Saving image data to: {output_img_path}")
                        with open(output_img_path, "wb") as f: f.write(img_data)
                        if not os.path.exists(output_img_path) or os.path.getsize(output_img_path) == 0: raise IOError("Temp image file not found/empty.")
                        self.output_img_paths.append(output_img_path)

                        # Set Current Frame in Blender
                        print(f"  Setting scene frame to: {frame_num}")
                        context.scene.frame_set(frame_num)

                        # Call SPA GP Reference Import
                        print(f"  Calling import_image_as_gp_reference for '{output_img_path}'...")
                        # *** NOTE: SPA function might resize based on camera view, ignoring the 960x720 output aspect ratio ***
                        # This is likely the desired behavior - fitting the reference to the current camera.
                        spa_gp_core.import_image_as_gp_reference(
                            context=context, obj=gp_object, img_filepath=output_img_path,
                            pack_image=True, add_new_layer=False, add_new_keyframe=True,
                        )
                        print(f"  Successfully created GP reference for frame {frame_num}.")
                        success_count += 1

                    except Exception as e:
                        self.report({"ERROR"}, f"Failed creating GP reference for image {i} (frame {frame_num}): {e}")
                        import traceback; traceback.print_exc()
                        if os.path.exists(output_img_path): 
                            try: os.remove(output_img_path); 
                            except OSError: pass
                    finally:
                         wm.progress_update(i + 1)
            # --- End For Loop ---
        finally:
            # --- Ensure mode is reset to OBJECT ---
            print("Attempting to restore OBJECT mode...")
            try:
                if context.view_layer.objects.active == gp_object and gp_object.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT'); print("  Mode restored to OBJECT.")
                elif context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT'); print("  Mode restored to OBJECT (fallback).")
            except Exception as mode_reset_e: print(f"  Warning: Could not restore OBJECT mode: {mode_reset_e}")
        # --- End Context Override ---

        # --- Restore original frame ---
        if context.scene and context.scene.frame_start <= original_scene_frame <= context.scene.frame_end: context.scene.frame_set(original_scene_frame)
        wm.progress_end()

        # --- Restore original mode if different from OBJECT ---
        # (Mode restoration logic - unchanged from previous)
        if original_mode_at_start != 'OBJECT':
            print(f"Attempting to restore original mode ({original_mode_at_start}) for object '{gp_object.name}'...")
            try:
                if gp_object and gp_object.name in context.view_layer.objects:
                     context.view_layer.objects.active = gp_object
                     valid_modes = {"OBJECT", "EDIT", "SCULPT", "PAINT", "WEIGHT", "EDIT_GPENCIL", "SCULPT_GPENCIL", "PAINT_GPENCIL", "WEIGHT_GPENCIL"}
                     if original_mode_at_start in valid_modes: bpy.ops.object.mode_set(mode=original_mode_at_start); print(f"  Restored original mode to: {original_mode_at_start}")
                     else: print(f"  Cannot restore non-standard original mode: '{original_mode_at_start}'.")
                else: print(f"  Cannot restore original mode, GP object '{self.target_gp_object_name}' not found.")
            except Exception as e: print(f"Could not restore original mode {original_mode_at_start}: {e}")


        final_op_status = f"Successfully created {success_count} / {num_to_process} GP references in '{gp_object.name}'."
        self.report({"INFO"}, final_op_status)
        wm.comfyui_modal_status = "Finished"
        wm.comfyui_modal_progress = final_op_status
        print("Exiting Execute Finish method (GPencil).")
        return {"FINISHED"}

    # --- Modal Method (Handles Timer, Cancel, Thread Check) ---
    def modal(self, context, event):
        wm = context.window_manager

        # --- Check thread completion first ---
        if self._thread and self.final_result is None and not self._thread.is_alive():
            print(f"Worker thread finished (detected on event: {event.type}).")
            try: self.final_result = self._result_queue.get_nowait()
            except queue.Empty: self.report({"ERROR"}, "Thread finished but queue empty."); self.final_result = RuntimeError("Queue empty.")
            except Exception as e: self.report({"ERROR"}, f"Error getting result from queue: {e}"); self.final_result = e

            if self.final_result is not None:
                final_status = self.execute_finish(context)
                return self.finish_or_cancel(context, cancelled=("CANCELLED" in final_status))

        # --- Handle Events ---
        if context.area: context.area.tag_redraw() # Force UI update

        if event.type == "ESC" or not getattr(wm, "comfyui_modal_operator_running", True):
            print("ESC pressed or operator cancelled externally.")
            return self.finish_or_cancel(context, cancelled=True)

        if event.type == "TIMER":
            # Update UI from thread status variables if thread is running
            if self._thread and self.final_result is None:
                current_status = self._thread_status; current_progress = self._thread_progress
                if wm.comfyui_modal_status != current_status: wm.comfyui_modal_status = current_status
                if wm.comfyui_modal_progress != current_progress: wm.comfyui_modal_progress = current_progress
                status_bar_text = f"ComfyUI: {current_status}" + (f" ({current_progress})" if current_progress else "")
                context.workspace.status_text_set(status_bar_text)

        return {"PASS_THROUGH"}


    # --- Invoke Method (Starts the process) ---
    def invoke(self, context, event):
        # Poll checks handle preconditions
        wm = context.window_manager
        if getattr(wm, "comfyui_modal_operator_running", False): self.report({"WARNING"}, "Operation already running."); return {"CANCELLED"}

        # --- Initial Setup ---
        self._thread_status = "Initializing..."; self._thread_progress = ""; self.final_result = None
        prefs = context.preferences.addons[__name__].preferences
        self.server_address = prefs.comfyui_address.strip()
        if not self.server_address.startswith(("http://", "https://")):
            self.report({"ERROR"}, "ComfyUI server address invalid in preferences."); bpy.ops.preferences.open_comfyui_addon_prefs('INVOKE_DEFAULT'); return {"CANCELLED"}

        # --- Get Target GP Object and Settings ---
        self.target_gp_object_name = context.active_object.name
        scene_props = context.scene.comfyui_props

        # Store settings for the thread
        self.scene_user_prompt = scene_props.user_prompt
        self.scene_frame_rate = scene_props.frame_rate
        self.scene_depth_strength = scene_props.controlnet_depth_strength # Might be unused by some workflows
        self.scene_invert_depth = scene_props.invert_depth_input
        self.scene_reference_image_path = bpy.path.abspath(scene_props.reference_image_path) if scene_props.reference_image_path else ""
        self.scene_use_preceding = scene_props.use_preceding_frame
        self.scene_use_succeeding = scene_props.use_succeeding_frame
        self.selected_lora_name = scene_props.selected_lora
        self.selected_lora_strength_model = scene_props.lora_strength_model
        self.selected_lora_strength_clip = scene_props.lora_strength_clip

        # Store current state
        self.original_frame = context.scene.frame_current
        self.previous_obj = context.active_object
        self.previous_mode = self.previous_obj.mode if self.previous_obj else "OBJECT"

        # --- Determine Workflow Type and Frames ---
        self.frames_to_process = []
        self.preceding_frame_path = None
        self.succeeding_frame_path = None
        preceding_frame_num = None
        succeeding_frame_num = None

        if scene_props.frame_mode == "CURRENT":
            self.workflow_type = "SINGLE"
            self.frames_to_process = [self.original_frame]
        else: # RANGE
            start_f = scene_props.frame_start
            end_f = scene_props.frame_end
            if start_f > end_f: self.report({"ERROR"}, "Start frame > End frame."); return {"CANCELLED"}
            self.frames_to_process = list(range(start_f, end_f + 1))

            if self.scene_use_preceding or self.scene_use_succeeding:
                self.workflow_type = "MULTI_CONTEXT"
                if self.scene_use_preceding:
                    preceding_frame_num = start_f - 1
                    if preceding_frame_num < 0: # Or scene.frame_start?
                         self.report({"WARNING"}, f"Cannot get preceding frame for start frame {start_f}. Disabling.")
                         self.scene_use_preceding = False # Turn off if invalid
                         preceding_frame_num = None
                if self.scene_use_succeeding:
                     succeeding_frame_num = end_f + 1
                     # Check against scene.frame_end?
                     # if succeeding_frame_num > context.scene.frame_end:
                     #     self.report({"WARNING"}, f"Cannot get succeeding frame past scene end {context.scene.frame_end}. Disabling.")
                     #     self.scene_use_succeeding = False
                     #     succeeding_frame_num = None

            else: # Neither context frame selected
                 self.workflow_type = "MULTI_REF"
                 if not self.scene_reference_image_path or not os.path.exists(self.scene_reference_image_path):
                      self.report({"ERROR"}, f"Reference image required for this mode, but path is invalid/not set: '{self.scene_reference_image_path}'")
                      return {"CANCELLED"}

        if not self.frames_to_process: self.report({"WARNING"}, "No frames selected."); return {"CANCELLED"}
        print(f"Selected Workflow Type: {self.workflow_type}")
        print(f"Frames to process: {self.frames_to_process}")
        if preceding_frame_num is not None: print(f"Preceding frame: {preceding_frame_num}")
        if succeeding_frame_num is not None: print(f"Succeeding frame: {succeeding_frame_num}")

        if self.workflow_type == "SINGLE":
            if self.selected_lora_name in ["INVALID_PATH", "SCAN_ERROR"]:
                 self.report({"ERROR"}, f"Cannot run: Invalid LoRA setting ({self.selected_lora_name}). Check addon preferences or directory.")
                 return {"CANCELLED"}
             # Optional: Check if NONE is selected but the workflow absolutely requires a LoRA file name
             # For now, we assume disabling strength is enough.

        # --- Load Workflow JSON ---
        workflow_filename = WORKFLOW_FILES.get(self.workflow_type)
        if not workflow_filename: self.report({"ERROR"}, f"No workflow file defined for type: {self.workflow_type}"); return {"CANCELLED"}
        workflow_filepath = os.path.join(WORKFLOW_DIR, workflow_filename)
        if not os.path.exists(workflow_filepath): self.report({"ERROR"}, f"Workflow file not found: {workflow_filepath}"); return {"CANCELLED"}

        try:
            with open(workflow_filepath, 'r') as f:
                self.workflow = json.load(f)
            self.node_ids = WORKFLOW_NODE_IDS.get(self.workflow_type)
            if not self.node_ids: raise ValueError(f"Node ID mapping not defined for workflow type {self.workflow_type}")
            print(f"Loaded workflow from: {workflow_filepath}")
            # Basic validation (optional): Check if expected node IDs exist in the loaded workflow
            for key, node_id in self.node_ids.items():
                 if node_id not in self.workflow:
                     # Make missing output node non-fatal for now, handled in worker thread
                     if key not in ["OUTPUT_DECODE", "SAVE_IMAGE", "VIDEO_COMBINE", "NEGATIVE_PROMPT"]: # Allow optional/alternative nodes
                         raise ValueError(f"Workflow {workflow_filename} missing expected node ID for '{key}': {node_id}")

        except (json.JSONDecodeError, ValueError, KeyError, Exception) as e:
            self.report({"ERROR"}, f"Failed to load/validate workflow '{workflow_filepath}': {e}")
            import traceback; traceback.print_exc(); return {"CANCELLED"}

        # --- Create Temp Directory ---
        try:
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="blender_comfy_gp_")
            self.temp_dir_path = self.temp_dir_obj.name
            print(f"Using temporary directory: {self.temp_dir_path}")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to create temp directory: {e}"); self._cleanup_temp_files(); return {"CANCELLED"}

        # --- SYNCHRONOUS Frame Capture ---
        capture_frames = list(self.frames_to_process) # Copy
        if preceding_frame_num is not None: capture_frames.append(preceding_frame_num)
        if succeeding_frame_num is not None: capture_frames.append(succeeding_frame_num)
        capture_frames = sorted(list(set(capture_frames))) # Unique and sorted

        self._thread_status = "Capturing frames..."; self._thread_progress = ""
        wm.comfyui_modal_status = self._thread_status; wm.comfyui_modal_progress = self._thread_progress
        context.workspace.status_text_set(f"ComfyUI: {self._thread_status} (May take time)")
        self.report({"INFO"}, f"Capturing {len(capture_frames)} frame(s)...")
        wm.progress_begin(0, len(capture_frames))
        self.frame_paths = {} # Reset frame paths dict
        capture_success = True

        # Switch to Object mode for capture stability
        mode_switched_for_capture = False
        if self.previous_mode != "OBJECT" and self.previous_obj:
            try: context.view_layer.objects.active = self.previous_obj; bpy.ops.object.mode_set(mode="OBJECT"); mode_switched_for_capture = True; print("Switched to Object mode for capture.")
            except Exception as e: self.report({"WARNING"}, f"Could not switch to Object mode for capture: {e}")

        capture_start_time = time.time()
        for i, frame_num in enumerate(capture_frames):
            frame_start_time = time.time()
            context.scene.frame_set(frame_num)
            self._thread_progress = f"{i+1}/{len(capture_frames)} (Frame {frame_num})"
            wm.progress_update(i); context.view_layer.update()

            # Use the frame number in the filename (handles non-sequential captures)
            frame_filename = self.frame_pattern % frame_num
            output_path = os.path.join(self.temp_dir_path, frame_filename)
            try:
                print(f"Capturing frame {frame_num} to {output_path}...")
                capture_viewport(output_path, context)
                self.frame_paths[frame_num] = output_path # Store path by frame number
                print(f"  Frame {frame_num} captured in {time.time() - frame_start_time:.2f}s")

                # Store paths for context frames specifically
                if frame_num == preceding_frame_num: self.preceding_frame_path = output_path
                if frame_num == succeeding_frame_num: self.succeeding_frame_path = output_path

            except Exception as e:
                self.report({"ERROR"}, f"Failed to capture frame {frame_num}: {e}"); import traceback; traceback.print_exc()
                capture_success = False; break

        wm.progress_end()
        print(f"Total capture time: {time.time() - capture_start_time:.2f}s")

        # Restore original mode if switched
        if mode_switched_for_capture and self.previous_obj and self.previous_obj.name in context.view_layer.objects:
            try:
                print(f"Restoring mode to {self.previous_mode} after capture.")
                context.view_layer.objects.active = self.previous_obj
                valid_modes = {"OBJECT", "EDIT", "SCULPT", "PAINT", "WEIGHT", "EDIT_GPENCIL", "SCULPT_GPENCIL", "PAINT_GPENCIL", "WEIGHT_GPENCIL"}
                if self.previous_mode in valid_modes: bpy.ops.object.mode_set(mode=self.previous_mode)
                else: print(f"  Cannot restore invalid original mode '{self.previous_mode}', staying in Object."); bpy.ops.object.mode_set(mode='OBJECT')
            except Exception as e: 
                print(f"Could not restore mode after capture: {e}"); 
                try: bpy.ops.object.mode_set(mode='OBJECT') 
                except: pass

        # Check capture success
        if not capture_success or not self.frame_paths:
            self.report({"ERROR"}, "Frame capture failed. Aborting."); self._cleanup_temp_files()
            wm.comfyui_modal_operator_running = False; wm.comfyui_modal_status = "Idle"; wm.comfyui_modal_progress = ""; context.workspace.status_text_set(None)
            return {"CANCELLED"}
        # Verify main sequence frames were captured
        for f_num in self.frames_to_process:
            if f_num not in self.frame_paths:
                self.report({"ERROR"}, f"Missing captured frame for main sequence: {f_num}. Aborting."); self._cleanup_temp_files()
                wm.comfyui_modal_operator_running = False; wm.comfyui_modal_status = "Idle"; wm.comfyui_modal_progress = ""; context.workspace.status_text_set(None)
                return {"CANCELLED"}
        # Verify context frames if needed
        if self.workflow_type == "MULTI_CONTEXT":
             if self.scene_use_preceding and not self.preceding_frame_path:
                 self.report({"ERROR"}, "Preceding frame requested but failed to capture/find path. Aborting."); self._cleanup_temp_files(); wm.comfyui_modal_operator_running = False; wm.comfyui_modal_status = "Idle"; wm.comfyui_modal_progress = ""; context.workspace.status_text_set(None); return {"CANCELLED"}
             if self.scene_use_succeeding and not self.succeeding_frame_path:
                 self.report({"ERROR"}, "Succeeding frame requested but failed to capture/find path. Aborting."); self._cleanup_temp_files(); wm.comfyui_modal_operator_running = False; wm.comfyui_modal_status = "Idle"; wm.comfyui_modal_progress = ""; context.workspace.status_text_set(None); return {"CANCELLED"}

        # --- REMOVED DYNAMIC LATENT DIMENSIONS SETTING ---
        print("Skipping dynamic latent dimension setting (using workflow defaults).")

        # --- Start Background Thread ---
        self._result_queue = queue.Queue()
        self.client_id = str(uuid.uuid4())
        self._thread = threading.Thread(target=self._comfyui_worker_thread, name="ComfyUI_Worker_GP")
        self._thread.daemon = True # Allow Blender to exit if thread hangs

        self._thread_status = "Starting ComfyUI request..."; self._thread_progress = ""
        wm.comfyui_modal_operator_running = True
        wm.comfyui_modal_status = self._thread_status; wm.comfyui_modal_progress = self._thread_progress
        context.workspace.status_text_set(f"ComfyUI: {self._thread_status}")
        self._thread.start()
        print("Worker thread started.")

        # --- Register Modal Handler ---
        self._timer = context.window_manager.modal_handler_add(self)
        print("Modal handler added.")
        return {"RUNNING_MODAL"}


    # --- Cleanup and State Restoration ---
    def finish_or_cancel(self, context, cancelled=False):
        """ Cleans up resources and restores Blender state. """
        print(f"Finishing or cancelling GP operation (Cancelled: {cancelled})")
        wm = context.window_manager

        # Timer is removed automatically by returning FINISHED/CANCELLED
        self._timer = None

        # Clear status bar and WM state
        context.workspace.status_text_set(None)
        wm.comfyui_modal_operator_running = False
        if cancelled and wm.comfyui_modal_status != "Finished with Error":
            wm.comfyui_modal_status = "Cancelled"; wm.comfyui_modal_progress = ""

        # Ensure thread is finished
        if self._thread and self._thread.is_alive():
            print("Warning: Worker thread still alive. Attempting join...")
            self._thread.join(timeout=5.0)
            if self._thread.is_alive(): print("ERROR: Worker thread did not terminate!")
        self._thread = None

        # Cleanup temporary files
        self._cleanup_temp_files()

        # Restore Blender state (frame, selection, mode)
        try:
            # Restore frame
            if hasattr(self, "original_frame"):
                if context.scene and context.scene.frame_start <= self.original_frame <= context.scene.frame_end:
                     if context.scene.frame_current != self.original_frame:
                          print(f"Restoring scene frame to {self.original_frame}")
                          context.scene.frame_set(self.original_frame)
                else: print(f"Skipping frame restoration: Original frame {self.original_frame} invalid.")

            # Restore selection/mode to original object
            original_gp_obj = bpy.data.objects.get(getattr(self, "target_gp_object_name", None))
            prev_mode = getattr(self, "previous_mode", None)
            if original_gp_obj and prev_mode:
                 print(f"Restoring selection/mode to: {original_gp_obj.name} / {prev_mode}")
                 try:
                     current_mode = context.mode
                     if current_mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
                     bpy.ops.object.select_all(action='DESELECT')
                     context.view_layer.objects.active = original_gp_obj
                     original_gp_obj.select_set(True)
                     if prev_mode != context.active_object.mode:
                           valid_modes = {"OBJECT", "EDIT", "SCULPT", "PAINT", "WEIGHT", "EDIT_GPENCIL", "SCULPT_GPENCIL", "PAINT_GPENCIL", "WEIGHT_GPENCIL"}
                           if prev_mode in valid_modes: bpy.ops.object.mode_set(mode=prev_mode); print(f"  Restored original mode: {prev_mode}")
                           else: print(f"  Cannot restore invalid original mode '{prev_mode}'.")
                 except Exception as restore_e: print(f"  Warning: Could not fully restore original selection/mode: {restore_e}")
            else: print(f"Skipping original selection/mode restoration.")

        except Exception as e: self.report({"WARNING"}, f"Error during state restoration: {e}")

        self.report({"INFO"}, f"ComfyUI GP operation {'cancelled' if cancelled else 'finished'}.")
        print("-" * 30)
        return {"CANCELLED"} if cancelled else {"FINISHED"}


    # --- Cleanup Helper (Unchanged) ---
    def _cleanup_temp_files(self):
        """ Helper to remove temporary files and directory using shutil. """
        if hasattr(self, "temp_dir_obj") and self.temp_dir_obj:
            print(f"Cleaning up temporary directory object: {self.temp_dir_obj.name}...")
            try: self.temp_dir_obj.cleanup(); print("  Temp dir object cleaned.")
            except Exception as e:
                print(f"  Warning: Temp dir object cleanup failed: {e}. Attempting manual removal.")
                if self.temp_dir_path and os.path.exists(self.temp_dir_path):
                    try: shutil.rmtree(self.temp_dir_path, ignore_errors=True); print(f"  Manual removal of {self.temp_dir_path} attempted.")
                    except Exception as manual_e: print(f"  ERROR: Manual removal of {self.temp_dir_path} failed: {manual_e}"); self.report({"WARNING"}, f"Manual cleanup failed for {self.temp_dir_path}.")
            finally: self.temp_dir_obj = None; self.temp_dir_path = None
        elif hasattr(self, "temp_dir_path") and self.temp_dir_path and os.path.exists(self.temp_dir_path):
            print(f"Attempting manual cleanup of directory: {self.temp_dir_path}...")
            try: shutil.rmtree(self.temp_dir_path, ignore_errors=True); print(f"  Manual removal attempted.")
            except Exception as manual_e: print(f"  ERROR: Manual removal failed: {manual_e}"); self.report({"WARNING"}, f"Manual cleanup failed for {self.temp_dir_path}.")
            finally: self.temp_dir_path = None
        # Clear lists and paths
        self.frame_paths = {}; self.output_img_paths = []; self.temp_video_path = None
        self.preceding_frame_path = None; self.succeeding_frame_path = None


# -------------------------------------------------------------------
# HELPER OPERATOR: Open Addon Preferences (Unchanged)
# -------------------------------------------------------------------
class PREFERENCES_OT_open_comfyui_addon_prefs(bpy.types.Operator):
    """Opens the preferences specific to this addon"""
    bl_idname = "preferences.open_comfyui_addon_prefs"
    bl_label = "Open ComfyUI Addon Preferences"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context): return __name__ in context.preferences.addons
    def execute(self, context):
        try:
            bpy.ops.screen.userpref_show("INVOKE_DEFAULT")
            bpy.context.preferences.active_section = "ADDONS"
            # Try to find prefs window/area and filter
            prefs_window = next((w for w in context.window_manager.windows if w.screen.name == "User Preferences"), None)
            if prefs_window:
                 prefs_area = next((a for a in prefs_window.screen.areas if a.type == 'PREFERENCES'), None)
                 if prefs_area: prefs_area.spaces.active.filter_text = bl_info["name"]
            return {"FINISHED"}
        except Exception as e:
            print(f"Error opening preferences for '{__name__}': {e}"); self.report({"ERROR"}, f"Could not open preferences automatically."); return {"CANCELLED"}

# -------------------------------------------------------------------
# PANEL: UI in the 3D View sidebar
# -------------------------------------------------------------------
class VIEW3D_PT_comfyui_panel(bpy.types.Panel):
    bl_label = "ComfyUI GP Gen v3" # Updated Label
    bl_idname = "VIEW3D_PT_comfyui_gp_panel_v3" # Unique ID
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ComfyUI" # Tab name

    @classmethod
    def poll(cls, context): return SPA_GP_AVAILABLE and hasattr(spa_gp_core, 'import_image_as_gp_reference')
    def draw_header(self, context): self.layout.label(text="", icon='GREASEPENCIL')

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        scene_props = context.scene.comfyui_props
        is_running = getattr(wm, "comfyui_modal_operator_running", False)

        # --- Main Operator Button ---
        row = layout.row()
        # Poll method handles active GP object check
        op = row.operator(OBJECT_OT_run_comfyui_modal.bl_idname, icon="PLAY", text="Run Workflow") # Simpler Text
        row.enabled = not is_running

        # --- Status Display ---
        if is_running:
            status = getattr(wm, "comfyui_modal_status", "Running...")
            progress = getattr(wm, "comfyui_modal_progress", "")
            box = layout.box()
            row = box.row(align=True)
            row.label(text=f"Status:", icon='INFO')
            row.label(text=status)
            if progress: row.label(text=f"({progress})")
            row = box.row()
            cancel_op = row.operator("wm.comfyui_cancel_modal", icon='X', text="Cancel")

        # --- Operator Settings ---
        col = layout.column(align=True)
        col.enabled = not is_running

        col.prop(scene_props, "user_prompt")
        col.separator()
        col.prop(scene_props, "frame_mode")

        # --- Frame Range Specific Options ---
        if scene_props.frame_mode == "RANGE":
            box = col.box()
            row = box.row(align=True)
            row.prop(scene_props, "frame_start", text="Start")
            row.prop(scene_props, "frame_end", text="End")
            scene = context.scene
            box.label( text=f"(Scene: {scene.frame_start}-{scene.frame_end})", icon="TIME")
            box.prop(scene_props, "frame_rate")

            # --- Context Frame Options ---
            box_context = box.box()
            box_context.label(text="Context Frames (Multiframe Context Workflow):")
            row_context = box_context.row(align=True)
            row_context.prop(scene_props, "use_preceding_frame", text="Use Preceding", toggle=True, icon='TRIA_LEFT')
            row_context.prop(scene_props, "use_succeeding_frame", text="Use Succeeding", toggle=True, icon='TRIA_RIGHT')

            # --- Reference Image Option (Show only if context frames NOT used) ---
            if not scene_props.use_preceding_frame and not scene_props.use_succeeding_frame:
                 box_ref = box.box()
                 box_ref.label(text="Reference Image (Multiframe Reference Workflow):")
                 box_ref.prop(scene_props, "reference_image_path", text="") # No label needed, path is clear
                 # Add warning if path invalid?
                 ref_path = bpy.path.abspath(scene_props.reference_image_path) if scene_props.reference_image_path else ""
                 if scene_props.reference_image_path and not os.path.exists(ref_path):
                     box_ref.label(text="File not found!", icon='ERROR')


        # --- Moved Frame Rate for Current Frame ---
        elif scene_props.frame_mode == "CURRENT":
            # Frame rate isn't really used for single frame, but keep UI consistent?
            # Or hide it? Let's keep it but maybe disable or label it.
            row = col.row()
            row.prop(scene_props, "frame_rate")
            row.enabled = False # Disable for single frame as video not created
            # Could add label: layout.label(text="(Frame rate used for video modes)")

            box_lora = col.box()
            box_lora.label(text="LoRA Settings (Single Frame):")
            box_lora.prop(scene_props, "selected_lora")
            # Show strength only if a LoRA is selected
            if scene_props.selected_lora != "NONE" and scene_props.selected_lora != "INVALID_PATH" and scene_props.selected_lora != "SCAN_ERROR":
                row_lora_str = box_lora.row(align=True)
                row_lora_str.prop(scene_props, "lora_strength_model", text="Model Str.")
                row_lora_str.prop(scene_props, "lora_strength_clip", text="CLIP Str.")

        col.separator()

        # --- Common Settings (like Invert Depth) ---
        box_common = col.box()
        box_common.label(text="Common Settings:")
        box_common.prop(scene_props, "invert_depth_input", text="Invert Input for Depth/Control") # Clarify purpose
        # CN Strength only applies to single frame workflow now
        if scene_props.frame_mode == "CURRENT":
             box_common.prop(scene_props, "controlnet_depth_strength", text="ControlNet Strength")
        else:
             # Optionally show it disabled or hide it for multi-frame
             row = box_common.row()
             row.prop(scene_props, "controlnet_depth_strength", text="ControlNet Strength")
             row.enabled = False # Disable for non-single frame modes
             # box_common.label(text="(CN Strength only for Single Frame workflow)")

        layout.separator()
        # --- Preferences Button ---
        layout.operator(PREFERENCES_OT_open_comfyui_addon_prefs.bl_idname, text="Settings", icon="PREFERENCES")


def get_lora_items(self, context):
    """ Dynamically generates enum items for available LoRA files. """
    items = [("NONE", "None", "Do not use any LoRA")]
    prefs = context.preferences.addons[__name__].preferences
    lora_dir = prefs.lora_directory

    if not lora_dir or not os.path.isdir(lora_dir):
        items.append(("INVALID_PATH", "LoRA Path Invalid!", "Set LoRA directory in addon preferences"))
        return items

    try:
        valid_extensions = (".pt", ".safetensors")
        count = 0
        for filename in sorted(os.listdir(lora_dir)):
            if filename.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(lora_dir, filename)):
                # Identifier, Name, Description
                items.append((filename, filename, f"Use LoRA: {filename}"))
                count += 1
        # print(f"Found {count} LoRA files in {lora_dir}")

    except Exception as e:
        print(f"Error scanning LoRA directory '{lora_dir}': {e}")
        items.append(("SCAN_ERROR", "Error Scanning Dir", f"Check console/permissions for {lora_dir}"))

    return items

# -------------------------------------------------------------------
# Scene Properties
# -------------------------------------------------------------------
class ComfyUISceneProperties(bpy.types.PropertyGroup):
    user_prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Positive prompt for the ComfyUI workflow",
        default="cinematic, masterpiece, best quality, 1girl, detailed",
    )
    frame_mode: bpy.props.EnumProperty(
        items=[
            ("CURRENT", "Current Frame", "Use singleframe workflow"),
            ("RANGE", "Frame Range", "Use multiframe workflow (Reference or Context)"),
        ],
        default="CURRENT",
        name="Frame Mode",
        description="Choose workflow type based on frame selection",
    )
    selected_lora: bpy.props.EnumProperty(
        items=get_lora_items,
        name="LoRA Model",
        description="Select LoRA to use (Only for Current Frame mode)",
        default=0,
    )
    frame_start: bpy.props.IntProperty(
        name="Start Frame", description="Starting frame number for range processing",
        default=1, min=0,
    )
    frame_end: bpy.props.IntProperty(
        name="End Frame", description="Ending frame number for range processing",
        default=10, min=0,
    )
    frame_rate: bpy.props.IntProperty(
        name="Video Frame Rate", description="Frame rate for temporary video sent to ComfyUI (multi-frame modes)",
        default=8, min=1, max=120,
    )
    # --- New properties for multi-frame options ---
    use_preceding_frame: bpy.props.BoolProperty(
        name="Use Preceding Frame", description="Include frame before Start Frame in context (uses multiframe_depth+context workflow)",
        default=False,
    )
    use_succeeding_frame: bpy.props.BoolProperty(
        name="Use Succeeding Frame", description="Include frame after End Frame in context (uses multiframe_depth+context workflow)",
        default=False,
    )
    reference_image_path: bpy.props.StringProperty(
        name="Reference Image", description="Path to the reference image (used only if Frame Range selected and Context Frames are OFF - uses multiframe_depth+reference workflow)",
        default="", subtype='FILE_PATH',
    )
    # --- Common settings ---
    controlnet_depth_strength: bpy.props.FloatProperty(
        name="ControlNet Strength", description="Strength of the ControlNet effect (primarily for Single Frame workflow)",
        default=0.5, min=0.0, max=1.0, # Adjusted default
    )
    invert_depth_input: bpy.props.BoolProperty(
        name="Invert Input", description="Invert the input image/video before processing (affects depth/control input)",
        default=False, # Defaulting to False might be safer for depth
    )
    lora_strength_model: bpy.props.FloatProperty(
        name="LoRA Strength (Model)",
        description="Strength applied to the model by the selected LoRA",
        default=1.0, min=0.0, max=2.0,
    )
    lora_strength_clip: bpy.props.FloatProperty(
        name="LoRA Strength (CLIP)",
        description="Strength applied to CLIP by the selected LoRA",
        default=1.0, min=0.0, max=2.0,
    )

# -------------------------------------------------------------------
# Simple Operator to Cancel Modal (Unchanged)
# -------------------------------------------------------------------
class WM_OT_ComfyUICancelModal(bpy.types.Operator):
    """ Signals cancellation to the running modal operator """
    bl_idname = "wm.comfyui_cancel_modal"
    bl_label = "Cancel ComfyUI Operation"
    bl_description = "Stops the background ComfyUI process"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context): return getattr(context.window_manager, "comfyui_modal_operator_running", False)
    def execute(self, context):
        wm = context.window_manager
        if getattr(wm, "comfyui_modal_operator_running", False):
            print("Cancel button pressed. Signaling modal operator.")
            wm.comfyui_modal_operator_running = False # Set flag
            self.report({'INFO'}, "ComfyUI cancellation requested.")
        else: self.report({'WARNING'}, "ComfyUI operation not running.")
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
    WM_OT_ComfyUICancelModal,
)

def register():
    print("-" * 30)
    print(f"Registering {bl_info['name']} Add-on...")

    # Check essential dependencies first
    if not websocket:
        raise ImportError(f"Addon '{bl_info['name']}' requires 'websocket-client'. Install it.")
    if not SPA_GP_AVAILABLE:
         raise ImportError(f"Addon '{bl_info['name']}' requires SPA Studios GP features. Enable SPA addon.")
    # Check workflow dir/files
    if not os.path.isdir(WORKFLOW_DIR):
         print(f"  WARNING: Workflow directory NOT FOUND: {WORKFLOW_DIR}")
    else:
         missing = [f for f in WORKFLOW_FILES.values() if not os.path.exists(os.path.join(WORKFLOW_DIR, f))]
         if missing: print(f"  WARNING: Missing workflow files: {', '.join(missing)} in {WORKFLOW_DIR}")
         else: print(f"  Workflow files found in: {WORKFLOW_DIR}")

    # Check for ffmpeg (basic check) - Keep as warning
    try:
        startupinfo = None
        if os.name == "nt": startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startupinfo.wShowWindow = subprocess.SW_HIDE
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, timeout=5, startupinfo=startupinfo)
        print("  ffmpeg found.")
    except Exception: print("  Warning: ffmpeg check failed. Ensure it's installed and in PATH.")

    # Register classes
    for cls in classes:
        try: bpy.utils.register_class(cls)
        except Exception as e:
            print(f"  ERROR: Failed to register class {cls.__name__}: {e}"); unregister_on_error(); raise

    # Add Scene Properties
    bpy.types.Scene.comfyui_props = bpy.props.PointerProperty(type=ComfyUISceneProperties)

    # Add WindowManager properties for modal state
    bpy.types.WindowManager.comfyui_modal_operator_running = bpy.props.BoolProperty(default=False)
    bpy.types.WindowManager.comfyui_modal_status = bpy.props.StringProperty(default="Idle")
    bpy.types.WindowManager.comfyui_modal_progress = bpy.props.StringProperty(default="")

    print(f"{bl_info['name']} Add-on registered successfully.")
    print("-" * 30)

# Separate unregister function for error handling (Unchanged)
def unregister_on_error():
    """ Attempts to unregister classes during error recovery. """
    print("Attempting to unregister partially registered classes...")
    try: del bpy.types.WindowManager.comfyui_modal_operator_running
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_status
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_progress
    except AttributeError: pass
    try:
        if hasattr(bpy.types.Scene, "comfyui_props"): del bpy.types.Scene.comfyui_props
    except Exception as e: print(f"  Warn: Error deleting Scene.comfyui_props: {e}")
    for cls in reversed(classes):
        if hasattr(cls, "bl_rna"):
             try: bpy.utils.unregister_class(cls)
             except Exception as e: print(f"  Warn: Failed unregistering {cls.__name__}: {e}")

def unregister():
    print("-" * 30)
    print(f"Unregistering {bl_info['name']} Add-on...")

    # Delete WindowManager properties
    try: del bpy.types.WindowManager.comfyui_modal_operator_running
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_status
    except AttributeError: pass
    try: del bpy.types.WindowManager.comfyui_modal_progress
    except AttributeError: pass

    # Delete Scene properties
    try:
        if hasattr(bpy.types.Scene, "comfyui_props"): del bpy.types.Scene.comfyui_props
    except Exception as e: print(f"  Warn: Error deleting Scene.comfyui_props: {e}")

    # Unregister classes
    for cls in reversed(classes):
        if hasattr(cls, "bl_rna"):
            try: bpy.utils.unregister_class(cls)
            except Exception as e: print(f"  Warn: Failed unregistering {cls.__name__}: {e}")

    print(f"{bl_info['name']} Add-on unregistered.")
    print("-" * 30)

if __name__ == "__main__":
    # Allow running the script directly in Blender Text Editor
    try: unregister() # Clean up previous registration if any
    except Exception: pass
    register()