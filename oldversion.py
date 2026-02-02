#!/usr/bin/env python3

import subprocess
import threading
import queue
import time
import signal
import io
import os
import cv2
import pytesseract
from PIL import Image
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
CAMERA_H264_PROFILE = "baseline"

MJPEG_WIDTH = 320
MJPEG_HEIGHT = 240
MJPEG_FPS = 15

FRAME_BUFFER_SIZE = 20  # max number of MJPEG frames in the buffer

HTTP_PORT = 8000
MJPEG_BOUNDARY = b"--FRAME--"

OCR_FRAME_INTERVAL = 30 # process every Nth frame for OCR (e.g., 15 FPS / 30 = 0.5 FPS for OCR)


stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=FRAME_BUFFER_SIZE) # Thread-safe queue for MJPEG frames
raspivid_process = None
ffmpeg_process = None
latest_ocr_result = ""

def mjpeg_frame_reader(ffmpeg_stdout):
    jpeg_data = b''
    while not stop_event.is_set():
        try:
            chunk = ffmpeg_stdout.read(4096)
            if not chunk:
                print("MJPEG Reader: FFmpeg stdout closed.")
                break
            jpeg_data += chunk

            start_index = jpeg_data.find(b'\xff\xd8')
            end_index = jpeg_data.find(b'\xff\xd9')

            while start_index != -1 and end_index != -1 and start_index < end_index:
                frame = jpeg_data[start_index : end_index + 2] # +2 to include the EOI marker
                try:
                    frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    # if full, then drop the oldest one and put new one there
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(frame)
                        print("MJPEG Reader: Queue was full, dropping oldest rame.")
                    except queue.Empty:
                        pass

                # prepare the next one
                jpeg_data = jpeg_data[end_index + 2:]
                start_index = jpeg_data.find(b'\xff\xd8')
                end_index = jpeg_data.find(b'\xff\xd9')

        except Exception as e:
            if not stop_event.is_set(): # dont print error if user wants to stop
                print(f"MJPEG Reader: Error reading frame: {e}")
            break
    print("MJPEG Reader: Thread finished.")

class CustomHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html_content = f"""
            <html>
            <head>
                <title>Raspi Camera OCR Stream</title>
                <style>
                    body {{ margin: 0; padding: 0; font-family: sans-serif; }}
                    #container {{ width: {MJPEG_WIDTH}px; margin: 20px auto; padding: 10px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                    #videoFeed {{ display: block; margin: 0 auto; border: 1px solid #ccc; }}
                    #ocrText {{ margin-top: 10px; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap; word-wrap: break-word; min-height: 50px; max-height: 200px; overflow-y: auto; font-size: 14px; }}
                    h2, h3 {{ text-align: center; }}
                </style>
            </head>
            <body>
                <div id="container">
                    <h2>Raspberry Pi Camera Stream with OCR</h2>
                    <img id="videoFeed" src="/stream.mjpeg" width="{MJPEG_WIDTH}" height="{MJPEG_HEIGHT}" />
                    <h3>Recognized Text:</h3>
                    <div id="ocrText">Waiting for first OCR result...</div>

                    <script>
                        function fetchOcrText() {{
                            fetch('/ocr.txt')
                                .then(response => {{
                                    if (!response.ok) {{
                                        throw new Error(`HTTP error! status: ${{response.status}}`);
                                    }}
                                    return response.text();
                                }})
                                .then(text => {{
                                    document.getElementById('ocrText').innerText = text || 'No text detected yet.';
                                }})
                                .catch(error => {{
                                    console.error('Error fetching OCR text:', error);
                                    document.getElementById('ocrText').innerText = 'Error fetching text.';
                                }});
                        }}

                        // Fetch OCR text initially and then periodically
                        fetchOcrText();
                        setInterval(fetchOcrText, 1000); // Fetch every 1 second (adjust as needed)
                    </script>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html_content.encode('utf-8'))

        elif self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', f'multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY.decode()}')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            try:
                while not stop_event.is_set():
                    try:
                        frame = frame_queue.get(timeout=0.5) # ask for a frame
                        self.wfile.write(MJPEG_BOUNDARY + b'\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    except queue.Empty:
                        time.sleep(0.05) # just wait if theres nothing in queue
                    except BrokenPipeError:
                        print("Streamer: Client disconnected.") # this line just makes the code looks cool, i dont really need this
                        break
                    except Exception as e:
                        if not stop_event.is_set():
                            print(f"Streamer: Error sending frame: {e}")
                        break
            except Exception as e:
                if not stop_event.is_set():
                    print(f"Streamer: Main loop exception: {e}")
            finally:
                print("Streamer: Client connection closed or stream stopped.")

        elif self.path == '/ocr.txt':
            # send latest results of ocr as plain text for the JS to fetch
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            # Add Cache-Control headers to ensure browsers don't cache the text aggressively
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(latest_ocr_result.encode('utf-8'))

        else:
            self.send_error(404)
            self.end_headers()


def web_server_thread_func():
    try:
        print(f"Web Server: Starting HTTP server on port {HTTP_PORT}")
        httpd = ThreadingHTTPServer(('0.0.0.0', HTTP_PORT), CustomHTTPRequestHandler)
        httpd.serve_forever()

        while not stop_event.is_set():
            httpd.handle_request()

    except Exception as e:
        if not stop_event.is_set():
            print(f"Web Server failed. Erorr: {e}")
    finally:
        if 'httpd' in locals() and httpd:
            httpd.server_close()
        print("Web Server Thread finished.")

def ocr_thread_func():
    global latest_ocr_result
    frame_count = 0
    while not stop_event.is_set():
        try:
            # get one frame
            frame = frame_queue.get(timeout=0.5)
            frame_count += 1

            # do ocr
            if frame_count % OCR_FRAME_INTERVAL == 0:
                try:
                    pil_img = Image.open(io.BytesIO(frame)) # tesseract cant directly read jpeg frame, it can only read from a file or a file like object, i convert the frame (jpeg) to a file like object
                    img_np = np.array(pil_img)
                    img_processed = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                    # img_thresholded = cv2.threshold(img_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # this line is not needed, it just a preprocessing step, but the cpu usage is already too high so i removed it
                    text = pytesseract.image_to_string(img_processed) # this line is the line that actually get the wrok done, other lines are just preprocessing to improve the outcome
                    if text.strip():
                        print(f"OCR Result: {text.strip()}") # assign the result to the global variable if its not empty
                        latest_ocr_result = text.strip()
                    else:
                        print("OCR Result: No text detected.")

                except Exception as e:
                    print(f"OCR Thread failed. Error: {e}")

        except queue.Empty:
            time.sleep(0.05) # if queue empty then wait a bit, the previous versions dont actually encounter this issue, or at least i dont see error on this being reported
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"OCR Thread failed. Error: {e}")
            break
    print("OCR Thread finished.")


def main():
    global raspivid_process, ffmpeg_process
    def signal_handler(sig, frame):
        print("\nCtrl+C received. Shutting down ...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # -pf baseline : H.264 profile for better compatibility / lower complexity
        # -ih : Insert PPS/SPS headers inline
        raspivid_cmd = [
            'raspivid',
            '-o', '-',
            '-t', '0',
            '-w', str(CAMERA_WIDTH),
            '-h', str(CAMERA_HEIGHT),
            '-fps', str(CAMERA_FPS),
            '-pf', CAMERA_H264_PROFILE,
            '-ih',
        ]
        print(f"Starting raspivid subprocess")
        try:
            raspivid_process = subprocess.Popen(raspivid_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) # allow piping the standrad output, and ignore standard error
            print("raspivid subprocess started.")
        except Exception as e:
            print(f"raspivid subprocess failed. Error: {e}") # i think this will never happen, it just sits there
            return

        # -i pipe:0 : Input from stdin
        # -f h264 : Input format
        # -c:v h264_v4l2m2m : Attempt hardware accelerated H.264 decoding (Raspberry Pi specific)
        #   Alternative decoders: h264_mmal (older), or just libx264 (CPU decoding if HW fails)
        # -vf format=yuvj420p : Pixel format often needed for MJPEG
        # -s <width>x<height> : Output resolution
        # -r <fps> : Output frame rate
        # -c:v mjpeg : Output codec MJPEG
        # -q:v 3 : MJPEG quality (2-5 is often good, lower is better quality, larger size)
        # -f mjpeg : Output format MJPEG
        # pipe:1 : Output to stdout
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'error', # its raelly annoying
            '-i', 'pipe:0',       # pipe stdin of ffmpeg to stdout of raspivid
            '-f', 'h264',         # telling it the input format
            # '-c:v', 'h264_v4l2m2m', # Try V4L2 M2M hardware decoder for H.264
            '-vf', f'fps={MJPEG_FPS},scale={MJPEG_WIDTH}:{MJPEG_HEIGHT}',
            '-c:v', 'mjpeg',      # convert to new format
            '-q:v', '4',          # MJPEG quality (adjust as needed)
            '-f', 'mjpeg',        # telling to output in mjpeg
            'pipe:1'              # Output to stdout (Python script's input)
        ]
        print(f"Starting FFmpeg subprocess.")
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=raspivid_process.stdout,
            stdout=subprocess.PIPE,        # allow piping to this python script
            stderr=subprocess.PIPE         # as i said, it is annoying so I will pipe it somewhere else
        )
        print("FFmpeg subprocess started.")

        # allow raspivid's stdout to be closed when ffmpeg exits
        if raspivid_process and raspivid_process.stdout:
             raspivid_process.stdout.close() # FFmpeg is now the sole reader of this pipe

        # Thread to read FFmpeg's stderr for debugging
        def log_ffmpeg_errors():
            if ffmpeg_process and ffmpeg_process.stderr:
                for line in iter(ffmpeg_process.stderr.readline, b''):
                    if stop_event.is_set():
                        break
                    print(f"FFmpeg STDERR: {line.decode().strip()}")
                ffmpeg_process.stderr.close()

        ffmpeg_error_logger_thread = threading.Thread(target=log_ffmpeg_errors, daemon=True)
        ffmpeg_error_logger_thread.start()

        # start mjpeg reader thread
        mjpeg_reader = threading.Thread(target=mjpeg_frame_reader, args=(ffmpeg_process.stdout,))
        mjpeg_reader.daemon = True # Daemonize so it exits when main exits, though we join explicitly
        mjpeg_reader.start()
        print("MJPEG frame reader thread started.")

        # start web server thread
        web_server = threading.Thread(target=web_server_thread_func)
        web_server.daemon = True
        web_server.start()
        print("Web server thread started.")

        # start ocr thread
        ocr_processor = threading.Thread(target=ocr_thread_func)
        ocr_processor.daemon = True
        ocr_processor.start()
        print("OCR thread started.")

        while not stop_event.is_set():
            time.sleep(0.5) # check for stopping signal every 0.5 second

    except Exception as e:
        print(f"Main process failed. Error: {e}")
        stop_event.set() # tell everything to stop

    finally:
        print("\nShutting down...")
        stop_event.set() # at this point, it is either the users wants to stop or something seriously went wrong, so just stop everything

        # stop and join threads
        if 'mjpeg_reader' in locals() and mjpeg_reader.is_alive():
            print("Joining MJPEG reader thread...")
            mjpeg_reader.join(timeout=2)
            if mjpeg_reader.is_alive(): print("MJPEG reader thread did not terminate cleanly.") # i have no idea what to do if this scenario happens

        if 'web_server' in locals() and web_server.is_alive():
            print("Joining web server thread...")
            web_server.join(timeout=2)
            if web_server.is_alive(): print("Web server thread did not terminate cleanly.")

        if 'ocr_processor' in locals() and ocr_processor.is_alive():
            print("Joining OCR thread...")
            ocr_processor.join(timeout=2)
            if ocr_processor.is_alive(): print("OCR thread did not terminate cleanly.")

        # terminate FFmpeg process
        if ffmpeg_process:
            print("Terminating FFmpeg process...")
            if ffmpeg_process.poll() is None: # check if its still running
                ffmpeg_process.terminate() # send SIGTERM
                try:
                    ffmpeg_process.wait(timeout=5) # wait for termination
                except subprocess.TimeoutExpired:
                    print("FFmpeg did not terminate gracefully, killing...")
                    ffmpeg_process.kill() # i send SIGTERM before SIGKILL because someone somewhere on the internet said that killing is more forcefull than terminating, and it seems true
                    try:
                        ffmpeg_process.wait(timeout=2) # wait for kill
                    except subprocess.TimeoutExpired:
                        print("FFmpeg could not be killed.")
            if ffmpeg_process.stdout: ffmpeg_process.stdout.close()
            if ffmpeg_process.stderr: ffmpeg_process.stderr.close()
            print("Main: FFmpeg process cleanup done.")

        # terminate raspivid process
        if raspivid_process:
            print("Terminating raspivid process...")
            if raspivid_process.poll() is None:
                raspivid_process.terminate()
                try:
                    raspivid_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("raspivid did not terminate gracefully, killing...")
                    raspivid_process.kill()
                    try:
                        raspivid_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        print("raspivid could not be killed.")
            if raspivid_process.stdout: raspivid_process.stdout.close() # should already be closed by FFmpeg pipe logic
            print("Main: raspivid process cleanup done.")

        # clear queue, i dont think it raelly matters that much, but just thinking that the queue might still contain some data makes me feel uncomfortable
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        print("Frame queue cleared.")

        if 'ffmpeg_error_logger_thread' in locals() and ffmpeg_error_logger_thread.is_alive():
            ffmpeg_error_logger_thread.join(timeout=1)

        print("Shutdown complete. Resources should be released.") # if not then i will use the old way: reboot the pi

if __name__ == '__main__':
    main()
