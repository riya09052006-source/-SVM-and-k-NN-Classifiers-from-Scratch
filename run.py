import os
import sys
import time
import subprocess
import webbrowser
import socketserver
from http.server import SimpleHTTPRequestHandler
import threading

def run_backend():
    backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'app.py')
    print(f"Launching Flask backend API ({backend_path})...")
    subprocess.Popen([sys.executable, backend_path])

def serve_frontend(port=8000):
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
    os.chdir(frontend_dir)
    
    class SilentHTTPHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # suppress access logs in terminal

    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("127.0.0.1", port), SilentHTTPHandler) as httpd:
            print(f"Frontend is serving at http://127.0.0.1:{port}")
            threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(f"http://127.0.0.1:{port}"))).start()
            print("Press Ctrl+C to terminate both servers.")
            httpd.serve_forever()
    except Exception as e:
        print(f"Error starting frontend server: {e}")

if __name__ == '__main__':
    print("=========================================================")
    print("Starting SVM and k-NN Custom ML Playground Server")
    print("=========================================================")
    run_backend()
    time.sleep(1.2)
    try:
        serve_frontend(port=8000)
    except KeyboardInterrupt:
        print("\nShutting down servers. Goodbye!")
        sys.exit(0)