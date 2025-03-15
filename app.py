#!/usr/bin/env python3
import argparse
import asyncio
import errno
import logging
import os
import platform
import shlex
import signal
import struct
import psutil
import sys

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager


import socketio

# -------------------------------------------------------------------------------
# Conditional Imports for Unix-like Systems
# -------------------------------------------------------------------------------
if platform.system() in ["Linux", "Darwin"]:
    import pty
    import termios
    import fcntl

# -------------------------------------------------------------------------------
# Conditional Imports for Windows using pywinpty (ConPTY)
# -------------------------------------------------------------------------------
if platform.system() == "Windows":
    try:
        from winpty import PTY  # low-level API (see package usage)
    except ImportError:
        PTY = None

__version__ = "0.1.0"

# ==============================================================================
# Terminal Classes
# ==============================================================================

class BaseTerminal:
    """Abstract base class for terminal handling."""
    def start(self):
        raise NotImplementedError

    def write_input(self, data):
        raise NotImplementedError

    def resize(self, rows, cols):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError

# ------------------------------------------------------------------------------
# Unix Terminal Implementation (Event-driven using asyncio loop.add_reader)
# ------------------------------------------------------------------------------
if platform.system() in ["Linux", "Darwin"]:
    class UnixTerminal(BaseTerminal):
        """Terminal handling for Unix-like systems with event-driven PTY I/O."""
        def __init__(self, cmd_list):
            self.cmd_list = cmd_list
            self.child_pid = None
            self.fd = None
            self.loop = None
            self.alive = True  # Mark the terminal as alive

        def set_winsize(self, fd, rows, cols, xpix=0, ypix=0):
            logging.debug("Setting window size to %dx%d", rows, cols)
            winsize = struct.pack("HHHH", rows, cols, xpix, ypix)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

        def start(self):
            command = self.cmd_list[0]
            cmd_args = self.cmd_list[1:]
            logging.info("Starting Unix terminal with command: %s %s", command, " ".join(cmd_args))
            try:
                self.child_pid, self.fd = pty.fork()
            except OSError as e:
                logging.error("Could not fork a new process: %s", e)
                return False

            if self.child_pid == 0:
                # Child process: execute the command.
                try:
                    os.execvp(command, [command] + cmd_args)
                except Exception as e:
                    logging.error("Failed to exec command: %s", e)
                    os._exit(1)
            else:
                # Parent process: configure the PTY.
                self.set_winsize(self.fd, 50, 100)
                flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
                fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                self.loop = asyncio.get_running_loop()
                self.loop.add_reader(self.fd, self._on_read_ready)
                logging.info("Unix terminal started with PID %d", self.child_pid)
            return True

        def _on_read_ready(self):
            """
            Called when the PTY file descriptor is ready for reading.
            Reads available data, updates the shared screen buffer, and emits the output.
            """
            try:
                data = os.read(self.fd, 4096)
                # If this terminal is no longer the active one, do nothing.
                if app.state.terminal is not self:
                    return
                if data:
                    decoded = data.decode(errors="ignore")
                    # Update the global screen buffer.
                    app.state.screen_buffer += decoded
                    if len(app.state.screen_buffer) > 5000:
                        app.state.screen_buffer = app.state.screen_buffer[-5000:]
                    # Only emit output if this is the current terminal.
                    if app.state.terminal is self:
                        asyncio.create_task(
                            sio.emit("pty-output", {"output": decoded}, namespace="/pty")
                        )
                else:
                    logging.info("PTY reached EOF, cleaning up Unix terminal.")
                    self.loop.remove_reader(self.fd)
                    self.terminate()
                    if app.state.terminal is self:
                        app.state.terminal = None
            except OSError as e:
                if e.errno == errno.EIO:
                    logging.info("PTY reached EOF (EIO), cleaning up Unix terminal.")
                    self.loop.remove_reader(self.fd)
                    self.terminate()
                    if app.state.terminal is self:
                        app.state.terminal = None
                else:
                    logging.error("Error in _on_read_ready: %s", e)
            except Exception as e:
                logging.error("Error in _on_read_ready: %s", e)

        def write_input(self, data):
            if self.fd:
                try:
                    os.write(self.fd, data.encode())
                    logging.debug("Wrote input to PTY: %s", data)
                except OSError as e:
                    logging.error("Error writing to PTY: %s", e)

        def resize(self, rows, cols):
            if self.fd:
                self.set_winsize(self.fd, rows, cols)

        def terminate(self):
            if self.child_pid:
                logging.info("Terminating Unix terminal with PID %d", self.child_pid)
                try:
                    os.kill(self.child_pid, signal.SIGTERM)
                except OSError as e:
                    logging.error("Error terminating process: %s", e)
                self.child_pid = None
            self.alive = False  # Mark the terminal as no longer alive

# ------------------------------------------------------------------------------
# Windows Terminal Implementation (Polling with async read loop)
# ------------------------------------------------------------------------------
if platform.system() == "Windows":
    class WindowsTerminal(BaseTerminal):
        """Terminal handling for Windows using pywinpty (ConPTY via winpty.PTY)
        with an async read loop offloading the blocking read to a background thread."""
        def __init__(self, cmd_list):
            self.cmd_list = cmd_list
            self.pty = None
            self.reading = False
            self.alive = True  # Mark the terminal as alive

        def start(self):
            if not PTY:
                logging.error("pywinpty is not installed. Please install it using 'pip install pywinpty'")
                return False

            command = self.cmd_list[0]
            cmd_args = self.cmd_list[1:]
            logging.info("Starting Windows terminal with command: %s %s", command, " ".join(cmd_args))
            try:
                self.pty = PTY(100, 50)
                full_command = " ".join([command] + cmd_args)
                self.pty.spawn(full_command)
                self.reading = True
                logging.info("Windows terminal spawned successfully")
            except Exception as e:
                logging.error("Failed to start Windows terminal: %s", e)
                return False

            # Start the async read loop
            asyncio.create_task(self.start_read_loop())
            return True

        async def start_read_loop(self):
            while self.reading:
                # Check if this instance is still the active terminal.
                if app.state.terminal is not self:
                    break  # Stop the read loop if a new terminal has taken over.
                try:
                    output = await asyncio.to_thread(self.pty.read)
                    if output:
                        # Append output to the shared screen buffer.
                        app.state.screen_buffer += output
                        if len(app.state.screen_buffer) > 5000:
                            app.state.screen_buffer = app.state.screen_buffer[-5000:]
                        # Emit output only if still the current terminal.
                        if app.state.terminal is self:
                            await sio.emit("pty-output", {"output": output}, namespace="/pty")
                    else:
                        await asyncio.sleep(0.01)
                except Exception as e:
                    if "Standard out reached EOF" in str(e):
                        logging.info("PTY reached EOF, terminating Windows read loop.")
                        self.reading = False
                        self.alive = False
                        self.terminate()
                        if app.state.terminal is self:
                            app.state.terminal = None
                        break
                    logging.error("Error reading from Windows PTY: %s", e)
                    await asyncio.sleep(0.05)

        def write_input(self, data):
            if self.pty:
                try:
                    self.pty.write(data)
                    logging.debug("Wrote input to PTY: %s", data)
                except Exception as e:
                    logging.error("Error writing to PTY: %s", e)

        def resize(self, rows, cols):
            if self.pty:
                try:
                    self.pty.set_size(cols, rows)
                    logging.debug("Resized terminal to %dx%d", rows, cols)
                except Exception as e:
                    logging.error("Error resizing terminal: %s", e)

        def terminate(self):
            if self.pty:
                logging.info("Terminating Windows terminal")
                self.reading = False
                self.alive = False
                try:
                    if hasattr(self.pty, "close"):
                        self.pty.close()
                except Exception as e:
                    logging.error("Error terminating terminal: %s", e)
                self.pty = None

# ==============================================================================
# Factory Function
# ==============================================================================
def create_terminal(cmd_list):
    system = platform.system()
    if system in ["Linux", "Darwin"]:
        return UnixTerminal(cmd_list)
    elif system == "Windows":
        return WindowsTerminal(cmd_list)
    else:
        raise NotImplementedError(f"Unsupported platform: {system}")

# ============================================================================== 
# FastAPI + SocketIO Setup
# ==============================================================================

sio = socketio.AsyncServer(async_mode="asgi")

# Use lifespan handler when creating the app (defined below)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Starting system stats broadcaster...")
    app.state.system_stats_task = asyncio.create_task(system_stats_broadcaster())
    yield
    # Shutdown
    logging.info("Shutting down system stats broadcaster...")
    app.state.system_stats_task.cancel()

app = FastAPI(lifespan=lifespan)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Shared app state
app.state.terminal = None
app.state.connected_sids = set()
app.state.cmd = None
app.state.screen_buffer = ""

# Wrap FastAPI with Socket.IO
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="/socket.io")

# ==============================================================================
# Helper Functions
# ==============================================================================

async def maybe_terminate_terminal():
    """Terminate the terminal process after all clients disconnect."""
    await asyncio.sleep(5)
    if not app.state.connected_sids and app.state.terminal:
        term_pid = getattr(app.state.terminal, "child_pid", "N/A")
        logging.info(f"No clients left. Terminating terminal with PID {term_pid}.")
        app.state.terminal.terminate()
        app.state.terminal = None
        app.state.screen_buffer = ""

async def system_stats_broadcaster():
    """Broadcast system metrics to connected clients."""
    while True:
        if app.state.connected_sids:
            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_usage = ram.used / (1024 ** 3)
            ram_total = ram.total / (1024 ** 3)
            ping = 0  # Optional: implement real ping if necessary

            stats = {
                "cpu": f"{cpu_percent}%",
                "ram": f"{ram_usage:.1f}GB / {ram_total:.1f}GB",
                "ping": f"{ping}ms",
                "session": hex(id(app.state.terminal)) if app.state.terminal else "None"
            }

            logging.debug(f"Emitting system stats: {stats}")
            await sio.emit("system-stats", stats, namespace="/pty")

        await asyncio.sleep(2)

# ==============================================================================
# FastAPI Routes
# ==============================================================================

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ==============================================================================
# Socket.IO Handlers (PTY)
# ==============================================================================

@sio.on("pty-input", namespace="/pty")
async def pty_input(sid, data):
    terminal = app.state.terminal
    if terminal:
        input_data = data.get("input", "")
        logging.debug(f"Received terminal input: {input_data}")
        terminal.write_input(input_data)

@sio.on("resize", namespace="/pty")
async def resize(sid, data):
    terminal = app.state.terminal
    if terminal:
        rows = data.get("rows", 50)
        cols = data.get("cols", 100)
        logging.info(f"Resizing terminal to {rows}x{cols}")
        terminal.resize(rows, cols)

@sio.event(namespace="/pty")
async def connect(sid, environ):
    logging.info(f"Client connected: {sid}")
    app.state.connected_sids.add(sid)

    # Send buffered output if terminal already exists
    if app.state.terminal and app.state.terminal.alive:
        await sio.emit("pty-output", {"output": app.state.screen_buffer}, room=sid, namespace="/pty")
        return

    # Start a new terminal session
    app.state.terminal = None  # Clear dead terminal if present
    terminal = create_terminal(app.state.cmd)
    success = terminal.start()

    if success:
        app.state.terminal = terminal
        cmd_str = " ".join(shlex.quote(c) for c in app.state.cmd)
        logging.info(f"Started new terminal with command: {cmd_str}")
    else:
        logging.error("Failed to start terminal session")

@sio.event(namespace="/pty")
async def disconnect(sid):
    logging.info(f"Client disconnected: {sid}")
    app.state.connected_sids.discard(sid)
    if not app.state.connected_sids and app.state.terminal:
        asyncio.create_task(maybe_terminate_terminal())

# ==============================================================================
# Main Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TermX A fully functional terminal in your browser (FastAPI + Socket.IO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--port", default=5000, type=int, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--command", default="bash" if platform.system() != "Windows" else "cmd.exe", help="Terminal command to run")
    parser.add_argument("--cmd-args", default="", help="Arguments for the terminal command")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    # Set the terminal command state
    app.state.cmd = [args.command] + shlex.split(args.cmd_args)

    # Logging setup
    log_format = "%(levelname)s (%(funcName)s:%(lineno)d) %(message)s"
    logging.basicConfig(format=log_format, stream=sys.stdout,
                        level=logging.DEBUG if args.debug else logging.INFO)

    logging.info(f"Server running at http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(asgi_app, host=args.host, port=args.port,
                log_level="debug" if args.debug else "info")

if __name__ == "__main__":
    main()