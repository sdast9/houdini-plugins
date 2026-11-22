# =============================================================================
# Running PolyFEM: launch, follow, stop, open
#
# Both run buttons leave files in output/ that say how to follow the run:
#   .polyfem_run.json     who started what, when (kind, launcher pid, log)
#   run-manifest.json     written by PolyFEM itself: its pid and, at the
#                         end, the completion status and exit status
#   .polyfem_exit_status  the terminal script's record of the exit status
# A poller (an event-loop callback in the interface) reads them about once a
# second and writes Run Status. PolyFEM's exit statuses: 0 completed,
# 1 named failure (the log's "PolyFEM stopped:" line says why), 3 resource
# limit (a safety stop; the steps written are intact), a signal = a crash.
# =============================================================================

_RUN_RECORD = ".polyfem_run.json"
_EXIT_STATUS_FILE = ".polyfem_exit_status"
_LAUNCHER_PID_FILE = ".polyfem_launcher_pid"
_TERMINAL_LOG = "terminal_log.txt"
_CONSOLE_LOG = "console_log.txt"
_POLL_SECONDS = 1.0
_STEP_RE = re.compile(r"\b(\d+)/(\d+)\s+t=([-+0-9.eE]+)")
_STOPPED_RE = re.compile(r"PolyFEM stopped: (.*)")


def _launch_command(parent, params_path):
    polyfem_bin = _resolved_binary(parent.evalParm("polyfem_bin").strip())
    if not polyfem_bin or not os.path.isfile(polyfem_bin):
        raise hou.NodeError("No valid PolyFEM binary provided!")
    params_path = os.path.abspath(params_path)
    input_dir = os.path.dirname(params_path)
    log_level = parent.evalParm("log_level")
    max_threads = parent.evalParm("max_threads")
    args = [polyfem_bin, "-j", params_path, "-o", "../output/",
            "--log_level", str(log_level)]
    if max_threads > 0:
        args += ["--max_threads", str(max_threads)]
    return input_dir, args


def _output_dir(parent):
    working_dir = parent.evalParm("working_dir")
    if not working_dir.strip():
        return None
    return os.path.normpath(os.path.abspath(
        os.path.join(working_dir, "output")))


def _runs():
    """This session's runs, {output folder: run} (kept in hou.session so a
    reloaded asset definition still sees them)."""
    return hou.session.__dict__.setdefault("polyfem_run_registry", {})


def _read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def _posix_process(pid):
    """Start time (one-second precision) and command from the POSIX process table."""
    if os.name != "posix":
        return None
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "lstart=,command="],
            env=dict(os.environ, LC_ALL="C"), capture_output=True, text=True,
            timeout=2, check=True)
        day_name, month, day, clock, year, command = result.stdout.split(maxsplit=5)
        months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
        start = time.mktime((int(year), months.index(month) + 1, int(day),
                             *map(int, clock.split(":")), 0, 0, -1))
        return start, command.strip()
    except (OSError, ValueError, OverflowError, subprocess.SubprocessError):
        return None


def _posix_run_alive(pid, expect, binary, created, earliest, latest, launcher):
    process = _posix_process(pid)
    if process is None or (created is None and earliest is None and latest is None):
        return False
    start, command = process
    if ((created is not None and abs(start - created) >= 1.0)
            or (earliest is not None and start + 1.0 < earliest)
            or (latest is not None and start > latest)):
        return False
    if launcher:
        # ps prints argv as a string. Match the complete script path after
        # the interpreter, including any spaces in the working directory.
        return command == launcher or command.partition(" ")[2] == launcher
    if not binary or not expect or not os.path.isabs(expect):
        return False
    return (command.startswith(binary + " ") or
            command.startswith(os.path.realpath(binary) + " ")) and bool(
        re.search(r"(?:^|\s)(?:-j|--json)\s+" + re.escape(expect) +
                  r"(?=\s|$)", command))


def _process_alive(pid, expect=None, *, cwd=None, binary=None, created=None,
                   earliest=None, latest=None, launcher=None, for_signal=False):
    """Verify a run process against its input path and recorded start."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        return not for_signal and _posix_run_alive(
            pid, expect, binary, created, earliest, latest, launcher)
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        actual_cwd = process.cwd() if cwd is not None else None
        if cwd is not None and _normalized_path(actual_cwd) != _normalized_path(cwd):
            return False
        if binary and _normalized_path(process.exe()) != _normalized_path(binary):
            return False
        start = process.create_time()
        if ((created is not None and abs(start - created) > 0.01)
                or (earliest is not None and start < earliest)
                or (latest is not None and start > latest)):
            return False
        command = process.cmdline()
        if launcher is not None:
            return any(_normalized_path(os.path.join(actual_cwd, arg))
                       == _normalized_path(launcher) for arg in command[:2])
        if cwd is not None:
            params = next((arg for option, arg in zip(command, command[1:])
                           if option in ("-j", "--json")), None)
            return params is not None and _normalized_path(
                os.path.join(actual_cwd, params)) == _normalized_path(expect)
        return expect is None or expect in command
    except psutil.AccessDenied:
        return not for_signal and _posix_run_alive(
            pid, expect, binary, created, earliest, latest, launcher)
    except (psutil.Error, OSError, ValueError, TypeError):
        return False


def _normalized_path(path):
    return os.path.normcase(os.path.realpath(path))


def _process_created(pid):
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return psutil.Process(pid).create_time()
        except psutil.AccessDenied:
            pass
        except psutil.Error:
            return None
    return None  # the recorded launch window is used for POSIX fallback


def _run_pid_matches(run, pid, *, launcher=False, for_signal=False):
    if not pid:
        return False
    output = run["output"]
    input_dir = os.path.join(os.path.dirname(output), "input")
    started = run.get("started")
    if started is not None and (type(started) not in (int, float) or not math.isfinite(started)):
        return False
    earliest = started - 2 if started is not None else None
    if launcher:
        try:
            recorded_at = os.path.getmtime(os.path.join(output, _LAUNCHER_PID_FILE))
        except OSError:
            return False
        script = os.path.join(output, "run_polyfem.command"
                              if platform.system() == "Darwin" else "run_polyfem.sh")
        return _process_alive(pid, cwd=input_dir, launcher=script,
                              earliest=earliest, latest=recorded_at + 1,
                              for_signal=for_signal)

    manifest = _manifest_of_run(run) if started is not None else _manifest(output)
    process = manifest.get("process")
    executable = process.get("executable") if isinstance(process, dict) else None
    manifest_binary = executable.get("path") if isinstance(executable, dict) else None
    binary = run.get("binary") or manifest_binary
    if not isinstance(binary, str) or not binary:
        return False
    latest = None
    stamp = manifest.get("started_at")
    if isinstance(stamp, str):
        from datetime import datetime

        try:
            latest = datetime.fromisoformat(stamp).timestamp() + 1
        except ValueError:
            pass
    if not manifest and started is not None:
        latest = started + 2  # the HDA records a Popen immediately after launch
    return _process_alive(pid, os.path.join(input_dir, "params.json"),
                          cwd=input_dir, binary=binary,
                          created=run.get("process_created"),
                          earliest=earliest, latest=latest,
                          for_signal=for_signal)


def _manifest(output):
    return _read_json(os.path.join(output, "run-manifest.json")) or {}


def _manifest_of_run(run):
    """PolyFEM's run manifest when it belongs to this run (written after it
    was started), else {}."""
    path = os.path.join(run["output"], "run-manifest.json")
    try:
        if os.path.getmtime(path) < run["started"] - 0.5:
            return {}
    except OSError:
        return {}
    return _manifest(run["output"])


def _solver_pid(run):
    if run.get("popen") is not None:
        return run["popen"].pid
    if run.get("pid"):
        return run["pid"]
    process = _manifest_of_run(run).get("process", {})
    pid = process.get("pid") if isinstance(process, dict) else None
    return pid if isinstance(pid, int) else None


def _log_tail(path, size=262144):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - size))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _progress(text):
    """(step, steps, time) of the last 'k/N  t=...' line PolyFEM logged."""
    matches = _STEP_RE.findall(text)
    if not matches:
        return None
    step, steps, t = matches[-1]
    return int(step), int(steps), t


def _stopped_reason(text, manifest=None):
    matches = _STOPPED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    completion = (manifest or {}).get("completion", {})
    message = completion.get("message") if isinstance(completion, dict) \
        else None
    return message.strip() if isinstance(message, str) else ""


def _elapsed(seconds):
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def exit_status_meaning(code, signal_number=None):
    """Plain-language meaning of PolyFEM's exit status."""
    if signal_number:
        import signal
        try:
            name = signal.Signals(signal_number).name
        except ValueError:
            name = f"signal {signal_number}"
        return (f"killed by {name} (signal {signal_number}), i.e. a solver "
                "crash or an outside kill; the log shows how far it got")
    if code == 0:
        return "completed"
    if code == 1:
        return ("a named failure, not a crash: PolyFEM stopped and said why "
                "(the 'PolyFEM stopped:' line in the log)")
    if code == 3:
        return ("a resource limit, not a crash: a safety stop before memory "
                "ran out; the steps already written are intact")
    return f"exit status {code}"


def _finished(run):
    """(exit code, signal number) once the run has ended, else None."""
    popen = run.get("popen")
    if popen is not None:
        code = popen.poll()
        if code is None:
            return None
        return (None, -code) if code < 0 else (code, None)
    status_file = os.path.join(run["output"], _EXIT_STATUS_FILE)
    if run["kind"] == "terminal" and os.path.isfile(status_file):
        try:
            code = int(open(status_file).read().strip() or "-1")
        except (OSError, ValueError):
            code = -1
        if code > 128:
            return (None, code - 128)
        return (code, None)
    pid = _solver_pid(run)
    if pid and _run_pid_matches(run, pid):
        return None
    launcher = run.get("launcher_pid") or _read_launcher_pid(run["output"])
    if launcher and _run_pid_matches(run, launcher, launcher=True):
        return None
    completion = _manifest_of_run(run).get("completion", {})
    if isinstance(completion, dict) and completion.get("status") not in (
            None, "running"):
        code = completion.get("exit_status")
        if not isinstance(code, int):
            code = 0 if completion.get("status") == "completed" else 1
        return (code, None)
    if time.time() - run["started"] < 20.0 and not pid:
        return None     # still starting: no pid known yet
    return (-1, None)   # ended without leaving a status


def run_status_text(run, now=None):
    """The Run Status line for a run, from its process, log and manifest."""
    now = time.time() if now is None else now
    elapsed = _elapsed((run.get("ended") or now) - run["started"])
    tail = _log_tail(run["log"]) if run.get("log") else ""
    # PolyFEM logs "k/N  t=..." when step k is done. A busy step can push
    # that line out of the log's tail, so keep the last one seen.
    progress = _progress(tail) or run.get("progress")
    run["progress"] = progress
    finished = run.get("result")
    if finished is None:
        pid = _solver_pid(run)
        done = (f"{progress[0]} of {progress[1]} time steps done "
                f"(t = {progress[2]})" if progress
                else "setting up and solving the first time step")
        return f"Running{f' (pid {pid})' if pid else ''}: {done}, {elapsed}."
    code, signal_number = finished
    where = (f"after {progress[0]} of {progress[1]} time steps (t = "
             f"{progress[2]})" if progress
             else "before the first time step was done")
    if run.get("stop_requested"):
        return (f"Stopped by you {where}, {elapsed} into the run; the steps "
                "already written stay on disk.")
    if code == 0:
        return (f"Completed in {elapsed}"
                f"{f', {progress[1]} time steps' if progress else ''} (exit "
                "status 0). Press Open Results to view it.")
    if code == -1 and not signal_number:
        return (f"Ended {where}, {elapsed} into the run, without reporting "
                "an exit status; see Open Log.")
    reason = _stopped_reason(tail, _manifest_of_run(run))
    if signal_number:
        head = f"Crashed {where}: {exit_status_meaning(None, signal_number)}"
    elif code == 1:
        head = (f"Failed {where}: exit status 1, a named failure (not a "
                "crash)")
    elif code == 3:
        head = (f"Stopped {where}: exit status 3, a resource limit (a safety "
                "stop, not a crash; the steps already written are intact)")
    else:
        head = f"Failed {where}: exit status {code}"
    return head + (f". PolyFEM stopped: {reason}" if reason
                   else ". See Open Log for the reason.")


def _set_run_status(node, text):
    parm = node.parm("run_status") if node is not None else None
    if parm is not None and parm.unexpandedString() != text:
        with hou.undos.disabler():
            parm.set(text)


def _node_of(run):
    node = hou.node(run["node"]) if run.get("node") else None
    if node is None or node.sessionId() != run.get("session_id",
                                                   node.sessionId()):
        return None
    return node


def poll_runs():
    """Update every tracked run once; True while any is still going."""
    active = False
    for output, run in list(_runs().items()):
        if run.get("result") is None:
            result = _finished(run)
            if result is not None:
                run["result"] = result
                run["ended"] = time.time()
                handle = run.pop("log_handle", None)
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
            else:
                active = True
        node = _node_of(run)
        text = run_status_text(run)
        _set_run_status(node, text)
        if run.get("result") is not None and not run.get("announced"):
            run["announced"] = True
            _status(f"PolyFEM ({os.path.dirname(output)}): {text}")
    return active


def _poll_callback():
    state = hou.session.__dict__.setdefault("polyfem_poll_state", {})
    now = time.monotonic()
    if now - state.get("last", 0.0) < _POLL_SECONDS:
        return
    state["last"] = now
    try:
        active = poll_runs()
    except Exception as exc:     # never let a poll error spam the UI
        print(f"[PolyFEM HDA] run status poll failed: {exc}")
        active = False
    if not active:
        _stop_poller()


def _start_poller():
    if not hou.isUIAvailable():
        return
    callback = hou.session.__dict__.get("polyfem_poll_callback")
    if callback is not None and callback in hou.ui.eventLoopCallbacks():
        return
    hou.ui.addEventLoopCallback(_poll_callback)
    hou.session.polyfem_poll_callback = _poll_callback


def _stop_poller():
    callback = hou.session.__dict__.get("polyfem_poll_callback")
    if callback is not None and hou.isUIAvailable():
        try:
            hou.ui.removeEventLoopCallback(callback)
        except hou.OperationFailed:
            pass
    hou.session.__dict__.pop("polyfem_poll_callback", None)


def _register_run(parent, kind, output, log, popen=None, launcher_pid=None,
                  binary=None):
    run = {"output": output, "node": parent.path(),
           "session_id": parent.sessionId(), "kind": kind, "popen": popen,
           "pid": popen.pid if popen is not None else None,
           "launcher_pid": launcher_pid, "started": time.time(), "log": log,
           "binary": binary, "result": None,
           "process_created": _process_created(popen.pid) if popen else None}
    _runs()[output] = run
    _write_json(os.path.join(output, _RUN_RECORD), {
        "schema": "polyfem-houdini-run", "version": 1, "kind": kind,
        "node": parent.path(), "started": run["started"],
        "pid": run["pid"], "log": log, "binary": binary,
        "process_created": run["process_created"], "houdini_pid": os.getpid()})
    _set_run_status(parent, run_status_text(run))
    _start_poller()
    return run


def _run_for(parent, output=None):
    output = output or _output_dir(parent)
    return _runs().get(output) if output else None


def active_run(parent):
    """The run still writing into this node's output folder, or None.

    Looks at this session's runs and, for runs started elsewhere (another
    node, an earlier Houdini session, a terminal), at the folder's run
    record and PolyFEM's manifest; a dead or recycled pid does not count.
    """
    output = _output_dir(parent)
    if not output:
        return None
    run = _runs().get(output)
    if run is not None and run.get("result") is None:
        if _finished(run) is None:
            return run
        poll_runs()
    record = _read_json(os.path.join(output, _RUN_RECORD)) or {}
    external = {"output": output, "kind": "external",
                "started": record.get("started"), "binary": record.get("binary"),
                "process_created": record.get("process_created")}
    manifest = _manifest(output)
    completion = manifest.get("completion", {})
    process = manifest.get("process", {})
    pid = process.get("pid") if isinstance(process, dict) else None
    if isinstance(completion, dict) and completion.get("status") == "running" \
            and _run_pid_matches(external, pid):
        return dict(external, pid=pid)
    if not manifest and record.get("kind") == "background":
        pid = record.get("pid")
        if _run_pid_matches(external, pid):
            return dict(external, pid=pid)
    launcher = _read_launcher_pid(output)
    if record.get("kind") == "terminal" and launcher \
            and not os.path.isfile(os.path.join(output, _EXIT_STATUS_FILE)) \
            and _run_pid_matches(external, launcher, launcher=True):
        return dict(external, pid=None, launcher_pid=launcher)
    return None


def _read_launcher_pid(output):
    try:
        return int(open(os.path.join(output, _LAUNCHER_PID_FILE)).read())
    except (OSError, ValueError):
        return None


def _describe_process(run):
    pid = run.get("pid") if run.get("kind") == "external" \
        else _solver_pid(run)
    if pid:
        return f"process {pid}"
    if run.get("launcher_pid"):
        return f"terminal script, process {run['launcher_pid']}"
    return "starting"


def _clear_run_files(output):
    """Remove the previous run's status and log files before a new run starts,
    so nothing of it (its pid in PolyFEM's manifest, its exit status, or its
    logged steps) is read as the new run's. The user has already agreed to
    replace that run."""
    for name in (_EXIT_STATUS_FILE, _LAUNCHER_PID_FILE, "run-manifest.json",
                 "log.txt", _CONSOLE_LOG, _TERMINAL_LOG):
        path = os.path.join(output, name)
        if os.path.isfile(path):
            os.remove(path)


def run_background(kwargs):
    """Write params.json and run PolyFEM without a terminal, following it in
    Run Status (log: <working_dir>/output/log.txt)."""
    parent = kwargs["node"]
    params_path = _prepare_inputs(parent, "background")
    if not params_path:
        return None
    try:
        input_dir, args = _launch_command(parent, params_path)
    except hou.NodeError as e:
        _message(str(e))
        return None
    output = os.path.normpath(os.path.join(os.path.dirname(input_dir),
                                           "output"))
    _clear_run_files(output)
    log_path = os.path.join(output, "log.txt")
    capture_path = log_path
    if parent.evalParm("write_log"):
        # Save Log makes PolyFEM write log.txt itself; capturing the console
        # into the same file wrote every line twice. The console (which also
        # holds anything printed outside PolyFEM's logger) goes to its own
        # file, and Run Status follows PolyFEM's log.
        capture_path = os.path.join(output, _CONSOLE_LOG)
    options = {}
    if platform.system() == "Windows":
        options["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        # its own session: it survives Houdini quitting, and Stop can signal
        # the whole process group
        options["start_new_session"] = True
    try:
        with open(capture_path, "w") as log_file:
            proc = subprocess.Popen(args, cwd=input_dir, stdout=log_file,
                                    stderr=subprocess.STDOUT, **options)
    except OSError as exc:
        _message(f"Could not start PolyFEM: {exc}")
        return None
    run = _register_run(parent, "background", output, log_path, popen=proc,
                        binary=args[0])
    _status(f"PolyFEM running in background (pid {proc.pid}); log: {log_path}")
    return run


def write_launch_script(input_dir, output, args, system=None):
    """A script that runs PolyFEM in a terminal window and records what the
    node needs to follow it. Every path is quoted for the shell that runs
    it, so spaces, apostrophes and $ in folder names are safe."""
    system = system or platform.system()
    status = os.path.join(output, _EXIT_STATUS_FILE)
    if system == "Windows":
        def bat(text):
            return '"' + str(text).replace("%", "%%") + '"'
        path = os.path.join(output, "run_polyfem.bat")
        lines = ["@echo off",
                 "rem Written by the Houdini PolyFEM node: runs PolyFEM in "
                 "this window.",
                 f"cd /d {bat(input_dir)} || exit /b 1",
                 " ".join(bat(arg) for arg in args),
                 f"> {bat(status)} echo %ERRORLEVEL%",
                 "echo.",
                 "echo PolyFEM finished. The Houdini node's Run Status says "
                 "what the exit status means."]
        with open(path, "w", newline="\r\n") as handle:
            handle.write("\n".join(lines) + "\n")
        return path
    quote = shlex.quote
    path = os.path.join(output, "run_polyfem.command" if system == "Darwin"
                        else "run_polyfem.sh")
    command = " ".join(quote(arg) for arg in args)
    text = f"""#!/bin/bash
# Written by the Houdini PolyFEM node: runs PolyFEM in this terminal window.
# The node follows the run through the files written next to this script.
cd {quote(input_dir)} || exit 1
echo $$ > {quote(os.path.join(output, _LAUNCHER_PID_FILE))}
{command} 2>&1 | tee {quote(os.path.join(output, _TERMINAL_LOG))}
status=${{PIPESTATUS[0]}}
echo "$status" > {quote(status)}
echo
echo "PolyFEM finished with exit status $status (the Houdini node's Run Status says what it means). You can close this window."
"""
    with open(path, "w") as handle:
        handle.write(text)
    os.chmod(path, 0o755)
    return path


def terminal_command(script, system=None):
    """argv that opens a terminal window running `script`, or None."""
    system = system or platform.system()
    if system == "Darwin":
        return ["open", "-a", "Terminal", script]
    if system == "Windows":
        return ["cmd", "/c", "start", "", "cmd", "/k", script]
    for name, prefix in (("x-terminal-emulator", ["-e"]),
                         ("gnome-terminal", ["--"]), ("konsole", ["-e"]),
                         ("xfce4-terminal", ["-x"]), ("xterm", ["-hold", "-e"])):
        found = shutil.which(name)
        if found:
            return [found] + prefix + [script]
    return None


def write_params(kwargs):
    """Run PolyFEM button: write params.json and run PolyFEM in a terminal
    window, following it in Run Status like a background run."""
    parent = kwargs["node"]
    params_path = _prepare_inputs(parent, "run")
    if not params_path:
        return None
    try:
        input_dir, args = _launch_command(parent, params_path)
    except hou.NodeError as e:
        _message(str(e))
        return None
    output = os.path.normpath(os.path.join(os.path.dirname(input_dir),
                                           "output"))
    _clear_run_files(output)
    script = write_launch_script(input_dir, output, args)
    command = terminal_command(script)
    if command is None:
        _message("No terminal program was found (tried x-terminal-emulator, "
                 "gnome-terminal, konsole, xfce4-terminal, xterm); use Run in "
                 f"Background instead, or run {script} yourself.")
        return None
    try:
        subprocess.Popen(command)
    except OSError as exc:
        _message(f"Could not open terminal for PolyFEM: {exc}")
        return None
    run = _register_run(parent, "terminal", output,
                        os.path.join(output, _TERMINAL_LOG), binary=args[0])
    _status(f"PolyFEM started in a terminal window ({script}).")
    return run


def reattach_run(parent):
    """Follow a run this node started before the scene was reloaded (or that
    another session started in its folder). Called by OnLoaded."""
    output = _output_dir(parent)
    if not output:
        return None
    existing = _runs().get(output)
    if existing is not None and existing.get("result") is None \
            and _node_of(existing) is not None:
        return None
    record = _read_json(os.path.join(output, _RUN_RECORD))
    if not record:
        return None
    kind = record.get("kind", "background")
    log = record.get("log") or os.path.join(output, "log.txt")
    run = {"output": output, "node": parent.path(),
           "session_id": parent.sessionId(), "kind": kind, "popen": None,
           "pid": record.get("pid"),
           "launcher_pid": _read_launcher_pid(output),
           "started": float(record.get("started", time.time())), "log": log,
           "binary": record.get("binary"), "result": None,
           "process_created": record.get("process_created")}
    _runs()[output] = run
    poll_runs()
    if run.get("result") is None:
        _start_poller()
    return run


def stop_run(kwargs):
    """Stop button: end the run writing into this node's working directory."""
    parent = kwargs["node"]
    run = active_run(parent)
    if run is None:
        _status("No PolyFEM run is going on in this Working Directory.")
        return False
    if hou.isUIAvailable():
        choice = hou.ui.displayMessage(
            f"Stop the PolyFEM run in {run['output']}?\n\nThe time steps "
            "already written stay on disk.", buttons=("Stop", "Cancel"),
            severity=hou.severityType.Warning, default_choice=1,
            close_choice=1, title="PolyFEM: Stop")
        if choice != 0:
            return False
    import signal
    tracked = _runs().get(run["output"])
    popen = run.get("popen")
    pid = _solver_pid(run)
    try:
        if popen is not None and popen.poll() is None:
            if platform.system() == "Windows":
                popen.terminate()
            else:
                os.killpg(popen.pid, signal.SIGTERM)
        elif popen is None and _run_pid_matches(run, pid, for_signal=True):
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=10, check=True)
            else:
                os.kill(pid, signal.SIGTERM)
        else:
            launcher = run.get("launcher_pid") or _read_launcher_pid(run["output"])
            if popen is not None or not _run_pid_matches(
                    run, launcher, launcher=True, for_signal=True) \
                    or os.getpgid(launcher) != launcher:
                _status("PolyFEM Stop refused: process identity unknown.")
                return False
            os.killpg(launcher, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError) as exc:
        _message(f"Could not stop the run: {exc}")
        return False
    if tracked is not None:
        tracked["stop_requested"] = True
    if tracked is None:
        _set_run_status(parent, "Stopped by you; the steps already written "
                                "stay on disk.")
    else:
        poll_runs()
    return True


def log_path_for(parent):
    """The log of this node's latest run (Run in Background: log.txt; Run in
    Terminal: terminal_log.txt), or PolyFEM's own Save Log file."""
    output = _output_dir(parent)
    if not output:
        return None
    run = _runs().get(output)
    if run is not None and run.get("log") and os.path.isfile(run["log"]):
        return run["log"]
    candidates = [os.path.join(output, name)
                  for name in ("log.txt", _TERMINAL_LOG)]
    existing = [path for path in candidates if os.path.isfile(path)]
    if not existing:
        return None
    return max(existing, key=os.path.getmtime)


def _open_with_system(path):
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", "-t", path])
    elif system == "Windows":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])


def open_log(kwargs):
    """Open Log button: the latest run's log in the system's text viewer."""
    path = log_path_for(kwargs["node"])
    if not path:
        _message("There is no log yet: run the simulation first.")
        return None
    _open_with_system(path)
    return path


def show_log(kwargs):
    """Show Log button: the last lines of the latest run's log."""
    parent = kwargs["node"]
    log_path = log_path_for(parent)
    if not log_path:
        _message(f"No log found in {_output_dir(parent)}")
        return
    tail = _log_tail(log_path).splitlines()[-60:]
    if hou.isUIAvailable():
        hou.ui.displayMessage("\n".join(tail) or "(log empty)",
                              title=os.path.basename(log_path))
    else:
        print("\n".join(tail))


def results_pvd(parent):
    return os.path.join(parent.evalParm("working_dir"), "output",
                        parent.evalParm("paraview_file_name"))


_RESULTS_NODE = "polyfem_results_node"


def open_results(kwargs):
    """Open Results button: a Read PVD node on this run's .pvd (created next
    to this node the first time, then reused), the playbar set to its time
    steps, and the viewport framed on it."""
    parent = kwargs["node"]
    pvd = results_pvd(parent)
    if not os.path.isfile(pvd):
        _message(f"There are no results yet ({pvd} does not exist): run the "
                 "simulation first.")
        return None
    node_type = hou.nodeType(hou.objNodeTypeCategory(), "readPVD::1.0")
    if node_type is None:
        _message("The Read PVD 1.0 asset is not installed.")
        return None
    reader = None
    path = parent.userData(_RESULTS_NODE)
    if path:
        candidate = hou.node(path)
        if candidate is not None and candidate.type() == node_type:
            reader = candidate
    if reader is None:
        network = parent.parent()
        reader = network.createNode("readPVD::1.0",
                                    f"{parent.name()}_results")
        reader.setPosition(parent.position() + hou.Vector2(0, -1.5))
        parent.setUserData(_RESULTS_NODE, reader.path())
    value = portable_path(parent, pvd)
    same = os.path.normpath(reader.evalParm("PVD_file") or "") == \
        os.path.normpath(pvd)
    reader.parm("PVD_file").set(value)
    module = reader.hdaModule()
    if same:
        module.refresh({"node": reader}, force_clear_cache=True)
    else:
        module.start({"node": reader})
    reader.setDisplayFlag(True)
    if hou.isUIAvailable():
        reader.setSelected(True, clear_all_selected=True)
        reader.setCurrent(True)
        module._frame_result(reader)
    return reader


