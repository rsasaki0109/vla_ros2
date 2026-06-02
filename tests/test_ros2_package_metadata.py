"""Non-ROS metadata tests for the ros2/ packages.

These tests parse package.xml, setup.py, CMakeLists.txt, .msg, and launch files
with the standard library only, so CI can guard ROS2 integration shape without a
ROS2 / colcon installation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[1]
ROS_PKG = REPO_ROOT / "ros2" / "vla_zoo"
MSGS_PKG = REPO_ROOT / "ros2" / "vla_zoo_msgs"
LAUNCH_DIR = ROS_PKG / "launch"
MSG_DIR = MSGS_PKG / "msg"

EXPECTED_ENTRY_POINTS = {
    "vla_action_replay_node": "vla_zoo_ros.action_replay:main",
    "vla_runtime_node": "vla_zoo_ros.node:main",
    "vla_runtime_recorder": "vla_zoo_ros.log_recorder:main",
    "vla_smoke_input_node": "vla_zoo_ros.smoke_input:main",
}

EXPECTED_SMOKE_TOPIC_ARGS = {
    "image_topic",
    "instruction_topic",
    "joint_state_topic",
    "action_topic",
    "status_topic",
    "diagnostics_topic",
}


def _parse_package_xml(path: Path) -> dict[str, object]:
    root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    build_type_el = root.find("./export/build_type")
    return {
        "format": root.attrib.get("format"),
        "name": (root.findtext("name") or "").strip(),
        "build_type": (build_type_el.text or "").strip() if build_type_el is not None else None,
        "exec_depend": {el.text.strip() for el in root.findall("exec_depend") if el.text},
        "build_depend": {el.text.strip() for el in root.findall("build_depend") if el.text},
        "depend": {el.text.strip() for el in root.findall("depend") if el.text},
        "groups": {el.text.strip() for el in root.findall("member_of_group") if el.text},
    }


def _parse_msg_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" in line:  # skip blanks, comments, and constants
            continue
        parts = line.split()
        if len(parts) >= 2:
            fields[parts[1]] = parts[0]
    return fields


def _find_setup_call(source: str) -> ast.Call:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setup"
        ):
            return node
    raise AssertionError("setup() call not found in setup.py")


def _setup_keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"setup() keyword {name!r} not found")


def _declared_launch_args(source: str) -> dict[str, object]:
    """Return DeclareLaunchArgument name -> default (None if not a constant)."""

    args: dict[str, object] = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "DeclareLaunchArgument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = str(node.args[0].value)
        default: object = None
        for keyword in node.keywords:
            if keyword.arg == "default_value" and isinstance(keyword.value, ast.Constant):
                default = keyword.value.value
        args[name] = default
    return args


def _node_executables(source: str) -> set[str]:
    executables: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "Node":
            continue
        for keyword in node.keywords:
            if keyword.arg == "executable" and isinstance(keyword.value, ast.Constant):
                executables.add(str(keyword.value.value))
    return executables


def test_vla_zoo_package_xml() -> None:
    meta = _parse_package_xml(ROS_PKG / "package.xml")
    assert meta["name"] == "vla_zoo"
    assert meta["format"] == "3"
    assert meta["build_type"] == "ament_python"
    assert {"rclpy", "sensor_msgs", "std_msgs", "diagnostic_msgs", "vla_zoo_msgs"}.issubset(
        meta["exec_depend"]  # type: ignore[arg-type]
    )


def test_vla_zoo_msgs_package_xml() -> None:
    meta = _parse_package_xml(MSGS_PKG / "package.xml")
    assert meta["name"] == "vla_zoo_msgs"
    assert meta["build_type"] == "ament_cmake"
    assert "rosidl_default_generators" in meta["build_depend"]  # type: ignore[operator]
    assert "rosidl_interface_packages" in meta["groups"]  # type: ignore[operator]


def test_cmakelists_lists_all_existing_messages() -> None:
    cmake = (MSGS_PKG / "CMakeLists.txt").read_text(encoding="utf-8")
    msg_files = sorted(p.name for p in MSG_DIR.glob("*.msg"))
    assert msg_files  # sanity: messages exist
    for name in msg_files:
        assert f"msg/{name}" in cmake, f"{name} not registered in CMakeLists.txt"


def test_msg_definitions_have_expected_fields() -> None:
    action = _parse_msg_fields(MSG_DIR / "VLAAction.msg")
    assert action.get("data") == "float32[]"
    assert action.get("action_space") == "string"
    assert action.get("model_name") == "string"

    chunk = _parse_msg_fields(MSG_DIR / "VLAActionChunk.msg")
    assert chunk.get("actions") == "vla_zoo_msgs/VLAAction[]"

    status = _parse_msg_fields(MSG_DIR / "VLAStatus.msg")
    assert status.get("ready") == "bool"
    assert status.get("dry_run") == "bool"

    instruction = _parse_msg_fields(MSG_DIR / "VLAInstruction.msg")
    assert instruction.get("text") == "string"
    assert instruction.get("task_id") == "string"


def test_setup_py_declares_node_entry_points() -> None:
    call = _find_setup_call((ROS_PKG / "setup.py").read_text(encoding="utf-8"))
    entry_points = ast.literal_eval(_setup_keyword(call, "entry_points"))
    scripts = entry_points["console_scripts"]
    parsed = dict(item.split(" = ", 1) for item in scripts)
    assert parsed == EXPECTED_ENTRY_POINTS


def test_entry_point_modules_exist() -> None:
    for target in EXPECTED_ENTRY_POINTS.values():
        module_path = target.split(":", 1)[0]  # e.g. vla_zoo_ros.node
        relative = Path(*module_path.split(".")).with_suffix(".py")
        assert (ROS_PKG / relative).is_file(), f"missing module for entry point: {target}"


def test_setup_py_installs_launch_and_config() -> None:
    source = (ROS_PKG / "setup.py").read_text(encoding="utf-8")
    assert "launch/*.launch.py" in source
    assert "config/*.yaml" in source


def test_launch_files_are_valid_python_with_entrypoint() -> None:
    launch_files = sorted(LAUNCH_DIR.glob("*.launch.py"))
    assert launch_files
    for path in launch_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)  # raises SyntaxError on broken launch files
        functions = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert "generate_launch_description" in functions, f"{path.name} missing entrypoint"


def test_smoke_launch_declares_runtime_topics_and_nodes() -> None:
    source = (LAUNCH_DIR / "smoke.launch.py").read_text(encoding="utf-8")
    args = _declared_launch_args(source)
    assert EXPECTED_SMOKE_TOPIC_ARGS.issubset(args.keys())
    executables = _node_executables(source)
    assert {"vla_runtime_node", "vla_smoke_input_node"}.issubset(executables)


def test_launch_dry_run_defaults_to_true() -> None:
    """Safety: any launch file exposing dry_run must default it to 'true'."""

    checked = 0
    for path in sorted(LAUNCH_DIR.glob("*.launch.py")):
        args = _declared_launch_args(path.read_text(encoding="utf-8"))
        if "dry_run" in args:
            checked += 1
            assert args["dry_run"] == "true", f"{path.name} dry_run default is not 'true'"
    assert checked > 0, "expected at least one launch file to declare dry_run"
