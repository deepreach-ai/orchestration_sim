"""
orchestration/rm75b_protocol.py
---------------------------------
JSON command schema for RM75-B real-robot TCP communication.

The RM75-B accepts JSON strings over TCP (default port 8080).
This module defines:
  - All command dataclasses (send: PC → arm)
  - All response dataclasses (receive: arm → PC)
  - RM75BClient: async TCP client that wraps send/receive
  - RM75BMockClient: drop-in replacement for Isaac Sim / offline testing

Usage (real robot):
    from orchestration.rm75b_protocol import RM75BClient, MoveJointsCmd
    async with RM75BClient("192.168.1.18") as arm:
        await arm.send(MoveJointsCmd(joints=[0.0, -0.3, 0.0, -1.4, 0.0, 1.1, 0.0], speed=20))

Usage (mock / sim):
    from orchestration.rm75b_protocol import RM75BMockClient
    arm = RM75BMockClient()           # identical API, no network needed
    await arm.send(MoveJointsCmd(...))

Reference: 睿尔曼 RM75-B JSON 通讯协议手册 (TCP port 8080)
Real robot IP default: 192.168.1.18   (set in configs/rm75b.yaml)

New concept: Protocol schema
-----------------------------
A "schema" here means the agreed structure of every JSON message.
Defining it in Python dataclasses gives us:
  - Type checking at write time (IDE autocomplete)
  - Automatic serialization to/from dict (via dataclasses.asdict)
  - Single source of truth: change here, propagates everywhere
  - Easy mock: same dataclass, different sender (network vs no-op)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from enum import IntEnum
from typing import List, Optional

logger = logging.getLogger("orchestration.rm75b_protocol")

# ── Real robot defaults (matches configs/rm75b.yaml) ─────────────────────────
DEFAULT_IP   = "192.168.1.18"
DEFAULT_PORT = 8080
TIMEOUT_S    = 2.0   # seconds to wait for ACK before raising

# ─────────────────────────────────────────────────────────────────────────────
#  Command type IDs — from RM75-B protocol manual
# ─────────────────────────────────────────────────────────────────────────────
class CmdType(IntEnum):
    """
    Top-level command type field ("command" key in JSON).
    Matches the RM75-B TCP protocol command set.
    """
    MOVE_JOINTS      = 1    # set 7 joint angles (position control)
    MOVE_CARTESIAN   = 2    # set EE pose in Cartesian space
    SET_GRIPPER      = 3    # open / close gripper
    GET_STATE        = 4    # query: return current joint angles + EE pose
    MOVE_HOME        = 5    # move to factory home position
    EMERGENCY_STOP   = 99   # immediate halt (all joints brake)


# ─────────────────────────────────────────────────────────────────────────────
#  Base command
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class BaseCmd:
    """
    All commands include a 'command' type int.
    to_json() serialises to the wire format.

    command defaults to 0 here; every subclass overwrites it
    in __post_init__ with the correct CmdType value.
    Using a default avoids the 'missing positional argument' error
    that occurs when a parent dataclass field has no default but
    a child field does (Python requires defaults-last ordering).
    """
    command: int = dataclasses.field(default=0, init=False)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
#  Concrete commands
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MoveJointsCmd(BaseCmd):
    """
    Move arm to absolute joint positions.

    joints : list of 7 floats, radians, must be within URDF limits.
    speed  : percentage 1–100 (mapped to max_velocity in rm75b.yaml)

    Wire JSON example:
        {"command":1,"joints":[0.0,-0.3,0.0,-1.4,0.0,1.1,0.0],"speed":20}
    """
    joints: List[float] = dataclasses.field(default_factory=lambda: [0.0]*7)
    speed:  int         = 20       # safe default: 20% speed

    def __post_init__(self):
        self.command = int(CmdType.MOVE_JOINTS)
        assert len(self.joints) == 7, "RM75-B has exactly 7 joints"
        assert 1 <= self.speed <= 100, "speed must be 1-100"


@dataclasses.dataclass
class MoveCartesianCmd(BaseCmd):
    """
    Move EE to Cartesian pose (position + quaternion).

    position   : [x, y, z] in metres (world frame)
    quaternion : [qx, qy, qz, qw] unit quaternion
    speed      : percentage 1-100

    Wire JSON example:
        {"command":2,"position":[0.3,0.0,0.5],"quaternion":[1.0,0.0,0.0,0.0],"speed":20}
    """
    position:   List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.5])
    quaternion: List[float] = dataclasses.field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    speed:      int         = 20

    def __post_init__(self):
        self.command = int(CmdType.MOVE_CARTESIAN)
        assert len(self.position) == 3
        assert len(self.quaternion) == 4


@dataclasses.dataclass
class SetGripperCmd(BaseCmd):
    """
    Control Left_1_Joint (the only revolute gripper joint).

    position : float in [0.0, 0.5] rad.
               0.0 = fully closed, 0.5 = fully open.
    force    : int 1-100, gripping force percentage.

    Wire JSON example:
        {"command":3,"position":0.0,"force":50}
    """
    position: float = 0.5    # default: open
    force:    int   = 50

    def __post_init__(self):
        self.command = int(CmdType.SET_GRIPPER)
        assert 0.0 <= self.position <= 0.5, "gripper position must be in [0.0, 0.5] rad"

    @classmethod
    def open(cls, force: int = 30) -> "SetGripperCmd":
        return cls(position=0.5, force=force)

    @classmethod
    def close(cls, force: int = 50) -> "SetGripperCmd":
        return cls(position=0.0, force=force)


@dataclasses.dataclass
class GetStateCmd(BaseCmd):
    """
    Query current robot state. No extra fields needed.
    Wire JSON: {"command":4}
    """
    def __post_init__(self):
        self.command = int(CmdType.GET_STATE)


@dataclasses.dataclass
class MoveHomeCmd(BaseCmd):
    """
    Move to factory home position.
    Wire JSON: {"command":5,"speed":20}
    """
    speed: int = 20

    def __post_init__(self):
        self.command = int(CmdType.MOVE_HOME)


@dataclasses.dataclass
class EmergencyStopCmd(BaseCmd):
    """
    Immediate halt — all joints brake.
    Wire JSON: {"command":99}

    ⚠️  After e-stop, send MoveHomeCmd to re-enable.
    Note: joints 5-7 have no brake; they may drift after power-off.
    """
    def __post_init__(self):
        self.command = int(CmdType.EMERGENCY_STOP)


# ─────────────────────────────────────────────────────────────────────────────
#  Response schema (arm → PC)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ArmResponse:
    """
    Parsed response from the RM75-B.

    success      : True if command was executed successfully
    error_code   : 0 = OK, non-zero = error (see RM75-B error table)
    error_msg    : human-readable error string (may be empty)
    joints       : current 7 joint angles (radians), populated by GET_STATE
    ee_position  : [x, y, z] EE position (metres), populated by GET_STATE
    ee_quaternion: [qx, qy, qz, qw] EE orientation, populated by GET_STATE
    gripper_pos  : current Left_1_Joint position (radians), populated by GET_STATE

    New concept: error_code table
    ------------------------------
    0   : OK
    1   : Joint limit exceeded
    2   : IK not converged (Cartesian move failed)
    3   : Collision detected (force threshold exceeded)
    4   : Communication timeout
    99  : Emergency stop active (must reset before next move)
    """
    success:       bool          = False
    error_code:    int           = 0
    error_msg:     str           = ""
    joints:        List[float]   = dataclasses.field(default_factory=list)
    ee_position:   List[float]   = dataclasses.field(default_factory=list)
    ee_quaternion: List[float]   = dataclasses.field(default_factory=list)
    gripper_pos:   float         = 0.5

    @classmethod
    def from_json(cls, raw: str) -> "ArmResponse":
        try:
            d = json.loads(raw)
            return cls(
                success       = d.get("success", False),
                error_code    = d.get("error_code", -1),
                error_msg     = d.get("error_msg", "parse error"),
                joints        = d.get("joints", []),
                ee_position   = d.get("ee_position", []),
                ee_quaternion = d.get("ee_quaternion", []),
                gripper_pos   = d.get("gripper_pos", 0.5),
            )
        except json.JSONDecodeError as e:
            return cls(success=False, error_code=-1, error_msg=f"JSON decode: {e}")

    @classmethod
    def ok(cls, **kwargs) -> "ArmResponse":
        """Convenience: successful response with optional field overrides."""
        return cls(success=True, error_code=0, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  RM75BClient — real TCP client
# ─────────────────────────────────────────────────────────────────────────────

class RM75BClient:
    """
    Async TCP client for the RM75-B real robot.

    New concept: async context manager
    ------------------------------------
    Using 'async with RM75BClient(ip) as arm:' ensures the TCP connection
    is always closed cleanly, even if an exception occurs. The __aenter__
    opens the socket, __aexit__ closes it. This is idiomatic Python for
    resource management in async code.

    Example:
        async with RM75BClient("192.168.1.18") as arm:
            resp = await arm.send(MoveHomeCmd())
            if not resp.success:
                print(f"Error {resp.error_code}: {resp.error_msg}")
    """

    def __init__(self, ip: str = DEFAULT_IP, port: int = DEFAULT_PORT,
                 timeout: float = TIMEOUT_S):
        self.ip      = ip
        self.port    = port
        self.timeout = timeout
        self._reader: Optional[asyncio.StreamReader]  = None
        self._writer: Optional[asyncio.StreamWriter]  = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.ip, self.port),
            timeout=self.timeout,
        )
        logger.info(f"✅ Connected to RM75-B at {self.ip}:{self.port}")

    async def disconnect(self) -> None:
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._reader = self._writer = None
        logger.info("🔌 Disconnected from RM75-B")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.disconnect()

    async def send(self, cmd: BaseCmd) -> ArmResponse:
        """
        Send a command and await the JSON ACK response.
        Returns ArmResponse (success=False on timeout or parse error).
        """
        if self._writer is None:
            raise RuntimeError("Not connected — call connect() or use async with")

        payload = (cmd.to_json() + "\n").encode()
        self._writer.write(payload)
        await self._writer.drain()
        logger.debug(f"→ {cmd.to_json()}")

        try:
            raw = await asyncio.wait_for(
                self._reader.readline(), timeout=self.timeout
            )
            resp = ArmResponse.from_json(raw.decode().strip())
            logger.debug(f"← {raw.decode().strip()}")
            if not resp.success:
                logger.warning(
                    f"⚠️  Arm error {resp.error_code}: {resp.error_msg} "
                    f"(cmd={cmd.command})"
                )
            return resp
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout waiting for ACK (cmd={cmd.command})")
            return ArmResponse(
                success=False, error_code=4,
                error_msg=f"TCP timeout after {self.timeout}s"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  RM75BMockClient — drop-in replacement for sim / offline testing
# ─────────────────────────────────────────────────────────────────────────────

class RM75BMockClient:
    """
    Mock client with identical API to RM75BClient.
    No network required. Used in:
      - Isaac Sim (LulaController drives the sim arm directly)
      - Unit tests
      - CI / offline development

    State tracking
    --------------
    The mock keeps a simulated joint state so GetStateCmd returns
    consistent values. MoveJointsCmd updates the internal state.
    This lets you test state-machine logic without hardware.

    New concept: dependency injection via mock
    -------------------------------------------
    Because both RM75BClient and RM75BMockClient share the same
    send(cmd) → ArmResponse API, the orchestration layer never needs
    to know which one it's using. You swap the client at startup:
        client = RM75BClient(ip) if USE_REAL else RM75BMockClient()
    This is the "dependency injection" pattern — high-level code
    depends on an abstract interface, not a concrete implementation.
    """

    HOME_JOINTS = [0.0, -0.3, 0.0, -1.4, 0.0, 1.1, 0.0]

    def __init__(self, verbose: bool = False):
        self._joints      = list(self.HOME_JOINTS)
        self._gripper_pos = 0.5   # open
        self._verbose     = verbose

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def send(self, cmd: BaseCmd) -> ArmResponse:
        ctype = cmd.command

        if ctype == CmdType.MOVE_JOINTS:
            self._joints = list(cmd.joints)  # type: ignore[attr-defined]
            if self._verbose:
                logger.info(f"[MOCK] MoveJoints → {self._joints}")
            return ArmResponse.ok(joints=self._joints)

        elif ctype == CmdType.MOVE_CARTESIAN:
            if self._verbose:
                logger.info(f"[MOCK] MoveCartesian → {cmd.position}")  # type: ignore
            return ArmResponse.ok()

        elif ctype == CmdType.SET_GRIPPER:
            self._gripper_pos = cmd.position  # type: ignore[attr-defined]
            if self._verbose:
                logger.info(f"[MOCK] SetGripper → {self._gripper_pos:.3f} rad")
            return ArmResponse.ok(gripper_pos=self._gripper_pos)

        elif ctype == CmdType.GET_STATE:
            return ArmResponse.ok(
                joints        = list(self._joints),
                ee_position   = [0.3, 0.0, 0.5],   # placeholder
                ee_quaternion = [1.0, 0.0, 0.0, 0.0],
                gripper_pos   = self._gripper_pos,
            )

        elif ctype == CmdType.MOVE_HOME:
            self._joints = list(self.HOME_JOINTS)
            if self._verbose:
                logger.info("[MOCK] MoveHome")
            return ArmResponse.ok(joints=self._joints)

        elif ctype == CmdType.EMERGENCY_STOP:
            logger.warning("[MOCK] ⚠️  EMERGENCY STOP")
            return ArmResponse.ok()

        return ArmResponse(success=False, error_code=-1,
                           error_msg=f"Unknown command: {ctype}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test (run directly: python orchestration/rm75b_protocol.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio, sys
    logging.basicConfig(level=logging.DEBUG)

    async def _test():
        print("\n── Mock client self-test ──────────────────────────────")
        arm = RM75BMockClient(verbose=True)

        resp = await arm.send(MoveHomeCmd())
        assert resp.success, "MoveHome failed"
        print(f"✅ MoveHome OK")

        resp = await arm.send(MoveJointsCmd(
            joints=[0.1, -0.3, 0.0, -1.4, 0.0, 1.1, 0.0], speed=30
        ))
        assert resp.success and resp.joints[0] == 0.1
        print(f"✅ MoveJoints OK → joints[0]={resp.joints[0]}")

        resp = await arm.send(SetGripperCmd.open())
        assert resp.gripper_pos == 0.5
        print(f"✅ SetGripper open OK → pos={resp.gripper_pos}")

        resp = await arm.send(SetGripperCmd.close())
        assert resp.gripper_pos == 0.0
        print(f"✅ SetGripper close OK → pos={resp.gripper_pos}")

        resp = await arm.send(GetStateCmd())
        assert len(resp.joints) == 7
        print(f"✅ GetState OK → joints={resp.joints}")

        resp = await arm.send(EmergencyStopCmd())
        assert resp.success
        print(f"✅ EmergencyStop OK")

        print("\n── Serialization check ────────────────────────────────")
        cmds = [
            MoveHomeCmd(speed=15),
            MoveJointsCmd(joints=[0.0]*7, speed=20),
            MoveCartesianCmd(position=[0.3, 0.0, 0.5], quaternion=[1,0,0,0]),
            SetGripperCmd.open(),
            SetGripperCmd.close(force=80),
            GetStateCmd(),
            EmergencyStopCmd(),
        ]
        for c in cmds:
            wire = c.to_json()
            parsed = json.loads(wire)
            assert parsed["command"] == c.command
            print(f"  {c.__class__.__name__:<22} → {wire}")

        print("\n✅ All tests passed")

    asyncio.run(_test())
