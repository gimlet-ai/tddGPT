import select
import subprocess
import asyncio
from typing import List, Optional, Type, Union

from pydantic import BaseModel, Field, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool

import platform

def _get_platform() -> str:
    """Get platform."""
    system = platform.system()
    if system == "Darwin":
        return "MacOS"
    return system

class CommandTimeout(Exception):
    pass

def run_command_with_timeout(cmd, timeout_sec):
    """Run cmd in the shell and return the output. 
    If there's no activity on stdout for timeout_sec, return a timeout message.
    """
    # If cmd is a list, join the elements into a single string
    if isinstance(cmd, list):
        cmd = ' && '.join(cmd)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    output = ''
    while True:
        # Wait for output or a timeout
        reads = [proc.stdout.fileno()]
        ret = select.select(reads, [], [], timeout_sec)

        if proc.stdout.fileno() in ret[0]:
            # There's output to read
            line = proc.stdout.readline()
            if line:
                output += line.decode()
            else:
                # No more output, break the loop
                break
        else:
            # Timeout with no output, kill the process
            proc.kill()
            return f"Command '{cmd}' timed out after {timeout_sec} seconds of inactivity on stdout {output}"

    # Check for errors
    if proc.poll():
        return f"Command '{cmd}' failed with error: {output}"

    return f"Command '{cmd}' succeeded with the following output:\n{output}"

class CLIInput(BaseModel):
    """Commands for the CLI tool."""

    commands: Union[str, List[str]] = Field(
        ...,
        description="List of shell commands to run. Deserialized using json.loads",
    )
    """List of shell commands to run."""

    @root_validator
    def _validate_commands(cls, values: dict) -> dict:
        """Validate commands."""
        commands = values.get("commands")
        if not isinstance(commands, list):
            values["commands"] = [commands]
        return values

class CLITool(BaseTool):
    name: str = "cli"
    """Name of tool."""

    description: str = f"Run cli commands on this {_get_platform()} machine."
    """Description of tool."""

    args_schema: Type[BaseModel] = CLIInput
    """Schema for input arguments."""

    def _run(
        self,
        commands: Union[str, List[str]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run commands and return final output."""
        output = run_command_with_timeout(commands, 60)
        return output

    async def _arun(
        self,
        commands: Union[str, List[str]],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run commands asynchronously and return final output."""
        return await asyncio.get_event_loop().run_in_executor(
            None, run_command_with_timeout, commands, 60
        )
