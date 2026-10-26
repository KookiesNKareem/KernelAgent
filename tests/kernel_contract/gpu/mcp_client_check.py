"""Drive the MCP server over stdio the way an agent would."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "kernel_contract.mcp_server", "--model",
                                                                 "kernel_contract.demo_models:llama_block"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
            sites = json.loads((await s.call_tool("list_sites", {"top": 4})).content[0].text)
            print([x["site"] for x in sites])
            rot = next(x["site"] for x in json.loads((await s.call_tool("list_sites", {"top": 10})).content[0].text)
                       if x["site"].startswith("apply_rotary"))
            for k in ("rotary_flat.py", "rotary_strided.py"):
                out = json.loads((await s.call_tool("check_kernel", {"site": rot, "kernel_path": str(HERE / k)})).content[0].text)
                print(k, out["verdict"], len(out["failures"]), out["deployable"], out["failures"][0]["means"] if out["failures"] else "")
            print((await s.call_tool("install_and_compare", {})).content[0].text)

asyncio.run(main())
