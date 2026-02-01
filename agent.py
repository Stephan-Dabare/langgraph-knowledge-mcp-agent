# -*- coding: utf-8 -*-
import json
import asyncio
import os
import sys
from dotenv import load_dotenv

# Fix Unicode output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools.base import ToolException
from langgraph.types import Command
from typing import Callable, Union

# MCP Imports
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """
    Custom middleware to handle tool execution errors gracefully.
    Converts ToolException errors into ToolMessage objects so the agent can recover.
    Also shows indicators when tools are being used.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]],
    ) -> Union[ToolMessage, Command]:
        """Sync version of tool call wrapper."""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        # Show tool usage with query if available
        query = tool_args.get("query", "")
        if query:
            print(f"\n🔍 Using tool: {tool_name} (query: \"{query}\")...", flush=True)
        else:
            print(f"\n🔍 Using tool: {tool_name}...", flush=True)

        try:
            result = handler(request)

            # Show retrieved content preview (limited to first 500 chars)
            if isinstance(result, ToolMessage):
                content_str = str(result.content)
                content_length = len(content_str)
                preview_length = 500

                if content_length > preview_length:
                    preview = content_str[:preview_length] + "..."
                    print(f"✅ Tool {tool_name} completed (Retrieved {content_length} chars)", flush=True)
                    print(f"📄 Content preview:\n{preview}\n", flush=True)
                else:
                    print(f"✅ Tool {tool_name} completed (Retrieved {content_length} chars)", flush=True)
                    print(f"📄 Content:\n{content_str}\n", flush=True)
            else:
                print(f"✅ Tool {tool_name} completed", flush=True)

            return result
        except ToolException as e:
            print(f"⚠️ Tool {tool_name} returned an error: {str(e)}", flush=True)
            return ToolMessage(
                content=f"Tool returned an error: {str(e)}. Please try a different search query or provide an answer based on your knowledge.",
                tool_call_id=request.tool_call["id"]
            )
        except Exception as e:
            print(f"❌ Tool {tool_name} failed: {str(e)}", flush=True)
            return ToolMessage(
                content=f"Unexpected tool error: {str(e)}. Please try again or answer from your knowledge.",
                tool_call_id=request.tool_call["id"]
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]],
    ) -> Union[ToolMessage, Command]:
        """Async version of tool call wrapper."""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        # Show tool usage with query if available
        query = tool_args.get("query", "")
        if query:
            print(f"\n🔍 Using tool: {tool_name} (query: \"{query}\")...", flush=True)
        else:
            print(f"\n🔍 Using tool: {tool_name}...", flush=True)

        try:
            result = await handler(request)

            # Show retrieved content preview (limited to first 500 chars)
            if isinstance(result, ToolMessage):
                content_str = str(result.content)
                content_length = len(content_str)
                preview_length = 500

                if content_length > preview_length:
                    preview = content_str[:preview_length] + "..."
                    print(f"✅ Tool {tool_name} completed (Retrieved {content_length} chars)", flush=True)
                    print(f"📄 Content preview:\n{preview}\n", flush=True)
                else:
                    print(f"✅ Tool {tool_name} completed (Retrieved {content_length} chars)", flush=True)
                    print(f"📄 Content:\n{content_str}\n", flush=True)
            else:
                print(f"✅ Tool {tool_name} completed", flush=True)

            return result
        except ToolException as e:
            print(f"⚠️ Tool {tool_name} returned an error: {str(e)}", flush=True)
            return ToolMessage(
                content=f"Tool returned an error: {str(e)}. Please try a different search query or provide an answer based on your knowledge.",
                tool_call_id=request.tool_call["id"]
            )
        except Exception as e:
            print(f"❌ Tool {tool_name} failed: {str(e)}", flush=True)
            return ToolMessage(
                content=f"Unexpected tool error: {str(e)}. Please try again or answer from your knowledge.",
                tool_call_id=request.tool_call["id"]
            )


def load_mcp_config(config_path: str = "mcp.json"):
    """
    Reads the mcp.json file and returns the MCP servers configuration.
    This makes the system extensible: add a server to JSON, and the agent sees it.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    return config.get("mcpServers", {})


async def run_agent_with_mcp():
    """
    Main function that runs the agent with MCP client.
    As of langchain-mcp-adapters 0.1.0+, the client is stateless by default.
    """
    # 1. Load MCP configuration
    print("🔌 Connecting to MCP Servers defined in mcp.json...")
    try:
        mcp_servers = load_mcp_config()
    except Exception as e:
        print(f"⚠️  Warning: Could not load MCP config: {e}")
        mcp_servers = {}

    if not mcp_servers:
        print("⚠️  No MCP servers configured - agent will operate without external tools")
        await run_agent_loop([])
        return

    # 2. Create MultiServerMCPClient (stateless by default as of 0.1.0+)
    mcp_client = MultiServerMCPClient(mcp_servers)

    # Dynamically fetch tools from the remote MCP server
    tools = []
    try:
        # The client queries the endpoint and converts MCP primitives to LangChain Tools
        tools = await mcp_client.get_tools()
        print(f"🛠️  Loaded {len(tools)} tools: {[t.name for t in tools]}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load MCP tools: {e}")
        print("⚠️  Continuing without MCP tools...")
        tools = []

    if not tools:
        print("⚠️  No tools available - agent will operate without external tools")

    await run_agent_loop(tools)


async def run_agent_loop(tools):
    """
    Run the agent chat loop with the provided tools.
    """
    # 3. Initialize Groq LLM
    # Using llama-3.3-70b-versatile model hosted on Groq Free tier
    # This model has 128k context window (vs 8k for openai/gpt-oss-20b)
    # Get your API key from: https://console.groq.com/keys
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to .env file.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.1,
        max_tokens=2048,
        model_kwargs={
            "top_p": 0.9,
        },
    )

    # 4. Create the Agent (ReAct Pattern) with error handling middleware
    agent = create_agent(
        llm,
        tools,
        system_prompt="""You are a helpful AI assistant specialized in LangGraph and LangChain.

When users ask about LangGraph or LangChain concepts, use the SearchDocsByLangChain tool with SPECIFIC technical queries.

IMPORTANT - Search Query Tips:
- Use specific API names like "StateGraph add_node", "add_edge", "compile"
- Use exact function/class names like "create_react_agent", "ToolNode", "MessageGraph"
- Combine concepts: "StateGraph nodes edges" instead of just "nodes"
- If a search fails, try more specific terms or API method names
- Examples of GOOD queries: "StateGraph add_node", "create_react_agent tools", "LangGraph streaming", "checkpointer memory"
- Examples of BAD queries: "nodes", "LangChain nodes", "how to use" (too vague)

When answering:
1. First search with specific technical terms from the user's question
2. If no results, try alternative specific queries (e.g., API names, class names)
3. If searches still fail after 2-3 attempts, provide your best knowledge
4. Include code examples when relevant
5. Be concise and focus on answering the user's specific question""",
        middleware=[ToolErrorHandlerMiddleware()],  # Add error handling middleware
    )

    print("\n🤖 Agent is ready! (Type 'quit' to exit)")
    print("-" * 50)

    # 5. Chat Loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        try:
            # Stream the response
            print("Agent: ", end="", flush=True)
            inputs = {"messages": [HumanMessage(content=user_input)]}

            response_content = ""
            async for chunk in agent.astream(inputs, stream_mode="values"):
                # Extract the last message from the chunk
                if "messages" in chunk and len(chunk["messages"]) > 0:
                    message = chunk["messages"][-1]
                    if message.type == "ai" and message.content:
                        # Only print if it's new content
                        if message.content != response_content:
                            new_content = message.content[len(response_content):]
                            print(new_content, end="", flush=True)
                            response_content = message.content

            # Print final result
            if not response_content:
                print("(No response generated)")
            print()
            print("-" * 50)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(run_agent_with_mcp())
