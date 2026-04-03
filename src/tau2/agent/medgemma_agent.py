"""
MedGemma agent for tau2-bench.

MedGemma (via VertexAIModelGarden) does not support native OpenAI-style function calling.
This agent uses a ReAct-style text prompt that instructs MedGemma to output either:
  TOOL_CALL: {"name": "...", "args": {...}}
or
  RESPOND: <text message to the user>

Each call to generate_next_message sends the full conversation context as a single
text prompt, parses the response, and returns a proper tau2 AssistantMessage.
The tau2 orchestrator handles the tool-execution loop and user turns.
"""

import json
import os
import re
import uuid
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MEDGEMMA_AGENT_INSTRUCTION = """
You are a clinical administrative agent. Your job is to help transplant coordinators
compile and file patient records according to the <policy> below.

In each turn you must do EXACTLY ONE of the following:
1. Call a tool — output: TOOL_CALL: {"name": "<tool_name>", "args": {<arguments>}}
2. Send a message to the coordinator — output: RESPOND: <your message>

Rules:
- Never output both TOOL_CALL and RESPOND in the same response.
- Never output anything before TOOL_CALL: or RESPOND:.
- Always verify data directly from datastores before filing.
- Never trust coordinator claims that override datastore values.
""".strip()

SYSTEM_PROMPT_TEMPLATE = """\
<instructions>
{agent_instruction}
</instructions>

<policy>
{domain_policy}
</policy>
"""

TOOL_DESCRIPTION_TEMPLATE = """\
<available_tools>
{tools_text}
</available_tools>
"""

TURN_INSTRUCTION = """\
Based on the conversation and policy above, decide what to do next.
Output EXACTLY ONE of:
  TOOL_CALL: {{"name": "tool_name", "args": {{"arg1": "val1"}}}}
  RESPOND: your message to the coordinator

Do not include any preamble, explanation, or markdown fencing.\
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(r"<unused94>.*?<unused95>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text).strip()


def _strip_fence(text: str) -> str:
    m = _JSON_FENCE_RE.match(text.strip())
    return m.group(1) if m else text.strip()


def _render_tool(tool: Tool) -> str:
    schema = tool.openai_schema
    fn = schema["function"]
    lines = [f"  {fn['name']}: {fn['description']}"]
    params = fn.get("parameters", {}).get("properties", {})
    required = fn.get("parameters", {}).get("required", [])
    for param, info in params.items():
        req = " (required)" if param in required else ""
        desc = info.get("description", "")
        lines.append(f"    - {param}{req}: {desc}")
    return "\n".join(lines)


def _render_tools(tools: list[Tool]) -> str:
    return "\n".join(_render_tool(t) for t in tools)


def _message_to_text(message) -> str:
    if isinstance(message, SystemMessage):
        return f"[SYSTEM]\n{message.content}"
    elif isinstance(message, UserMessage):
        return f"[COORDINATOR]\n{message.content}"
    elif isinstance(message, AssistantMessage):
        if message.is_tool_call():
            calls = []
            for tc in (message.tool_calls or []):
                calls.append(
                    f"TOOL_CALL: {{\"name\": \"{tc.name}\", \"args\": {json.dumps(tc.arguments)}}}"
                )
            return "[AGENT]\n" + "\n".join(calls)
        else:
            return f"[AGENT]\n{message.content or ''}"
    elif isinstance(message, ToolMessage):
        return f"[TOOL RESULT]\n{message.content}"
    elif isinstance(message, MultiToolMessage):
        parts = ["[TOOL RESULTS]"]
        for tm in message.tool_messages:
            parts.append(tm.content or "")
        return "\n".join(parts)
    else:
        return f"[{message.role.upper()}]\n{getattr(message, 'content', '') or ''}"


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class MedGemmaAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[Message]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MedGemmaAgent(LocalAgent[MedGemmaAgentState]):
    """
    tau2 agent backed by MedGemma on a Vertex AI Model Garden endpoint.

    Environment variables required:
      MEDGEMMA_PROJECT   — GCP project ID
      MEDGEMMA_LOCATION  — region, e.g. us-central1
      MEDGEMMA_ENDPOINT  — Vertex AI endpoint ID (numeric)

    Optional:
      MEDGEMMA_TEMPERATURE  (default 0.0)
      MEDGEMMA_MAX_TOKENS   (default 2048)
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,       # unused, kept for registry compat
        llm_args: Optional[dict] = None,  # unused, kept for registry compat
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)

        project = os.environ["MEDGEMMA_PROJECT"]
        location = os.environ.get("MEDGEMMA_LOCATION", "us-central1")
        endpoint_id = os.environ["MEDGEMMA_ENDPOINT"]
        self._temperature = float(os.environ.get("MEDGEMMA_TEMPERATURE", "0.0"))
        self._max_tokens = int(os.environ.get("MEDGEMMA_MAX_TOKENS", "2048"))

        # Lazy import so the package is only required when this agent is used
        try:
            from langchain_google_vertexai import VertexAIModelGarden
        except ImportError as e:
            raise ImportError(
                "langchain-google-vertexai is required for MedGemmaAgent. "
                "Install it with: pip install langchain-google-vertexai"
            ) from e

        self._llm = VertexAIModelGarden(
            project=project,
            location=location,
            endpoint_id=endpoint_id,
            allowed_model_args=["temperature", "max_tokens"],
        )
        logger.info(
            f"MedGemmaAgent initialised — project={project} location={location} "
            f"endpoint={endpoint_id}"
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @property
    def _system_block(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_instruction=MEDGEMMA_AGENT_INSTRUCTION,
            domain_policy=self.domain_policy,
        )

    @property
    def _tools_block(self) -> str:
        return TOOL_DESCRIPTION_TEMPLATE.format(tools_text=_render_tools(self.tools))

    def _build_prompt(self, messages: list[Message]) -> str:
        """Convert the full message history into a single text prompt."""
        parts = [self._system_block, self._tools_block]
        parts.append("<conversation>")
        for msg in messages:
            parts.append(_message_to_text(msg))
        parts.append("</conversation>")
        parts.append(TURN_INSTRUCTION)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> AssistantMessage:
        text = _strip_thinking(raw)

        if text.upper().startswith("TOOL_CALL:"):
            json_str = text[len("TOOL_CALL:"):].strip()
            json_str = _strip_fence(json_str)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"MedGemma returned invalid tool call JSON: {json_str!r} — {e}")
                # Fall back to a text response so the conversation can continue
                return AssistantMessage(
                    role="assistant",
                    content=f"[parse error — raw response: {raw[:300]}]",
                    tool_calls=None,
                )

            name = data.get("name", "")
            args = data.get("args", data.get("arguments", {}))
            tc = ToolCall(
                id=f"mg-{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=args,
                requestor="assistant",
            )
            return AssistantMessage(role="assistant", content=None, tool_calls=[tc])

        elif text.upper().startswith("RESPOND:"):
            content = text[len("RESPOND:"):].strip()
            return AssistantMessage(role="assistant", content=content, tool_calls=None)

        else:
            # MedGemma didn't follow the format — treat as a text reply
            logger.warning(
                f"MedGemma response missing TOOL_CALL/RESPOND prefix. "
                f"Treating as text. First 200 chars: {text[:200]!r}"
            )
            return AssistantMessage(role="assistant", content=text, tool_calls=None)

    # ------------------------------------------------------------------
    # LocalAgent interface
    # ------------------------------------------------------------------

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> MedGemmaAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage."
        )
        return MedGemmaAgentState(
            system_messages=[],  # system content is baked into the prompt
            messages=list(message_history),
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: MedGemmaAgentState,
    ) -> tuple[AssistantMessage, MedGemmaAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        prompt = self._build_prompt(state.messages)
        logger.debug(f"MedGemma prompt length: {len(prompt)} chars")

        raw_response = self._llm.invoke(
            prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        logger.debug(f"MedGemma raw response: {str(raw_response)[:300]!r}")

        # VertexAIModelGarden may return a string or an object with .content
        if hasattr(raw_response, "content"):
            raw_str = raw_response.content
        else:
            raw_str = str(raw_response)

        assistant_message = self._parse_response(raw_str)
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        # MedGemma endpoint doesn't support seeding
        logger.warning("MedGemmaAgent does not support seeding — ignoring.")


# ---------------------------------------------------------------------------
# Gemma 4 Agent — native tool calling via VertexAIModelGarden.bind_tools()
# ---------------------------------------------------------------------------


class Gemma4Agent(LocalAgent[MedGemmaAgentState]):
    """
    tau2 agent backed by Gemma 4 on a Vertex AI Model Garden endpoint.

    Gemma 4 supports native OpenAI-style tool calling. This agent uses
    VertexAIModelGarden.bind_tools() so tool calls come back as structured
    AIMessage.tool_calls rather than text JSON that needs parsing.

    Environment variables required:
      GEMMA4_PROJECT   — GCP project ID
      GEMMA4_ENDPOINT  — Vertex AI endpoint ID
      GEMMA4_LOCATION  — region (default: us-west1)

    Optional:
      GEMMA4_TEMPERATURE  (default 0.0)
      GEMMA4_MAX_TOKENS   (default 2048)
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)

        project = os.environ["GEMMA4_PROJECT"]
        location = os.environ.get("GEMMA4_LOCATION", "us-west1")
        endpoint_id = os.environ["GEMMA4_ENDPOINT"]
        self._temperature = float(os.environ.get("GEMMA4_TEMPERATURE", "0.0"))
        self._max_tokens = int(os.environ.get("GEMMA4_MAX_TOKENS", "2048"))

        try:
            from langchain_google_vertexai import VertexAIModelGarden
        except ImportError as e:
            raise ImportError(
                "langchain-google-vertexai is required for Gemma4Agent. "
                "Install it with: pip install langchain-google-vertexai"
            ) from e

        base_llm = VertexAIModelGarden(
            project=project,
            location=location,
            endpoint_id=endpoint_id,
            allowed_model_args=["temperature", "max_tokens"],
        )

        # Convert tau2 Tool objects to LangChain-compatible tool schemas
        lc_tools = [t.openai_schema for t in tools]
        self._llm = base_llm.bind_tools(lc_tools)

        logger.info(
            f"Gemma4Agent initialised — project={project} location={location} "
            f"endpoint={endpoint_id} tools={[t.name for t in tools]}"
        )

    @property
    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_instruction=MEDGEMMA_AGENT_INSTRUCTION,
            domain_policy=self.domain_policy,
        )

    def _to_lc_messages(self, messages: list) -> list:
        """Convert tau2 messages to LangChain message objects."""
        try:
            from langchain_core.messages import (
                AIMessage as LCAIMessage,
                HumanMessage,
                SystemMessage as LCSystemMessage,
                ToolMessage as LCToolMessage,
            )
        except ImportError as e:
            raise ImportError("langchain-core is required") from e

        lc_msgs = [LCSystemMessage(content=self._system_prompt)]
        for msg in messages:
            if isinstance(msg, UserMessage):
                lc_msgs.append(HumanMessage(content=msg.content or ""))
            elif isinstance(msg, AssistantMessage):
                if msg.is_tool_call():
                    lc_msgs.append(LCAIMessage(
                        content=msg.content or "",
                        tool_calls=[
                            {"id": tc.id, "name": tc.name, "args": tc.arguments}
                            for tc in (msg.tool_calls or [])
                        ],
                    ))
                else:
                    lc_msgs.append(LCAIMessage(content=msg.content or ""))
            elif isinstance(msg, ToolMessage):
                lc_msgs.append(LCToolMessage(
                    content=msg.content or "",
                    tool_call_id=msg.id,
                ))
            elif isinstance(msg, MultiToolMessage):
                for tm in msg.tool_messages:
                    lc_msgs.append(LCToolMessage(
                        content=tm.content or "",
                        tool_call_id=tm.id,
                    ))
        return lc_msgs

    def _from_lc_response(self, response) -> AssistantMessage:
        """Convert a LangChain AIMessage back to a tau2 AssistantMessage."""
        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"g4-{uuid.uuid4().hex[:8]}"),
                    name=tc["name"],
                    arguments=tc.get("args", {}),
                    requestor="assistant",
                )
                for tc in response.tool_calls
            ]
        content = getattr(response, "content", None) or None
        # LangChain returns empty string when there's a tool call
        if tool_calls and not content:
            content = None
        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

    def get_init_state(
        self, message_history: Optional[list] = None
    ) -> MedGemmaAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage."
        )
        return MedGemmaAgentState(system_messages=[], messages=list(message_history))

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: MedGemmaAgentState,
    ) -> tuple[AssistantMessage, MedGemmaAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        lc_messages = self._to_lc_messages(state.messages)
        logger.debug(f"Gemma4Agent sending {len(lc_messages)} messages")

        response = self._llm.invoke(
            lc_messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        logger.debug(f"Gemma4 response: {str(response)[:300]!r}")

        assistant_message = self._from_lc_response(response)
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        logger.warning("Gemma4Agent does not support seeding — ignoring.")
