"""
Streamlit make_chain + RAG demo app

설치:
    pip install streamlit openai

실행:
    streamlit run make_chain_streamlit_app.py

특징:
- 웹페이지에서 OpenAI API key 입력
- 에이전트(system prompt)와 chain step 직접 편집
- 첨부 파일 업로드 후 OpenAI File Search 기반 RAG 사용
- 최종 상호작용은 챗봇 UI로 진행
"""

from __future__ import annotations

import hashlib
import io
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st
from openai import OpenAI


# -----------------------------
# 기본 프롬프트: 원래 노트북 예시 기반
# -----------------------------
DEFAULT_GENERATOR_PROMPT = """
너는 마케팅 전문가야. 내가 고객 정보와, 내가 판매하고 싶은 물건을 적어주면, 그 고객이 클릭할만한 마케팅 문구를 만들어줘.
마케팅 문구는 15단어 이하, 2문장 이하로 만들어 줘.
다른 사족을 붙이지 말고, 간단하게 마케팅 문구만 만들어 줘.
""".strip()

DEFAULT_EVALUATOR_PROMPT = """
너는 마케팅 전문가야. 내가 고객 정보와, 내가 판매하고 싶은 물건, 그리고 그들로 만든 마케팅 문구를 만들면, 그 문구를 참신성, 안전성, 효과성 차원으로 10점 척도로 평가해줘.
다른 사족을 붙이지 말고, 참신성: n점, 이유\n안전성: n점, 이유\n효과성: n점, 이유 와 같은 포맷으로 적어줘.
나는 무조건적인 칭찬이 아니라 객관적이고 전문적인 평가를 원해.
고객이나 아이템에 대한 평가는 하지 말고, 마케팅 문구 그 자체에 대한 평가를 부탁해.
""".strip()

DEFAULT_CONTEXT_PROMPT = """
고객 정보: 20대 여성, 1인 가구, 최근에 전자렌지를 구매.
판매하고 싶은 물건: 비비고 깻잎 만두.
""".strip()

RAG_INSTRUCTION = """
업로드된 파일과 관련된 질문이거나, 파일 내용이 답변에 도움이 될 수 있으면 file_search 도구로 근거를 확인해라.
파일에 없는 내용은 파일에 있다고 단정하지 말고, 추론과 파일 근거를 구분해서 답해라.
""".strip()


# -----------------------------
# Session state 초기화
# -----------------------------
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def init_session_state() -> None:
    if "agents" not in st.session_state:
        agent_1 = new_id("agent")
        agent_2 = new_id("agent")
        st.session_state.agents = [
            {
                "id": agent_1,
                "name": "마케팅 문구 생성 에이전트",
                "system_prompt": DEFAULT_GENERATOR_PROMPT,
            },
            {
                "id": agent_2,
                "name": "마케팅 문구 검수 에이전트",
                "system_prompt": DEFAULT_EVALUATOR_PROMPT,
            },
        ]

        st.session_state.chain_steps = [
            {"id": new_id("step"), "kind": "agent", "agent_id": agent_1, "prompt": ""},
            {"id": new_id("step"), "kind": "prompt", "agent_id": "", "prompt": DEFAULT_CONTEXT_PROMPT},
            {"id": new_id("step"), "kind": "agent", "agent_id": agent_2, "prompt": ""},
        ]

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store_id", "")
    st.session_state.setdefault("uploaded_file_records", {})
    st.session_state.setdefault("last_trace", [])


# -----------------------------
# OpenAI / RAG 유틸리티
# -----------------------------
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def get_status(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("status")
    return getattr(obj, "status", None)


def uploaded_file_key(uploaded_file: Any) -> str:
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"{uploaded_file.name}:{len(data)}:{digest}"


def ensure_vector_store(client: OpenAI) -> str:
    if st.session_state.vector_store_id:
        return st.session_state.vector_store_id

    vector_store = client.vector_stores.create(
        name=f"streamlit-make-chain-rag-{uuid.uuid4().hex[:8]}"
    )
    st.session_state.vector_store_id = vector_store.id
    return vector_store.id


def upload_files_to_vector_store(
    client: OpenAI,
    uploaded_files: list[Any],
    poll_timeout_seconds: int = 120,
) -> None:
    """Streamlit uploaded files를 OpenAI File API + Vector Store에 업로드한다."""
    if not uploaded_files:
        return

    vector_store_id = ensure_vector_store(client)

    for uploaded_file in uploaded_files:
        key = uploaded_file_key(uploaded_file)
        if key in st.session_state.uploaded_file_records:
            continue

        with st.spinner(f"'{uploaded_file.name}' 파일을 RAG용 vector store에 업로드하는 중..."):
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            openai_file = client.files.create(
                file=(uploaded_file.name, file_bytes),
                purpose="assistants",
            )

            vector_file = client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=openai_file.id,
            )

            start = time.time()
            status = get_status(vector_file)
            while status not in {"completed", "failed", "cancelled"}:
                if time.time() - start > poll_timeout_seconds:
                    raise TimeoutError(
                        f"'{uploaded_file.name}' 파일 인덱싱이 {poll_timeout_seconds}초 안에 끝나지 않았습니다. "
                        "잠시 뒤 다시 질문하거나 더 작은 파일로 테스트하세요."
                    )

                time.sleep(1)
                vector_file = client.vector_stores.files.retrieve(
                    vector_store_id=vector_store_id,
                    file_id=openai_file.id,
                )
                status = get_status(vector_file)

            if status != "completed":
                raise RuntimeError(f"'{uploaded_file.name}' 파일 인덱싱 실패: status={status}")

            st.session_state.uploaded_file_records[key] = {
                "name": uploaded_file.name,
                "openai_file_id": openai_file.id,
                "vector_store_id": vector_store_id,
                "status": status,
            }


def response(
    *,
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    vector_store_id: str | None = None,
    use_rag: bool = True,
    max_num_results: int = 5,
    max_output_tokens: int | None = None,
) -> str:
    """원래 노트북의 response(system, user)를 Responses API 버전으로 확장한 함수."""
    instructions = system.strip()
    tools = []

    if use_rag and vector_store_id:
        instructions = f"{instructions}\n\n{RAG_INSTRUCTION}".strip()
        tools = [
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": max_num_results,
            }
        ]

    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": user}],
    }

    if tools:
        kwargs["tools"] = tools
        # file_search 결과를 응답 객체에 포함한다. 화면에는 output_text만 노출한다.
        kwargs["include"] = ["file_search_call.results"]

    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens

    result = client.responses.create(**kwargs)
    return result.output_text or ""


def make_agent(
    *,
    name: str,
    system_prompt: str,
    client: OpenAI,
    model: str,
    vector_store_id: str | None,
    use_rag: bool,
    max_num_results: int,
    max_output_tokens: int | None,
) -> Callable[[str], str]:
    """system prompt를 받아 LLM agent 함수를 만든다."""

    def llm_agent(user: str) -> str:
        return response(
            client=client,
            model=model,
            system=system_prompt,
            user=user,
            vector_store_id=vector_store_id,
            use_rag=use_rag,
            max_num_results=max_num_results,
            max_output_tokens=max_output_tokens,
        )

    llm_agent.__name__ = name
    return llm_agent


def make_chain(*args: Any) -> Callable[[str], str]:
    """원래 노트북의 make_chain 동작을 그대로 유지한 함수."""

    def chain(user_input: str) -> str:
        current = user_input
        for step in args:
            if callable(step):
                current = step(current)
            elif isinstance(step, str):
                current = step + "\n" + current
        return current

    return chain


@dataclass
class TraceItem:
    step_no: int
    step_type: str
    name: str
    input_text: str
    output_text: str


def run_chain_with_trace(user_input: str, steps: list[Any]) -> tuple[str, list[TraceItem]]:
    """UI에서 중간 결과를 보기 위한 trace 지원 실행 함수."""
    current = user_input
    trace: list[TraceItem] = []

    for i, step in enumerate(steps, start=1):
        before = current

        if callable(step):
            step_name = getattr(step, "__name__", "LLM agent")
            current = step(current)
            trace_type = "agent"
            trace_name = step_name
        elif isinstance(step, str):
            current = step + "\n" + current
            trace_type = "prompt"
            trace_name = "프롬프트 추가"
        else:
            continue

        trace.append(
            TraceItem(
                step_no=i,
                step_type=trace_type,
                name=trace_name,
                input_text=before,
                output_text=current,
            )
        )

    return current, trace


# -----------------------------
# UI 편집기
# -----------------------------
def render_agent_editor() -> None:
    st.subheader("에이전트 편집")

    delete_agent_id = None
    for index, agent in enumerate(st.session_state.agents):
        with st.expander(f"Agent {index + 1}: {agent['name']}", expanded=index < 2):
            agent["name"] = st.text_input(
                "에이전트 이름",
                value=agent["name"],
                key=f"agent_name_{agent['id']}",
            )
            agent["system_prompt"] = st.text_area(
                "System prompt",
                value=agent["system_prompt"],
                height=180,
                key=f"agent_prompt_{agent['id']}",
            )
            if len(st.session_state.agents) > 1:
                if st.button("이 에이전트 삭제", key=f"delete_agent_{agent['id']}"):
                    delete_agent_id = agent["id"]

    if delete_agent_id:
        st.session_state.agents = [
            agent for agent in st.session_state.agents if agent["id"] != delete_agent_id
        ]
        st.session_state.chain_steps = [
            step
            for step in st.session_state.chain_steps
            if step.get("agent_id") != delete_agent_id
        ]
        st.rerun()

    if st.button("+ 에이전트 추가"):
        st.session_state.agents.append(
            {
                "id": new_id("agent"),
                "name": f"새 에이전트 {len(st.session_state.agents) + 1}",
                "system_prompt": "너는 유용한 AI 어시스턴트야. 사용자의 요청에 정확하고 간결하게 답해줘.",
            }
        )
        st.rerun()


def render_chain_editor() -> None:
    st.subheader("체인 단계 편집")
    st.caption("agent 단계는 LLM을 호출하고, prompt 단계는 현재 입력 앞에 고정 문구를 붙입니다.")

    agent_options = {agent["id"]: agent["name"] for agent in st.session_state.agents}
    agent_ids = list(agent_options.keys())
    delete_step_id = None

    for index, step in enumerate(st.session_state.chain_steps):
        with st.expander(f"Step {index + 1}: {step.get('kind', 'agent')}", expanded=True):
            kind = st.selectbox(
                "단계 종류",
                options=["agent", "prompt"],
                index=0 if step.get("kind") == "agent" else 1,
                key=f"step_kind_{step['id']}",
            )
            step["kind"] = kind

            if kind == "agent":
                if not agent_ids:
                    st.warning("사용 가능한 에이전트가 없습니다.")
                    step["agent_id"] = ""
                else:
                    current_agent_id = step.get("agent_id") if step.get("agent_id") in agent_ids else agent_ids[0]
                    selected_agent_id = st.selectbox(
                        "사용할 에이전트",
                        options=agent_ids,
                        format_func=lambda agent_id: agent_options.get(agent_id, agent_id),
                        index=agent_ids.index(current_agent_id),
                        key=f"step_agent_{step['id']}",
                    )
                    step["agent_id"] = selected_agent_id
                    step["prompt"] = step.get("prompt", "")
            else:
                step["prompt"] = st.text_area(
                    "추가할 프롬프트",
                    value=step.get("prompt", ""),
                    height=140,
                    key=f"step_prompt_{step['id']}",
                )
                step["agent_id"] = step.get("agent_id", "")

            if st.button("이 단계 삭제", key=f"delete_step_{step['id']}"):
                delete_step_id = step["id"]

    if delete_step_id:
        st.session_state.chain_steps = [
            step for step in st.session_state.chain_steps if step["id"] != delete_step_id
        ]
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("+ Agent 단계 추가"):
            default_agent_id = agent_ids[0] if agent_ids else ""
            st.session_state.chain_steps.append(
                {"id": new_id("step"), "kind": "agent", "agent_id": default_agent_id, "prompt": ""}
            )
            st.rerun()
    with col2:
        if st.button("+ Prompt 단계 추가"):
            st.session_state.chain_steps.append(
                {"id": new_id("step"), "kind": "prompt", "agent_id": "", "prompt": ""}
            )
            st.rerun()
    with col3:
        if st.button("체인 초기화"):
            st.session_state.pop("agents", None)
            st.session_state.pop("chain_steps", None)
            init_session_state()
            st.rerun()


def build_runtime_steps(
    *,
    client: OpenAI,
    model: str,
    vector_store_id: str | None,
    use_rag: bool,
    max_num_results: int,
    max_output_tokens: int | None,
) -> list[Any]:
    agents_by_id = {agent["id"]: agent for agent in st.session_state.agents}
    runtime_agents: dict[str, Callable[[str], str]] = {}

    for agent in st.session_state.agents:
        runtime_agents[agent["id"]] = make_agent(
            name=agent["name"],
            system_prompt=agent["system_prompt"],
            client=client,
            model=model,
            vector_store_id=vector_store_id,
            use_rag=use_rag,
            max_num_results=max_num_results,
            max_output_tokens=max_output_tokens,
        )

    steps: list[Any] = []
    for step in st.session_state.chain_steps:
        if step.get("kind") == "agent":
            agent_id = step.get("agent_id")
            if agent_id in agents_by_id:
                steps.append(runtime_agents[agent_id])
        elif step.get("kind") == "prompt":
            prompt_text = step.get("prompt", "").strip()
            if prompt_text:
                steps.append(prompt_text)

    return steps


def format_recent_history(messages: list[dict[str, Any]], max_turns: int) -> str:
    if max_turns <= 0 or not messages:
        return ""

    recent = messages[-max_turns * 2 :]
    lines = []
    for msg in recent:
        role = "사용자" if msg.get("role") == "user" else "어시스턴트"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def render_trace(trace: list[TraceItem]) -> None:
    if not trace:
        return

    with st.expander("체인 중간 결과 보기", expanded=False):
        for item in trace:
            st.markdown(f"**Step {item.step_no} · {item.step_type} · {item.name}**")
            st.caption("입력")
            st.code(item.input_text, language="text")
            st.caption("출력")
            st.code(item.output_text, language="text")


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="make_chain Streamlit 시뮬레이터", page_icon="🔗", layout="wide")
    init_session_state()

    st.title("🔗 make_chain Streamlit 시뮬레이터")
    st.caption("에이전트와 프롬프트 체인을 편집하고, 업로드 파일을 RAG로 참조하면서 챗봇 형태로 테스트합니다.")

    with st.sidebar:
        st.header("실행 설정")
        api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", help="키는 이 앱 세션에서만 사용합니다.")
        model = st.text_input("모델", value="gpt-4o-mini", help="예: gpt-4o-mini, gpt-4.1-mini, gpt-5.5 등 계정에서 사용 가능한 모델")
        use_rag = st.checkbox("업로드 파일 RAG 사용", value=True)
        max_num_results = st.slider("RAG 검색 결과 수", min_value=1, max_value=20, value=5)
        max_output_tokens_enabled = st.checkbox("출력 토큰 제한 사용", value=False)
        max_output_tokens = None
        if max_output_tokens_enabled:
            max_output_tokens = st.number_input("max_output_tokens", min_value=128, max_value=8192, value=1200, step=128)

        st.divider()
        include_history = st.checkbox("최근 대화 문맥 포함", value=True)
        max_history_turns = st.slider("포함할 최근 대화 턴", min_value=0, max_value=10, value=3)

        st.divider()
        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.session_state.last_trace = []
            st.rerun()

        if st.button("업로드/RAG 상태 초기화"):
            st.session_state.vector_store_id = ""
            st.session_state.uploaded_file_records = {}
            st.rerun()

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        st.header("설정 패널")
        render_agent_editor()
        st.divider()
        render_chain_editor()
        st.divider()

        st.subheader("RAG 파일 업로드")
        uploaded_files = st.file_uploader(
            "참조할 파일을 업로드하세요",
            accept_multiple_files=True,
            help="PDF, TXT, Markdown, CSV, JSON, DOCX, PPTX 등 OpenAI File Search가 지원하는 파일을 사용하세요.",
        )

        if uploaded_files:
            if not api_key:
                st.info("파일을 RAG에 연결하려면 먼저 sidebar에 OpenAI API key를 입력하세요.")
            else:
                try:
                    client = get_client(api_key)
                    upload_files_to_vector_store(client, uploaded_files)
                    st.success("업로드된 파일이 RAG vector store에 연결되었습니다.")
                except Exception as exc:
                    st.error(f"파일 업로드 또는 인덱싱 중 오류가 발생했습니다: {exc}")

        if st.session_state.uploaded_file_records:
            st.markdown("**RAG에 연결된 파일**")
            for record in st.session_state.uploaded_file_records.values():
                st.write(f"- {record['name']} · {record['status']}")
            st.caption(f"Vector store ID: {st.session_state.vector_store_id}")
        else:
            st.caption("아직 RAG에 연결된 파일이 없습니다.")

    with right:
        st.header("챗봇 테스트")
        st.caption("아래 입력은 현재 편집된 chain을 통과합니다. 최종 assistant 메시지는 chain의 마지막 출력입니다.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("trace"):
                    render_trace(message["trace"])

        user_message = st.chat_input("질문이나 테스트 입력을 적어주세요")

        if user_message:
            st.session_state.messages.append({"role": "user", "content": user_message})
            with st.chat_message("user"):
                st.markdown(user_message)

            if not api_key:
                assistant_message = "OpenAI API key를 sidebar에 입력한 뒤 다시 시도하세요."
                trace = []
            else:
                try:
                    client = get_client(api_key)
                    vector_store_id = st.session_state.vector_store_id if use_rag else None
                    runtime_steps = build_runtime_steps(
                        client=client,
                        model=model.strip(),
                        vector_store_id=vector_store_id,
                        use_rag=use_rag,
                        max_num_results=max_num_results,
                        max_output_tokens=int(max_output_tokens) if max_output_tokens else None,
                    )

                    if not runtime_steps:
                        raise ValueError("실행할 chain step이 없습니다. 왼쪽에서 Agent 또는 Prompt 단계를 추가하세요.")

                    history_text = format_recent_history(st.session_state.messages[:-1], max_history_turns) if include_history else ""
                    effective_input = user_message
                    if history_text:
                        effective_input = f"[최근 대화]\n{history_text}\n\n[현재 사용자 입력]\n{user_message}"

                    with st.spinner("chain 실행 중..."):
                        # 원래 노트북과 같은 단순 실행 함수도 생성해 둔다.
                        # 필요하면 아래 chain(user_message)로 trace 없이 실행할 수 있다.
                        _chain = make_chain(*runtime_steps)
                        assistant_message, trace = run_chain_with_trace(effective_input, runtime_steps)

                    st.session_state.last_trace = trace
                except Exception as exc:
                    assistant_message = f"오류가 발생했습니다: {exc}"
                    trace = []

            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_message, "trace": trace}
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_message)
                render_trace(trace)


if __name__ == "__main__":
    main()
