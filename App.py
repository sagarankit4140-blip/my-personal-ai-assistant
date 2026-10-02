import streamlit as st
import re
import ast
import operator as op
from transformers import pipeline

# Optional internet search package:
# pip install ddgs
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="My Personal AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# AI MODEL
# -----------------------------
@st.cache_resource
def load_ai():
    return pipeline(
        "text-generation",
        model="distilgpt2"
    )

try:
    ai_generator = load_ai()
except Exception:
    ai_generator = None

# -----------------------------
# SAFE MATH CALCULATOR
# -----------------------------
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        # Avoid accidentally creating huge values
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")

def calculate(expr):
    try:
        cleaned = expr.strip()
        if not cleaned:
            return "Please enter a valid equation."
        if len(cleaned) > 200:
            return "Equation is too long."
        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval(tree)
        return f"Result: {result}"
    except Exception:
        return "Invalid Equation"

# -----------------------------
# INTERNET SEARCH
# -----------------------------
def web_search(query, max_results=5):
    if DDGS is None:
        return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception:
        return []

def build_web_context(results):
    parts = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        body = item.get("body", "")
        url = item.get("href", "")
        parts.append(
            f"{i}. {title}\n"
            f"Summary: {body}\n"
            f"URL: {url}"
        )
    return "\n\n".join(parts)

# -----------------------------
# APP HEADER
# -----------------------------
st.title("🤖 My Personal AI Assistant")
st.caption("AI Chat + Internet Search + Smart Math Calculator")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs([
    "💬 AI Chat",
    "🔢 Math Calculator"
])

# =====================================================
# AI CHAT
# =====================================================
with tab1:
    st.subheader("💬 Chat with AI")

    use_internet = st.toggle(
        "🌐 Search the Internet",
        value=True,
        help="When enabled, the assistant searches the web before answering."
    )

    if DDGS is None:
        st.warning(
            "Internet search is not installed. Run: "
            "`python -m pip install ddgs`"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.caption("🌐 Sources")
                for source in message["sources"]:
                    st.markdown(
                        f"- [{source['title']}]({source['url']})"
                    )

    user_msg = st.chat_input("Ask something...")

    if user_msg:
        st.session_state.messages.append({
            "role": "user",
            "content": user_msg
        })

        with st.chat_message("user"):
            st.write(user_msg)

        sources = []
        with st.chat_message("assistant"):
            if not ai_generator:
                st.error("AI model load nahi hua.")
            else:
                web_context = ""

                if use_internet:
                    with st.spinner("🌐 Searching the Internet..."):
                        results = web_search(user_msg, max_results=5)

                    if results:
                        web_context = build_web_context(results)
                        sources = [
                            {
                                "title": r.get("title", "Source"),
                                "url": r.get("href", "")
                            }
                            for r in results
                            if r.get("href")
                        ]
                    else:
                        st.info(
                            "Internet search se results nahi mile. "
                            "Main local AI se answer de raha hoon."
                        )

                # Keep the prompt small because distilgpt2 is a lightweight model.
                if web_context:
                    prompt = (
                        "Answer the user's question using the web information below. "
                        "Be concise and do not invent facts.\n\n"
                        f"WEB INFORMATION:\n{web_context}\n\n"
                        f"USER QUESTION: {user_msg}\n"
                        "ANSWER:"
                    )
                else:
                    prompt = f"Question: {user_msg}\nAnswer:"

                with st.spinner("🤔 Thinking..."):
                    try:
                        result = ai_generator(
                            prompt,
                            max_new_tokens=120,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=ai_generator.tokenizer.eos_token_id
                        )
                        answer = result[0]["generated_text"]
                        if answer.startswith(prompt):
                            answer = answer[len(prompt):].strip()
                        if not answer:
                            answer = "I couldn't generate an answer."
                    except Exception as e:
                        answer = f"AI error: {e}"

                st.write(answer)

                if sources:
                    st.caption("🌐 Web Sources")
                    for source in sources:
                        st.markdown(
                            f"- [{source['title']}]({source['url']})"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

# =====================================================
# MATH CALCULATOR
# =====================================================
with tab2:
    st.subheader("🔢 Smart Math Calculator")

    math_msg = st.text_input(
        "Enter Math Equation",
        placeholder="Example: 50*12+10"
    )

    if st.button("🧮 Calculate"):
        result = calculate(math_msg)

        if result.startswith("Result"):
            st.success(result)
        else:
            st.error(result)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Assistant")
    st.write("Internet search can be turned on/off from the AI Chat tab.")
    st.divider()
    st.caption("Powered by Streamlit + Transformers + DDGS")
