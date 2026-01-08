import json
import requests
import gradio as gr
from typing import List, Dict, Any, Tuple

from langchain_ollama import ChatOllama

#MODEL = "llama3.2:latest"
MODEL = "qwen2.5:0.5b-instruct"
API_URL = "https://pzvpngd6ij.execute-api.ap-southeast-5.amazonaws.com/dev/create"
#API_URL = "http://127.0.0.1:8000/receive"

llm = ChatOllama(model=MODEL)

INTENT_SYSTEM = """
You are a routing assistant. Decide what the user wants.

Return ONLY valid JSON with exactly this key:
{"intent": "info" | "create_instance" | "other"}

Rules:
- intent="info" if the user is asking to explain Odoo/ERP or general "what is/how does" questions.
- intent="create_instance" ONLY if the user clearly wants to create/spawn/start/setup an Odoo simulator instance (or says equivalent).
- otherwise "other".
Do NOT add extra keys. Do NOT wrap in markdown.
"""

INTAKE_SYSTEM = """
You are an information extraction assistant.

Extract the following fields from the user's message IF present:
- company_name: string
- company_background: string (short summary of what they do)
- customer_name: string (who to create the simulator instance for)

Return ONLY valid JSON with exactly these keys:
{
  "company_name": "",
  "company_background": "",
  "customer_name": ""
}

Rules:
- If a field is not present, keep it empty string.
- Do NOT add extra keys.
- Do NOT wrap in markdown.
"""

HELP_SYSTEM = """
You are a helpful assistant.
Explain clearly, concisely, and conversationally.
If the user asks about Odoo, explain what it is, what it’s used for, and give a short example.
If relevant, you may add one gentle suggestion that you can create an Odoo simulator instance if they want.
"""

def safe_json_extract(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def detect_intent(user_text: str) -> str:
    resp = llm.invoke([
        ("system", INTENT_SYSTEM),
        ("human", user_text),
    ]).content.strip()
    data = safe_json_extract(resp)
    intent = (data.get("intent") or "other").strip()
    return intent if intent in {"info", "create_instance", "other"} else "other"


def extract_intake(user_text: str) -> Dict[str, str]:
    resp = llm.invoke([
        ("system", INTAKE_SYSTEM),
        ("human", user_text),
    ]).content.strip()
    data = safe_json_extract(resp)
    return {
        "company_name": (data.get("company_name") or "").strip(),
        "company_background": (data.get("company_background") or "").strip(),
        "customer_name": (data.get("customer_name") or "").strip(),
    }

def merge_intake(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(old or {"company_name": "", "company_background": "", "customer_name": ""})
    for k in ["company_name", "company_background", "customer_name"]:
        if new.get(k):
            merged[k] = new[k]
    return merged


def next_question(intake: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns: (question_text, awaiting_field)
    awaiting_field is used to deterministically capture short answers like "talysqi".
    """
    if not intake.get("company_name"):
        return "Sure. What’s your company name?", "company_name"
    if not intake.get("company_background"):
        return "Nice. What does your company do (1–2 sentences)?", "company_background"
    if not intake.get("customer_name"):
        return "Great. Who should I create the Odoo simulator instance for? (customer name)", "customer_name"
    return "", ""


def send_customer_name(customer_name: str):
    payload = {"customer_name": customer_name}  # only pass customer_name
    r = requests.post(
        API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def llm_help_reply(user_text: str) -> str:
    return llm.invoke([
        ("system", HELP_SYSTEM),
        ("human", user_text),
    ]).content


def llm_chat_reply(messages: List[Dict]) -> str:
    lc_messages = []
    for m in messages:
        role = "human" if m["role"] == "user" else "assistant"
        lc_messages.append((role, m["content"]))
    return llm.invoke(lc_messages).content


def on_submit(
    user_input: str,
    messages: List[Dict],
    stage: str,
    intake_state: Dict[str, Any],
    awaiting_field: str,
):
    messages = messages or []
    user_input = (user_input or "").strip()
    intake_state = intake_state or {"company_name": "", "company_background": "", "customer_name": ""}
    awaiting_field = (awaiting_field or "").strip()

    if not user_input:
        return messages, stage, intake_state, awaiting_field

    messages.append({"role": "user", "content": user_input})

    # ---------- ROUTING (only when not already in confirm/chat) ----------
    if stage in {"idle", "intake"}:
        # If we asked a direct question last turn, trust the user's next reply
        # (prevents repeated "What's your company name?" loops)
        if stage == "intake" and awaiting_field:
            intake_state[awaiting_field] = user_input.strip()
            awaiting_field = ""
        else:
            # Detect intent only when we're not already collecting a specific field
            try:
                intent = detect_intent(user_input)
            except Exception as e:
                messages.append({"role": "assistant", "content": f"I had trouble understanding that. ({e})"})
                return messages, stage, intake_state, awaiting_field

            # Info question: answer normally, stay idle
            if intent == "info":
                try:
                    reply = llm_help_reply(user_input)
                except Exception as e:
                    reply = f"(error) {e}"
                messages.append({"role": "assistant", "content": reply})
                return messages, "idle", intake_state, awaiting_field

            # If they want to create an instance, start intake
            if intent == "create_instance":
                stage = "intake"

            # If we're in intake, also try extracting any fields they included
            if stage == "intake":
                try:
                    extracted = extract_intake(user_input)
                    intake_state = merge_intake(intake_state, extracted)
                except Exception as e:
                    messages.append({"role": "assistant", "content": f"I had trouble extracting the details. ({e})"})
                    return messages, stage, intake_state, awaiting_field

        # If we are in intake mode, ask what's missing
        if stage == "intake":
            q, awaiting_field = next_question(intake_state)
            if q:
                messages.append({"role": "assistant", "content": q})
                return messages, stage, intake_state, awaiting_field

            # All fields present -> confirm
            stage = "confirm"
            messages.append({
                "role": "assistant",
                "content": (
                    "Got it — here’s what I’ll use:\n"
                    f'- Company: {intake_state["company_name"]}\n'
                    f'- Background: {intake_state["company_background"]}\n'
                    f'- Customer name (instance for): {intake_state["customer_name"]}\n\n'
                    "Proceed to create the simulator instance? (yes/no)"
                )
            })
            return messages, stage, intake_state, awaiting_field

        # Otherwise, normal idle chat
        try:
            reply = llm_chat_reply(messages)
        except Exception as e:
            reply = f"(error) {e}"
        messages.append({"role": "assistant", "content": reply})
        return messages, "idle", intake_state, awaiting_field

    # ---------- CONFIRM ----------
    if stage == "confirm":
        if user_input.lower().startswith("y"):
            try:
                result = send_customer_name(intake_state["customer_name"])
                if result.get("success") is True:
                    messages.append({
                        "role": "assistant",
                        "content": (
                            f"Done — simulator instance created for **{intake_state['customer_name']}** "
                            f"(ID: {result.get('id')}). You can continue chatting."
                        )
                    })
                    stage = "chat"
                else:
                    messages.append({"role": "assistant", "content": f"API responded with: {result}"})
            except Exception as e:
                messages.append({"role": "assistant", "content": f"Request failed: {e}"})
            return messages, stage, intake_state, awaiting_field
        else:
            stage = "intake"
            messages.append({
                "role": "assistant",
                "content": "No worries — just reply with the corrected company name, background, or customer name."
            })
            return messages, stage, intake_state, awaiting_field

    # ---------- CHAT ----------
    if stage == "chat":
        try:
            reply = llm_chat_reply(messages)
        except Exception as e:
            reply = f"(error) {e}"
        messages.append({"role": "assistant", "content": reply})
        return messages, stage, intake_state, awaiting_field

    # fallback
    messages.append({"role": "assistant", "content": "I got a bit lost — let’s continue."})
    return messages, "idle", intake_state, ""


with gr.Blocks(title="Odoo Simulator Chat") as demo:
    gr.Markdown("## Odoo Simulator Chat")

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(
        placeholder="Ask anything (e.g., 'what is odoo' or 'create odoo simulator instance')",
        autofocus=True
    )

    state_stage = gr.State("idle")
    state_intake = gr.State({"company_name": "", "company_background": "", "customer_name": ""})
    state_awaiting = gr.State("")  # "", "company_name", "company_background", "customer_name"

    demo.load(
        lambda: (
            [{"role": "assistant", "content": "Hi! Ask me anything. If you want to create an Odoo simulator instance, just tell me."}],
            "idle",
            {"company_name": "", "company_background": "", "customer_name": ""},
            "",
        ),
        None,
        [chatbot, state_stage, state_intake, state_awaiting],
    )

    msg.submit(
        on_submit,
        inputs=[msg, chatbot, state_stage, state_intake, state_awaiting],
        outputs=[chatbot, state_stage, state_intake, state_awaiting],
    ).then(lambda: "", None, msg)

#demo.launch()
demo.launch(server_name="127.0.0.1",server_port=7860)
