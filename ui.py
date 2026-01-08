import json
import requests
import gradio as gr
from typing import List, Dict

from langchain_ollama import ChatOllama

MODEL = "llama3.2:latest"
API_URL = "https://pzvpngd6ij.execute-api.ap-southeast-5.amazonaws.com/dev/create"
#API_URL = "http://127.0.0.1:8000/receive"

EXTRACT_SYSTEM = """
You are an information extraction assistant.
Extract ONLY the CEO name from the user's message.

Return ONLY valid JSON with exactly this key:
{"ceo_name":""}

Rules:
- If CEO name is not present, return an empty string.
- Do NOT add extra keys.
- Do NOT wrap in markdown.
"""

llm = ChatOllama(model=MODEL)


def safe_json_extract(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def extract_ceo_name(user_text: str) -> str:
    resp = llm.invoke([
        ("system", EXTRACT_SYSTEM),
        ("human", user_text),
    ]).content.strip()

    data = safe_json_extract(resp)
    return (data.get("ceo_name") or "").strip()


def send_customer_name(ceo_name: str):
    payload = {"customer_name": ceo_name}
    r = requests.post(
        API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def on_submit(
    user_input: str,
    messages: List[Dict],
    stage: str,
    ceo_name: str,
):
    messages = messages or []
    user_input = (user_input or "").strip()

    if not user_input:
        return messages, stage, ceo_name

    # ---- intake ----
    if stage == "intake":
        try:
            ceo = extract_ceo_name(user_input)
        except Exception as e:
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": f"I couldn't understand that.\n({e})"})
            return messages, stage, ceo_name

        messages.append({"role": "user", "content": user_input})

        if not ceo:
            messages.append({
                "role": "assistant",
                "content": "I couldn’t find the CEO name. Please mention it explicitly."
            })
            return messages, stage, ceo_name

        ceo_name = ceo
        stage = "confirm"
        messages.append({
            "role": "assistant",
            "content": f'I extracted the CEO name as "{ceo}". Is this correct? (yes/no)'
        })
        return messages, stage, ceo_name

    # ---- confirm ----
    if stage == "confirm":
        messages.append({"role": "user", "content": user_input})

        if user_input.lower().startswith("y"):
            try:
                result = send_customer_name(ceo_name)

                if result.get("success") is True:
                    messages.append({
                        "role": "assistant",
                        "content": f"CEO name saved (ID: {result.get('id')}). You may continue chatting."
                    })
                    stage = "chat"
                else:
                    messages.append({
                        "role": "assistant",
                        "content": f" API responded with : {result}"
                    })

            except Exception as e:
                messages.append({
                    "role": "assistant",
                    "content": f" Request failed: {e}"
                })
            return messages, stage, ceo_name
        else:
            stage = "intake"
            ceo_name = ""
            messages.append({
                "role": "assistant",
                "content": "Okay, please re-enter the information and include the CEO name."
            })
            return messages, stage, ceo_name

    # ---- chat (normal, no streaming) ----
    if stage == "chat":
        messages.append({"role": "user", "content": user_input})

        # Convert messages to LangChain format
        lc_messages = []
        for m in messages:
            role = "human" if m["role"] == "user" else "assistant"
            lc_messages.append((role, m["content"]))

        try:
            reply = llm.invoke(lc_messages).content
        except Exception as e:
            reply = f"(error) {e}"

        messages.append({"role": "assistant", "content": reply})
        return messages, stage, ceo_name


with gr.Blocks(title="Chat") as demo:
    gr.Markdown("## Chat")

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Type here and press Enter", autofocus=True)

    state_stage = gr.State("intake")
    state_ceo = gr.State("")

    demo.load(
        lambda: (
            [{"role": "assistant", "content": "Tell me about your company or leadership. Please include the CEO name."}],
            "intake",
            "",
        ),
        None,
        [chatbot, state_stage, state_ceo],
    )

    msg.submit(
        on_submit,
        inputs=[msg, chatbot, state_stage, state_ceo],
        outputs=[chatbot, state_stage, state_ceo],
    ).then(lambda: "", None, msg)

demo.launch()
