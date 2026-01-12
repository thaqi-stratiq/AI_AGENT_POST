import json
import requests
import gradio as gr
from typing import List, Dict, Any, Tuple

from langchain_ollama import ChatOllama

#MODEL = "llama3.2:latest"
MODEL = "qwen2.5:0.5b-instruct"
API_URL = "https://pzvpngd6ij.execute-api.ap-southeast-5.amazonaws.com/dev/create"
#API_URL = "http://127.0.0.1:8000/receive"

INDUSTRIES = [
    "Aerospace",
    "Education & Training",
    "Medical Devices",
    "Smart Buildings-Cities",
    "Energy Technology",
    "Robotics and Automation",
    "Urban Air Mobility (UAM)",
]

llm = ChatOllama(model=MODEL)
SYSTEM_EXTRACTOR = """
You are a STRICT JSON assistant for an intake flow.

Input JSON:
{
  "expected_field": "company_name" | "company_background" | "industry_pick" | "customer_name",
  "user_message": "..."
}

Decide mode:
- mode="qa" if user_message is a general question (e.g., "what is odoo", "how ERP works").
- mode="intake" if user_message attempts to answer expected_field.

QA mode:
- Put the helpful answer (2–6 sentences) in "answer".
- Set "value" = "".

Intake mode:
- Extract ONLY the expected_field into "value".
- If user did not clearly provide the expected_field, set "value" = "".

Special:
- expected_field="industry_pick": normalize partial inputs to one of the allowed industry labels ONLY if clearly intended,
  otherwise value="".

Return ONLY valid JSON with EXACTLY these keys:
{
  "mode": "qa" | "intake",
  "answer": "",
  "value": ""
}
No extra keys. No markdown. No text outside JSON.
"""


SYSTEM_INDUSTRY = f"""
You are a STRICT industry classifier.

Choose EXACTLY ONE industry from this list:
{INDUSTRIES}

Input: company_background (text)

Rules:
- Always return exactly one industry from the list (never empty).
- Choose the best match even if the background is short.
- Return ONLY valid JSON with EXACTLY this key:
{{"industry_name": ""}}
No extra keys. No markdown. No text outside JSON.
"""


def on_submit(user_input, messages, state):
    messages = messages or []
    state = state or {
        "step": "ask_company_name",
        "company_name": "",
        "company_background": "",
        "industry_name": "",
        "industry_confirmed": False,
        "customer_name": "",
        "created": False,
    }

    user_input = (user_input or "").strip()
    if not user_input:
        return messages, state

    messages.append({"role": "user", "content": user_input})

    def parse_json(raw: str) -> dict:
        raw = (raw or "").strip()

        # Fast path: extract first JSON object if present
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass

        # If no JSON found, return a safe fallback dict instead of crashing
        # This prevents Gradio worker from dying
        return {
            "mode": "qa",
            "answer": f"(server) Model returned non-JSON output:\n{raw[:500]}",
            "value": ""
        }

    def llm_extract(expected_field: str, text: str) -> dict:
        payload = {"expected_field": expected_field, "user_message": text}
        raw = llm.invoke([("system", SYSTEM_EXTRACTOR), ("human", json.dumps(payload))]).content.strip()
        return parse_json(raw)

    def llm_classify_industry(background: str) -> str:
        raw = llm.invoke([("system", SYSTEM_INDUSTRY), ("human", background)]).content.strip()
        data = parse_json(raw)
        industry = (data.get("industry_name") or "").strip()
        return industry

    def is_yes(txt: str) -> bool:
        t = txt.strip().lower()
        return t in {"y", "yes", "yeah", "yep", "correct", "ya", "betul", "true"}

    def is_no(txt: str) -> bool:
        t = txt.strip().lower()
        return t in {"n", "no", "nope", "tak", "tidak", "bukan", "false"}

    # optional reset after creation
    if state.get("created"):
        state.update({
            "step": "ask_company_name",
            "company_name": "",
            "company_background": "",
            "industry_name": "",
            "industry_confirmed": False,
            "customer_name": "",
            "created": False,
        })
        messages.append({"role": "assistant", "content": "Want to create another instance? What’s your company name?"})
        return messages, state

    step = state["step"]


    if step == "ask_company_name":
        data = llm_extract("company_name", user_input)
        if data["mode"] == "qa":
            messages.append({"role": "assistant", "content": data["answer"].strip()})
            messages.append({"role": "assistant", "content": "Now, what’s your company name?"})
            return messages, state

        name = (data.get("value") or "").strip()
        if not name:
            messages.append({"role": "assistant", "content": "What company name should I use? (You can give a placeholder.)"})
            return messages, state

        state["company_name"] = name
        state["step"] = "ask_background"
        messages.append({"role": "assistant", "content": "Nice. What does your company do (1–2 sentences)?"})
        return messages, state


    if step == "ask_background":
        data = llm_extract("company_background", user_input)
        if data["mode"] == "qa":
            messages.append({"role": "assistant", "content": data["answer"].strip()})
            messages.append({"role": "assistant", "content": "Back to intake: what does your company do (1–2 sentences)?"})
            return messages, state

        bg = (data.get("value") or "").strip()
        if not bg:
            messages.append({"role": "assistant", "content": "Could you describe what your company does in 1–2 sentences?"})
            return messages, state

        state["company_background"] = bg

        # LLM #2 classification (always one of 7)
        industry = llm_classify_industry(bg)

        # hard guard in case model misbehaves
        if industry not in INDUSTRIES:
            messages.append({
                "role": "assistant",
                "content": "I couldn’t classify reliably. Please choose one:\n" + "\n".join([f"- {x}" for x in INDUSTRIES])
            })
            state["step"] = "pick_industry"
            return messages, state

        state["industry_name"] = industry
        state["industry_confirmed"] = False
        state["step"] = "confirm_industry"
        messages.append({"role": "assistant", "content": f"I mapped your industry to **{industry}**. Is that correct? (yes/no)"})
        return messages, state


    if step == "confirm_industry":
        # allow QA without advancing
        gate = llm_extract("industry_pick", user_input)
        if gate["mode"] == "qa":
            messages.append({"role": "assistant", "content": gate["answer"].strip()})
            messages.append({"role": "assistant", "content": f"Back to intake: is **{state['industry_name']}** correct? (yes/no)"})
            return messages, state

        if is_yes(user_input):
            state["industry_confirmed"] = True
            state["step"] = "ask_customer_name"
            messages.append({"role": "assistant", "content": "Great. Who should I create the simulator instance for? (customer name)"})
            return messages, state

        if is_no(user_input):
            state["industry_name"] = ""
            state["industry_confirmed"] = False
            state["step"] = "pick_industry"
            messages.append({"role": "assistant", "content": "Okay — choose one industry:\n" + "\n".join([f"- {x}" for x in INDUSTRIES])})
            return messages, state

        messages.append({"role": "assistant", "content": "Please reply **yes** or **no**."})
        return messages, state

    if step == "pick_industry":
        data = llm_extract("industry_pick", user_input)
        if data["mode"] == "qa":
            messages.append({"role": "assistant", "content": data["answer"].strip()})
            messages.append({"role": "assistant", "content": "Now, please choose one industry:\n" + "\n".join([f"- {x}" for x in INDUSTRIES])})
            return messages, state

        pick = (data.get("value") or "").strip()
        if pick not in INDUSTRIES:
            messages.append({"role": "assistant", "content": "Please pick exactly one:\n" + "\n".join([f"- {x}" for x in INDUSTRIES])})
            return messages, state

        state["industry_name"] = pick
        state["industry_confirmed"] = True
        state["step"] = "ask_customer_name"
        messages.append({"role": "assistant", "content": f"Got it — **{pick}**. Who should I create the simulator instance for? (customer name)"})
        return messages, state

    if step == "ask_customer_name":
        data = llm_extract("customer_name", user_input)
        if data["mode"] == "qa":
            messages.append({"role": "assistant", "content": data["answer"].strip()})
            messages.append({"role": "assistant", "content": "Back to intake: what customer name should I create the simulator instance for?"})
            return messages, state

        customer = (data.get("value") or "").strip()
        if not customer:
            messages.append({"role": "assistant", "content": "What customer name should I create the simulator instance for?"})
            return messages, state

        if not state.get("industry_confirmed") or state.get("industry_name") not in INDUSTRIES:
            state["step"] = "pick_industry"
            messages.append({"role": "assistant", "content": "Before creating, choose one industry:\n" + "\n".join([f"- {x}" for x in INDUSTRIES])})
            return messages, state

        state["customer_name"] = customer

        try:
            r = requests.post(
                API_URL,
                json={"customer_name": state["customer_name"], "industry_name": state["industry_name"]},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            r.raise_for_status()
            result = r.json()
        except Exception as e:
            messages.append({"role": "assistant", "content": f"Request failed: {e}"})
            return messages, state

        # user wont directly see the api status, only the processed result
        ok = (result.get("success") is True) or (result.get("status") == "Message received successfully")
        if ok:
            messages.append({
                "role": "assistant",
                "content": f"Created simulator instance for **{state['customer_name']}** under **{state['industry_name']}**."
            })
            state["created"] = True
            state["step"] = "done"
            return messages, state

        messages.append({"role": "assistant", "content": f"API responded with: {result}"})
        return messages, state

    messages.append({"role": "assistant", "content": "Done. If you want another, type anything to restart."})
    state["created"] = True
    return messages, state

#the ui part is here 
with gr.Blocks(title="Odoo Simulator Chat") as demo:
    gr.Markdown("## Odoo Simulator Chat")

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Type here…", autofocus=True)
    state = gr.State(None)

    demo.load(
        lambda: (
            [{"role": "assistant", "content": "Hi! What’s your company name?"}],
            None,
        ),
        None,
        [chatbot, state],
    )

    msg.submit(on_submit, inputs=[msg, chatbot, state], outputs=[chatbot, state]).then(lambda: "", None, msg)

#demo.launch()
demo.launch(server_name="127.0.0.1",server_port=7860)
