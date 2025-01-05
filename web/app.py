import gradio as gr
import requests

from web.utils.constants import *

def chat_with_backend(messages, history):

    if not messages.strip():  # 检查用户消息是否为空
        return Empty_QUERY_REPLY, gr.update(visible=False)

    url = "http://0.0.0.0:19198/chat/kb_chat"  # 后端接口
    headers = {"Content-Type": "application/json"}
    payload = {"query": messages, "history": history}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        reply = data.get("data", SERVER_REPLY_INVALID)
        return reply, gr.update(visible=False)
    except requests.exceptions.RequestException as e:
        return SERVER_CONNECT_FAILED, gr.update(visible=False)


with gr.Blocks(theme="soft", css=
                """
                .gradio-container {
                    max-width: 50% !important;
                    max-height: 90% !important;

                    position: absolute !important;
                    left: 0 !important;
                    right: 0 !important;
                    top: 0 !important;
                    bottom: 0 !important;
                    margin: auto !important;

                }
                .text {
                    position: fixed;
                    top: 35%;
                    left: 40%;
                    right: 40%;
                    margin: 0 auto;
                    z-index: 10;
                    height: auto; 
                    text-align: center;
                }

                """) as demo:
    example_text = gr.HTML("<div class='text'><h3>可以这样问我：</h3></div>")
    chat = gr.ChatInterface(fn=chat_with_backend, type='messages',
                examples=[EXAMPLE_1, EXAMPLE_2],
                fill_width=False,
                theme="soft",
                show_progress='hidden',
                additional_outputs=example_text,
                title=TITLE)


# demo.add(block)
demo.fill_height = True

if __name__ == '__main__':
    demo.launch()
