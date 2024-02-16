from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator

import re
import gradio as gr
from modules import chat, shared
from modules.ui import gather_interface_values
from modules.chat import get_generation_prompt
from modules.text_generation import (
    generate_reply,
    stop_everything_event,
    get_max_prompt_length,
    get_encoded_length
)

@dataclass
class Selection:
    '''Dataclass for storing the selection index and value.'''
    index: tuple[int] = (0,0)
    value: str = ""

@dataclass
class Generation:
    '''Dataclass for storing the generation parameters.'''
    text: str = "" # the text to generate from
    type: str = "" # the type of generation (default, selection, instruct)
    selection: Selection = None # the selection if any
    prompt: str = "" # the prompt used for generation
    reply: str = "" # the reply generated

params = {
        "display_name": "Copilot",
        "is_tab": True,
        "use_memory": False,
        "selection": None,
        "history": [],
}

quickInstructPattern = re.compile('^(.*)(?<=---\n)(.*?)(?=\n---$)', re.DOTALL)

def generate_system_prompt(render: Callable, memory: str, state: dict, extra_messages: list[str] = []) -> str:
    '''Generate the system prompt'''

    messages: list = []

    # Add the custom system message if it exists
    if state['custom_system_message'].strip():
        messages.append({"role": "system", "content": state['custom_system_message']})

    # If memory is enabled
    if params['use_memory']:
        messages.append({"role": "system", "content": memory})

    # Add any extra messages
    if extra_messages:
        messages += [{"role": "system", "content": msg} for msg in extra_messages]

    return render(messages=messages)

def render_prompt(text: str, memory: str, state: dict, extra_system_messages: list[str] = []) -> str:
    '''render regular prompt with template'''

    # Get the model's instruction template from the state
    instruction_template = chat.jinja_env.from_string(state['instruction_template_str'])
    render: Callable = partial(instruction_template.render, add_generation_prompt=False)

    # Generate the system prompt
    system_prompt: str = generate_system_prompt(render, memory, state, extra_system_messages)
    # get the assistant prefix
    prefix, _ = get_generation_prompt(render, impersonate=False)

    # Calculate the maximum length and truncate the text
    max_length: int = get_max_prompt_length(state) - get_encoded_length("\n".join([system_prompt, prefix]))
    if get_encoded_length(text) > max_length:
        text = text[-max_length:]

    # Simply merge the system prompt, the prefix, and text
    return "\n".join([system_prompt, prefix, text])

def render_instruct_prompt(instruction: list[str], memory: str, state: dict, chat_history: list[dict] = None) -> str:
    '''render instruct prompt with template'''

    previous_text: str = instruction[0]
    question: str = instruction[1]

    # Get the model's instruction template from the state
    instruction_template = chat.jinja_env.from_string(state['instruction_template_str'])
    render: Callable = partial(instruction_template.render, add_generation_prompt=False)

    # Generate the system prompt
    system_prompt = generate_system_prompt(render, memory, state)
    prefix, _ = get_generation_prompt(render, impersonate=False)

    messages: list = []
    messages.append({"role": "system", "content": previous_text})

    # Add the chat history to the message list if it's provided
    if chat_history:
        for message in chat_history:
            messages.append({"role": "user", "content": message[0]})
            messages.append({"role": "assistant", "content": message[1]})

    # Add the user message to the list and render the prompt. list should be 2 messages
    messages.append({"role": "user", "content": question})
    prompt: str = render(messages=messages)

    # Calculate the maximum length and truncate text
    max_length: int = get_max_prompt_length(state) - get_encoded_length("\n".join([system_prompt, question]))

    # if the text is too long, truncate it and readd the previous text to the message list and rerender the prompt
    # rerendering with an arbitrary amount of text sucks (speed?), but it's the only way to get the correct length
    if get_encoded_length(prompt) > max_length:
        previous_text = previous_text[-max_length:]
        messages[0] = {"role": "system", "content": previous_text}
        prompt: str = render(messages=messages)

    # merge the rendered prompts
    return "\n".join([system_prompt, prompt, prefix])

def reply_wrapper(text: str, memory: str, state: dict, retry: bool = False) -> Iterator[str]:
    '''Generate a standard reply from the model.'''

    if retry:
        current = params['history'].pop()
    else:
        current = Generation(text, "default", None, "", "")

        # check for instruct pattern
        if instructions := quickInstructPattern.findall(current.text):
            current.type = "instruct"
            # render instruction prompt
            current.prompt = render_instruct_prompt(instructions[-1], memory, state)
        else:
            # render regular prompt
            current.prompt = render_prompt(current.text, memory, state)

        # strip instruct delimiters
        current.prompt = current.prompt.replace('---', "")

    for reply in generate_reply(current.prompt, state, False, False, False, False):
        if current.type == "instruct":
            reply = "\n".join([current.text, reply])
        else:
            reply = "".join([current.text, reply])

        yield reply

    current.reply = reply
    params['history'].append(current)
    params['selection'] = None

def selection_reply_wrapper(text: str, memory: str, state: dict, retry: bool = False) -> Iterator[str]:
    '''Generate a reply from the model using a selection.'''

    if retry:
        current = params['history'].pop()
    else:
        # fallback to regular generation if there's no selection
        if params['selection'] is None:
            yield from reply_wrapper(text, memory, state)
            return

        current = Generation(text, "selection", params['selection'], "", "")
        x, y = current.selection.index

        # split before and after the selection.
        before: str = text[:x]
        selection: str = text[x:y]
        after: str = text[y:]

        current.prompt = render_prompt(selection, memory, state, extra_system_messages=[before])

        # strip instruct delimiters
        current.prompt = current.prompt.replace('---', "")

    for reply in generate_reply(current.prompt, state, False, False, False, False):
        if hasattr(shared, 'is_seq2seq') and not shared.is_seq2seq:
            reply = selection + reply

        reply = "".join([before, reply, after])
        yield reply

    current.reply = reply
    params['history'].append(current)
    params['selection'] = None

def copilot_reply_wrapper(text: str, memory: str, question: str, chat_history: list[list[str]], state: dict) -> Iterator[tuple[str, list[list[str, str]]]]:
    '''Generate a reply for the copilot.'''

    instruction = [text, question]
    prompt: str = render_instruct_prompt(instruction, memory, state, chat_history=chat_history)

    # add the user message and empty reply to the chat history
    chat_history = chat_history + [[question, ""]]

    for reply in generate_reply(prompt, state, False, False, False, False):
        # generate the reply and append it to the chat history
        chat_history[-1][1] = reply
        yield "", chat_history

def retry(text: str, memory: str, state: dict) -> Iterator[str]:
    '''Retry the last generation using the same parameters.'''

    if len(params['history']) > 0:
        current = params['history'][-1]
        if current.type == "selection":
            yield from selection_reply_wrapper(current.text, memory, state, True)
        else:
            yield from reply_wrapper(current.text, memory, state, True)

def undo() -> str:
    '''Undo the last generation and returns the previous reply.'''

    if len(params['history']) > 0:
        # remove the last generation from the history
        params['history'].pop()

        # if there's still a generation in the history, return the previous reply
        if len(params['history']) > 0:
            previous = params['history'][-1]
            return previous.reply

def copilot_retry(text: str, memory: str, question: str, chat_history: list[list[str]], state: dict) -> Iterator[tuple[str, list[list[str, str]]]]:
    '''Retry the last copilot generation using the same parameters.'''

    if len(chat_history) > 0:
        last = chat_history.pop()
        yield from copilot_reply_wrapper(text, memory, last[0], chat_history, state)

def copilot_undo(chat_history: list[list[str]]) -> list[list[str, str]]:
    '''Undo the last copilot generation and returns the previous chat history.'''

    if len(chat_history) > 0:
        chat_history.pop()
        return chat_history

def onSelect(evt: gr.SelectData):
    params.update({'selection': Selection(index=evt.index, value=evt.value)})

def update_memory(value: bool):
    params.update({'use_memory': value})

def custom_css():
    return """
    #chatbot { flex-grow: 1; overflow: auto; height: 65vh; }
    #chat_submit_button { flex-grow: 0; min-width: 3em; padding: 0 2em; }
    """

def ui():
    with gr.Row():
        # add a box. remove padding.
        with gr.Column(scale=2):
            textbox = gr.Textbox(value='', lines=20, label='', elem_classes=['textbox', 'add_scrollbar'], elem_id='textbox')
            with gr.Row():
                generate_button = gr.Button('Generate', variant='primary')
                generate_select_button = gr.Button('Generate From Selection', variant='primary')
                stop_button = gr.Button('Stop', interactive=True)
            with gr.Row():
                retry_button = gr.Button('Retry')
                undo_button = gr.Button('Undo')

        with gr.Column(scale=1):
            with gr.Row():
                with gr.Tab("Text"):
                    chatbot = gr.Chatbot(elem_id="chatbot", height="65vh")

                    with gr.Row():
                        chat_textbox = gr.Textbox(label='', show_label=False, value='', elem_classes=['textbox'])
                        chat_submit_button = gr.Button("Submit", variant='primary', elem_id="chat_submit_button")
                    with gr.Row():
                        chat_retry_button = gr.Button('Retry')
                        chat_undo_button = gr.Button('Undo')

                with gr.Tab("Options"):
                    with gr.Accordion("Memory"):
                        memory_checkbox = gr.Checkbox(value=params['use_memory'], label='Enabled')
                        memory_textbox = gr.Textbox(value='', lines=20, label='', show_label=False, elem_classes=['textbox', 'add_scrollbar'])

    ### Event Handlers
    inputs = [textbox, memory_textbox]
    chat_inputs = [textbox, memory_textbox, chat_textbox, chatbot]

    textbox.select(onSelect, None, None)
    generate_button.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(reply_wrapper, inputs=[*inputs, shared.gradio['interface_state']], outputs=[textbox], show_progress=False)
    generate_select_button.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(selection_reply_wrapper, inputs=[*inputs, shared.gradio['interface_state']], outputs=[textbox], show_progress=False)
    stop_button.click(stop_everything_event, None, None, queue=False)
    retry_button.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(retry, inputs=[*inputs, shared.gradio['interface_state']], outputs=[textbox], show_progress=False)
    undo_button.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(undo, None, [textbox], show_progress=False)

    chat_textbox.submit(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(copilot_reply_wrapper, [*chat_inputs, shared.gradio['interface_state']], [chat_textbox, chatbot])
    chat_submit_button.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(copilot_reply_wrapper, [*chat_inputs, shared.gradio['interface_state']], [chat_textbox, chatbot])

    chat_retry_button.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(copilot_retry, [*chat_inputs, shared.gradio['interface_state']], [chat_textbox, chatbot])

    chat_undo_button.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(copilot_undo, chatbot, [chatbot])

    memory_checkbox.change(update_memory, memory_checkbox, None)
