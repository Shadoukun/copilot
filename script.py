from typing import Iterator
from functools import partial

import re
from dataclasses import dataclass
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

params = {
        "display_name": "Copilot",
        "is_tab": True,
        "useMemory": False,
        "selection": None,
        "previous": { "selection": None, "text": ""}
}

quickInstructPattern = re.compile('^(.*)(?<=---\n)(.*?)(?=\n---$)', re.DOTALL)

def set_previous_generation(text: str, selection: Selection = None):
    '''Set the previous generation values for retry/undo'''
    params['previous']['selection'] = selection
    params['previous']['text'] = text

def generate_system_prompt(renderer, memory: str, state: dict, extra_messages: list[str] = []) -> str:
    '''Generate the system prompt'''

    messages: list = []

    # Add the custom system message if it exists
    if state['custom_system_message'].strip():
        messages.append({"role": "system", "content": state['custom_system_message']})

    # If memory is enabled
    if params['useMemory']:
        messages.append({"role": "system", "content": memory})

    # Add any extra messages
    if extra_messages:
        for msg in extra_messages:
            messages.append({"role": "system", "content": msg})

    prompt: str = renderer(messages=messages)

    return prompt

def render_prompt(text: str, memory: str, state: dict, extra_system_messages: list[str] = []) -> str:
    '''render regular prompt with template'''

    # Get the model's instruction template from the state
    instruction_template = chat.jinja_env.from_string(state['instruction_template_str'])
    renderer = partial(instruction_template.render, add_generation_prompt=False)

    # Generate the system prompt
    system_prompt: str = generate_system_prompt(renderer, memory, state, extra_system_messages)
    # get the assistant prefix
    prefix, _ = get_generation_prompt(renderer, impersonate=False)

    # Calculate the maximum length and truncate the text
    max_length: int = get_max_prompt_length(state) - get_encoded_length("\n".join([system_prompt, prefix]))
    if get_encoded_length(text) > max_length:
        text = text[-max_length:]

    # Simply merge the system prompt, the prefix, and text
    prompt = "\n".join([system_prompt, prefix, text])

    return prompt

def render_instruct_prompt(instruction: list[str], memory: str, state: dict, chat_history: list[dict] = None) -> str:
    '''render instruct prompt with template'''

    previous_text: str = instruction[0]
    question: str = instruction[1]

    # Get the model's instruction template from the state
    instruction_template = chat.jinja_env.from_string(state['instruction_template_str'])
    renderer = partial(instruction_template.render, add_generation_prompt=False)

    # Generate the system prompt
    system_prompt = generate_system_prompt(renderer, memory, state)
    prefix, _ = get_generation_prompt(renderer, impersonate=False)

    messages: list = []

    messages.append({"role": "system", "content": previous_text})

    # Add the chat history to the message list if it's provided
    if chat_history:
        for chat_message in chat_history:
            if chat_message[0]:
                messages.append({"role": "user", "content": chat_message[0]})
            if chat_message[1]:
                messages.append({"role": "assistant", "content": chat_message[1]})

    # Add the user message to the list and render the prompt. list should be 2 messages
    messages.append({"role": "user", "content": question})
    prompt: str = renderer(messages=messages)

    # Calculate the maximum length and truncate text
    max_length: int = get_max_prompt_length(state) - get_encoded_length("\n".join([system_prompt, question]))

    # if the text is too long, truncate it and readd the previous text to the message list and rerender the prompt
    # rerendering with an arbitrary amount of text sucks (speed?), but it's the only way to get the correct length
    if get_encoded_length(prompt) > max_length:
        previous_text = previous_text[-max_length:]
        messages[0] = {"role": "system", "content": previous_text}
        prompt: str = renderer(messages=messages)

    # merge the rendered prompts
    prompt = "\n".join([system_prompt, prompt, prefix])

    return prompt

def reply_wrapper(text: str, memory: str, state: dict) -> Iterator[str]:

    prompt: str = ""
    instruct: bool = False

    # check for instruct pattern
    if instructions := quickInstructPattern.findall(text):
        instruct = True
        # render instruction prompt
        prompt = render_instruct_prompt(instructions[-1], memory, state)
    else:
        # render regular prompt
        prompt = render_prompt(text, memory, state)

    # strip instruct delimiters
    prompt = prompt.replace('---', "")

    for reply in generate_reply(prompt, state, False, False, False, False):
        if instruct:
            reply = "\n".join([text, reply])
        else:
            reply = "".join([text, reply])

        yield reply

    set_previous_generation(text)
    params['selection'] = None

def selection_reply_wrapper(text: str, memory: str, state: dict) -> Iterator[str]:

    # fallback to regular generation if there's no selection
    if params['selection'] is None:
        reply_wrapper(text, memory, state)
        return

    prompt: str = ""
    select: Selection = params['selection']
    x, y = select.index[0], select.index[1]

    # split before and after the selection.
    before: str = text[:x]
    selection: str = text[x:y]
    after: str = text[y:]

    prompt = render_prompt(selection, memory, state, extra_system_messages=[before])

    # strip instruct delimiters
    prompt = prompt.replace('---', "")

    for reply in generate_reply(prompt, state, False, False, False, False):
        if hasattr(shared, 'is_seq2seq') and not shared.is_seq2seq:
            reply = selection + reply

        reply = "".join([before, reply, after])
        yield reply

    set_previous_generation(text, select)
    params['selection'] = None

def copilot_reply_wrapper(text: str, memory: str, question: str, chat_history: list[list[str]], state: dict) -> Iterator[tuple[str, list[list[str, str]]]]:

    instruction = [text, question]
    prompt: str = render_instruct_prompt(instruction, memory, state, chat_history=chat_history)

    # add the user message and empty reply to the chat history
    chat_history = chat_history + [[question, ""]]

    for reply in generate_reply(prompt, state, False, False, False, False):
        # generate the reply and append it to the chat history
        chat_history[-1][1] = reply
        yield "", chat_history

def onSelect(evt: gr.SelectData):
    params.update({'selection': Selection(index=evt.index, value=evt.value)})

def update_memory(value: bool):
    params.update({'useMemory': value})


def custom_css():
    return """
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto; height: 65vh;}
    """

def ui():

    with gr.Row():
        # add a box. remove padding.
        with gr.Column(scale=2):
            textBox = gr.Textbox(value='', lines=20, label ='', elem_classes=['textbox', 'add_scrollbar'], elem_id='textbox')
            with gr.Row():
                generateButton = gr.Button('Generate', variant='primary')
                generateSelectButton = gr.Button('Generate From Selection', variant='primary')
                stopButton = gr.Button('Stop', interactive=True)
            
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Tab("Text"):
                    chatbot = gr.Chatbot(elem_id="chatbot", height="65vh")
                    chattextBox = gr.Textbox(value='')

                with gr.Tab("Options"):
                    with gr.Accordion("Memory"):
                        memoryCheckbox = gr.Checkbox(value=params['useMemory'], label='Enabled')
                        memoryBox = gr.Textbox(value='', lines=20, label='', show_label=False, elem_classes=['textbox', 'add_scrollbar'])


    inputs = [textBox, memoryBox]

    ### Event Handlers
    chattextBox.submit(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(copilot_reply_wrapper, [*inputs, chattextBox, chatbot, shared.gradio['interface_state']], [chattextBox, chatbot], queue=True)

    textBox.select(onSelect, None, None)
    generateButton.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(reply_wrapper, inputs=[*inputs, shared.gradio['interface_state']], outputs=[textBox], show_progress=False)

    generateSelectButton.click(gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']) \
        .then(selection_reply_wrapper, inputs=[*inputs, shared.gradio['interface_state']], outputs=[textBox], show_progress=False)

    stopButton.click(stop_everything_event, None, None, queue=False)

    memoryCheckbox.change(update_memory, memoryCheckbox, None)
