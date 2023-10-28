import unittest
from ray_vllm_inference.prompt_format import Message
from ray_vllm_inference.model_config import load_model_config

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

class Llama2PromptFormatCases(unittest.TestCase):
    """See https://huggingface.co/blog/llama2#how-to-prompt-llama-2"""

    def test_generate_prompt(self):

        model_config = load_model_config(MODEL_ID)

        messages = [
            Message(role='system', content='{{ system_prompt }}'),
            Message(role='user', content='{{ user_msg_1 }}'),
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        expected = """<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST]"""            
        self.assertEqual(prompt, expected)


    def test_generate_two_user_messages_prompt(self):

        model_config = load_model_config(MODEL_ID)

        messages = [
            Message(role='system', content='{{ system_prompt }}'),
            Message(role='user', content='{{ user_msg_1 }}'),
            Message(role='assistant', content='{{ model_answer_1 }}'),
            Message(role='user', content='{{ user_msg_2 }}'),
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        expected = """<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]"""            
        self.assertEqual(prompt, expected)


    def test_generate_three_user_messages_prompt(self):
        model_config = load_model_config(MODEL_ID)
        messages = [
            Message(role='system', content='{{ system_prompt }}'),
            Message(role='user', content='{{ user_msg_1 }}'),
            Message(role='assistant', content='{{ model_answer_1 }}'),
            Message(role='user', content='{{ user_msg_2 }}'),
            Message(role='assistant', content='{{ model_answer_2 }}'),
            Message(role='user', content='{{ user_msg_3 }}'),
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        expected = """<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST] {{ model_answer_2 }} </s><s>[INST] {{ user_msg_3 }} [/INST]"""            
        self.assertEqual(prompt, expected)


    def test_generate_prompt_without_system_message(self):

        model_config = load_model_config(MODEL_ID)

        messages = [
            Message(role='user', content='{{ user_msg_1 }}'),
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        expected = """<s>[INST] <<SYS>>
You are an assistant.
<</SYS>>

{{ user_msg_1 }} [/INST]"""            
        self.assertEqual(prompt, expected)
