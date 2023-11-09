import unittest
from ray_vllm_inference.prompt_format import Message
from ray_vllm_inference.model_config import load_model_config

MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"

class CodeLlamaPromptFormatCases(unittest.TestCase):

    def test_generate_user_message(self):

        model_config = load_model_config(MODEL_ID)

        system = model_config.prompt_format.default_system_message
        user = "Write a function that computes the set of sums of all contiguous sublists of a given list."

        expected_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

        messages = [
            Message(role='user', content=user),
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        self.assertEqual(prompt, expected_prompt)


    def test_generate_user_message_with_system_message(self):

        model_config = load_model_config(MODEL_ID)

        system = "Provide answers in JavaScript"
        user = "Write a function that computes the set of sums of all contiguous sublists of a given list."

        expected_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

        messages = [
            Message(role='system', content=system),
            Message(role='user', content=user),
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        self.assertEqual(prompt, expected_prompt)


    def test_generate_conversation(self):

        model_config = load_model_config(MODEL_ID)

        system = "System prompt"
        user1 = "user_prompt_1"
        answer1 = "answer_1"
        user2 = "user_prompt_2"
        answer2 = "answer_2"
        user3 = "user_prompt_3"

        expected_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user1} [/INST]"
        expected_prompt += f" {answer1} </s><s>[INST] {user2} [/INST]"
        expected_prompt += f" {answer2} </s><s>[INST] {user3} [/INST]"

        messages = [
            Message(role='system', content=system),
            Message(role='user', content=user1),
            Message(role='assistant', content=answer1),
            Message(role='user', content=user2),
            Message(role='assistant', content=answer2),
            Message(role='user', content=user3)
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        self.assertEqual(prompt, expected_prompt)



    def test_foo(self):

        model_config = load_model_config(MODEL_ID)

        #system = "Provide answers in JavaScript"
        system = ''
        user = "Write a function that computes the set of sums of all contiguous sublists of a given list."

        messages = [
            Message(role='system', content=system),
            Message(role='user', content=user),
        ]
        prompt = model_config.prompt_format.generate_prompt(messages)
        print(prompt)

