import queue
import threading
import time
import openai
import json
import requests
from collections import deque
from datetime import datetime, timedelta
from requests.exceptions import RequestException, SSLError
from openai import OpenAI, APITimeoutError, APIConnectionError

from common import TranslationTask


def translate_text(text_to_translate):
    if not text_to_translate:
        return "......"

    url = "https://api.deeplx.org/translate"
    payload = json.dumps({
        "text": text_to_translate,
        "source_lang": "auto",
        "target_lang": "ZH"
    })
    headers = {
        "authority": "api.deeplx.org",
        "method": "POST",
        "path": "/translate",
        "scheme": "https",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Content-Length": "60",
        "Content-Type": "application/json",
        "Origin": "chrome-extension://bpoadfkcbjbfhfodiogcnhhhpibjhbnh",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "none",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            response_data = response.json()
            return response_data['data'] if response_data['data'] else "......"
        else:
            return "Error: Non-200 status code received"
    except SSLError:
        return "SSL Error occurred"
    except RequestException as e:
        return f"Request error: {e}"


def _translate_by_gpt(client: OpenAI,
                      translation_task: TranslationTask,
                      assistant_prompt: str,
                      model: str,
                      history_messages: list = []):
    # https://platform.openai.com/docs/api-reference/chat/create?lang=python
    try:
        url = 'http://127.0.0.1:3000/api/v1/chat/completions'
        headers = {'Content-Type': 'application/json'}
        data = {
            "messages": [{"role": "user",
                          "content": "Take a deep breath and translate the following Japanese sentences into Chinese, output only the translation, and do not have extra words, don't use web searches. The sentence is as follows:遊びましょ"},
                         {"role": "assistant", "content": "让我们一起玩吧"}, {"role": "user",
                                                                              "content": "Take a deep breath and translate the following Japanese sentences into Chinese, output only the translation, and do not have extra words, don't use web searches. The sentence is as follows:" + translation_task.transcribed_text}],
            "stream": False,
            "model": "Precise"
        }

        response = requests.post(url, headers=headers, json=data, verify=False)

        # 将响应文本解析为JSON
        response_json = response.json()

        # 提取content的值
        content = response_json["choices"][0]["delta"]["content"]
        translation_task.translated_text = content
    except (APITimeoutError, APIConnectionError) as e:
        print("API error: {}".format(e))


class ParallelTranslator():

    def __init__(self, prompt: str, model: str, timeout: int):
        self.prompt = prompt
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(base_url="http://127.0.0.1:3000/api/v1")
        self.processing_queue = deque()

    def trigger(self, translation_task: TranslationTask):
        self.processing_queue.append(translation_task)
        translation_task.start_time = datetime.utcnow()
        thread = threading.Thread(target=_translate_by_gpt,
                                  args=(self.client, translation_task, self.prompt, self.model))
        thread.daemon = True
        thread.start()

    def get_results(self):
        results = []
        while self.processing_queue and (self.processing_queue[0].translated_text or
                                         datetime.utcnow() - self.processing_queue[0].start_time
                                         > timedelta(seconds=self.timeout)):
            task = self.processing_queue.popleft()
            results.append(task)
            if not task.translated_text:
                print(
                    '\033[1m' + "GPT翻译超时,回落到deepl\n" + translate_text(task.transcribed_text) + '\033[0m' + "\n")
        return results

    def work(self, input_queue: queue.SimpleQueue[TranslationTask],
             output_queue: queue.SimpleQueue[TranslationTask]):
        while True:
            if not input_queue.empty():
                task = input_queue.get()
                self.trigger(task)
            finished_tasks = self.get_results()
            for task in finished_tasks:
                output_queue.put(task)
            time.sleep(0.1)


class SerialTranslator():

    def __init__(self, prompt: str, model: str, timeout: int, history_size: int):
        self.prompt = prompt
        self.model = model
        self.timeout = timeout
        self.history_size = history_size
        self.client = OpenAI()
        self.history_messages = []

    def work(self, input_queue: queue.SimpleQueue[TranslationTask],
             output_queue: queue.SimpleQueue[TranslationTask]):
        current_task = None
        while True:
            if current_task:
                if (current_task.translated_text or datetime.utcnow() - current_task.start_time
                        > timedelta(seconds=self.timeout)):
                    if current_task.translated_text:
                        # self.history_messages.append({"role": "user", "content": current_task.transcribed_text})
                        self.history_messages.append({
                            "role": "assistant",
                            "content": current_task.translated_text
                        })
                        while (len(self.history_messages) > self.history_size):
                            self.history_messages.pop(0)
                    else:
                        print("Translation timeout or failed: {}".format(
                            current_task.transcribed_text))
                    output_queue.put(current_task)
                    current_task = None

            if current_task is None and not input_queue.empty():
                current_task = input_queue.get()
                current_task.start_time = datetime.utcnow()
                thread = threading.Thread(target=_translate_by_gpt,
                                          args=(self.client, current_task, self.prompt, self.model,
                                                self.history_messages))
                thread.daemon = True
                thread.start()
            time.sleep(0.1)
