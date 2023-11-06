from typing import List, Dict, Any
import os
import sys
import logging
import time
import argparse
import requests
import json

logger = logging.getLogger()

def _get_result(response: requests.Response) -> Dict[str, Any]:
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise RuntimeError(
            f"Error decoding JSON from {response.url}. Text response: {response.text}",
            response=response,
        ) from e
    return result

def main(host:str, port:int, stream:bool, user_message:str, max_tokens:int, temperature:float):

    logging.basicConfig(level=logging.INFO)

    url = f'http://{host}:{port}/generate'
    headers = {'Content-type': 'application/json'}
    payload = {
               "messages":[
                    {"role":"system", "content":"You are a helpful and truthful assistant."}, 
                    {"role":"user", "content":f"{user_message}"}
                ], 
               "stream": stream, 
               "max_tokens": max_tokens, 
               "temperature": temperature
            }

    if stream:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        if response.status_code == 200:
            with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
                start = time.perf_counter()
                num_tokens = 0
                for line in response.iter_lines():
                    #print(line.decode("utf-8")) # raw JSON response
                    text = json.loads(line.decode("utf-8"))['output']
                    stdout.write(text.encode("utf-8"))
                    stdout.flush()
                    num_tokens += 1
                duration_s = time.perf_counter() - start
                print(f"{num_tokens / duration_s} token/s")


        else:
            print(f'HTTP status code: {response.status_code}')
            print(_get_result(response))
                    
    else:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            result = _get_result(response)
            print(result)
        else:
            print(f'HTTP status code: {response.status_code}')
            print(_get_result(response))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--user-message", type=str, default="What can I do on a weekend trip to London?")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--no-stream', dest='stream', action='store_false')
    parser.set_defaults(stream=False)
    args = parser.parse_args()
    main(host=args.host, port=args.port, stream=args.stream, user_message=args.user_message, 
         max_tokens=args.max_tokens, temperature=args.temperature)