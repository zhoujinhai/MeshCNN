import aiohttp
import asyncio
import base64
import os
import json
import time


def base64_from_filename(filename):
    with open(filename, "rb") as file_binary:
        data = file_binary.read()
        encoded = base64.b64encode(data)
        encoded_utf8 = encoded.decode('utf-8')
    return encoded_utf8


async def do_recognize(web_ip, web_port, file_path, obj_data=None):
    codename = os.path.basename(file_path)
    try:
        if web_ip == "127.0.0.1":
            request = dict(file_path=file_path, filename=codename)
        else:
            request = dict(obj_base64=obj_data, filename=codename)
        print('filename: {}'.format(file_path))
        async with aiohttp.ClientSession() as session:
            async with session.post(url='http://{}:{}/mesh/recognize'.format(web_ip, web_port),
                                    data=json.dumps(request),
                                    headers={'Content-Type': 'application/json; charset=utf-8'}) as resp:
                respond = await resp.text()
                respond = respond.replace('\\"', '"')
                respond = respond[1:-1]   # remove " at the begin and end
                result = json.loads(respond)
                data = result.get("text")
                status = result.get("returnCode")
                print("predict {}, res: {}".format(status, data))
    except:
        print("do_recognize Error {} \n".format(codename))


def run_test(web_ip, web_port, filename, loop):
    try:
        if web_ip == "127.0.0.1":
            loop.run_until_complete(do_recognize(web_ip, web_port, filename))
        else:
            b64_str = base64_from_filename(filename)
            loop.run_until_complete(do_recognize(web_ip, web_port, filename, b64_str))
    except:
       print("run_test Error")


if __name__ == '__main__':
    file_dir = "E:/code/python_web/MeshCNN/test_models/"
    filenames = os.listdir(file_dir)
    files = [os.path.join(file_dir, filename) for filename in filenames]
    ip = "192.168.102.13"  # "127.0.0.1" 192.168.102.13
    port = 8000
    start = time.time()
    try:
        event_loop = asyncio.get_event_loop()
        tasks = [run_test(ip, port, filename, event_loop) for filename in files]
        event_loop.close()
    except:
        print("__main__ Error")
    end = time.time()
    print("run time is : {}s".format(end - start))

