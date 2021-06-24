import aiohttp
from aiohttp import formdata
import asyncio
import os
import time
import json


async def do_recognize(web_ip, web_port, file_path):
    try:
        codename = os.path.basename(file_path)
        print('filename: {}'.format(file_path))
        # file_data = {"file": open(file_path, "rb")}
        file_data = formdata.FormData()
        file_data.add_field('file',
                       open(file_path, 'rb'),
                       # content_type="multipart/form-data; boundary=--dd7db5e4c3bd4d5187bb978aef4d85b1",
                       filename=codename)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url='http://{}:{}/mesh/recognize'.format(web_ip, web_port),
                    data=file_data,
                    # headers={'Content-Type': 'multipart/form-data; boundary=--dd7db5e4c3bd4d5187bb978aef4d85b1'}
            ) as resp:
                respond = await resp.text()
                respond = respond.replace('\\"', '"')
                respond = respond[1:-1]   # remove " at the begin and end
                result = json.loads(respond)
                data = result.get("text")
                status = result.get("returnCode")
                print("predict {}, res: {}".format(status, data))
    except:
        print("do_recognize Error {} \n".format(file_path))


def run_test(web_ip, web_port, filename, loop):
    try:
        loop.run_until_complete(do_recognize(web_ip, web_port, filename))
    except:
       print ("run_test Error")


if __name__ == '__main__':
    file_dir = "E:/code/python_web/MeshCNN/test_models/"
    filenames = os.listdir(file_dir)
    files = [os.path.join(file_dir, filename) for filename in filenames]
    ip = "127.0.0.1"   # "10.99.11.223"  #
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

