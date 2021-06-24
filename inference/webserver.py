from aiohttp import web
from inference_class import InferenceClass
import json
import asyncio
import os
import base64


def image_from_base64(base64_utf8):
    decode_data = base64.decodebytes(base64_utf8.encode('utf-8'))
    return decode_data


def throttle(func):
    counter = 0

    async def _func(*args, **kwargs):
        nonlocal counter
        if counter >= 2:
            return web.Response(status=429, text="Too many connections")
        counter += 1
        try:
            return await func(*args, **kwargs)
        finally:
            counter -= 1

    return _func


class MeshWebServer(object):
    def __init__(self, max_request=1, cache_dir=None):
        self._app = web.Application()
        self._engine = InferenceClass()
        self._concurrency = asyncio.BoundedSemaphore(max_request)
        self._lock = asyncio.Lock()
        if cache_dir is not None:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = os.path.join(os.getcwd(), 'cache')

        # Init first predict
        test_file = "./test_models/test.obj"
        test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file)
        self._engine.inference(test_file)
        print("************ Server was ready! ************")

    def run(self, port):
        self._app.add_routes([
            web.post('/mesh/recognize', self.__on_recognize)
        ])
        web.run_app(self._app, port=port)

    async def __on_recognize(self, request):
        obj_data, filename, file_path = "", "", ""
        content_type = request.content_type
        # print("type", content_type)
        if content_type == "application/json":
            try:
                data = await request.json()
                if 'obj' in data:
                    obj_data = image_from_base64(data['obj'])
                if 'filename' in data:
                    filename = data['filename']
                else:
                    filename = "temp.obj"
                if "file_path" in data:
                    file_path = data["file_path"]
                    filename = os.path.basename(file_path)
                # print(file_path)

                if os.path.isfile(file_path):
                    file = file_path
                    predict_method = 1
                else:
                    if obj_data != "":
                        # # 方式一：保存数据至本地文件 对应self._engine.inference(file)
                        # os.makedirs(self._cache_dir, exist_ok=True)
                        # del_file_list = os.listdir(self._cache_dir)  # 存在文件的话先清空
                        # for f in del_file_list:
                        #     file_path = os.path.join(self._cache_dir, f)
                        #     if os.path.isfile(file_path):
                        #         os.remove(file_path)
                        #
                        # file = os.path.join(self._cache_dir, filename)
                        # with open(file, "wb") as f:
                        #     f.write(obj_data)

                        # 方式二: 直接解析
                        obj_data = obj_data.decode()
                        file = ""
                        predict_method = 2
                    else:
                        file = ""
                        predict_method = 1
                async with self._lock:
                    if 1 == predict_method:
                        predict_class = self._engine.inference(file)
                    else:
                        predict_class = self._engine.inference_obj(obj_data)
                    # respond = dict(text=predict_class[0][1][-1], returnCode="Successed!", filename=filename)  # 边标签概率
                    # respond = dict(text=predict_class[0][-1], returnCode="Successed!", filename=filename)  # 边标签/牙龈线点
                    if len(predict_class) < 1:
                        respond = dict(text=predict_class, returnCode="Failed!", filename=filename)
                    else:
                        respond = dict(text=predict_class, returnCode="Successed!", filename=filename)

            except Exception as e:
                respond = dict(text='', returnCode="Failed!", filename=filename, returnMsg=repr(e))
        elif content_type == "multipart/form-data":
            try:
                # print("headers: ", request.headers)

                reader = await request.multipart()
                field = await reader.next()
                filename = field.filename if field.filename else "temp.obj"
                size = 0
                os.makedirs(self._cache_dir, exist_ok=True)
                del_file_list = os.listdir(self._cache_dir)  # 存在文件的话先清空
                for f in del_file_list:
                    file_path = os.path.join(self._cache_dir, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                # 方式一：直接解析data  对应self._engine.inference_obj(obj_data.decode())
                obj_data = "".encode()
                while True:
                    chunk = await field.read_chunk()  # 默认是8192个字节。
                    # print("chunk: ", type(chunk), chunk)
                    if not chunk:
                        break
                    obj_data += chunk

                # 方式二：保存至本地 对应self._engine.inference(file)
                # file = os.path.join(self._cache_dir, filename)
                # with open(file, 'wb') as f:
                #     while True:
                #         chunk = await field.read_chunk()  # 默认是8192个字节。
                #         # print("chunk: ", type(chunk), chunk)
                #         if not chunk:
                #             break
                #         size += len(chunk)
                #         obj_data += chunk
                #         f.write(chunk)

                # # ----小文件----
                # data = await request.post()
                # file_data = data["file"]
                # file = file_data.file
                # filename = file_data.filename
                # content = file.read()
                #
                # os.makedirs(self._cache_dir, exist_ok=True)
                # del_file_list = os.listdir(self._cache_dir)  # 存在文件的话先清空
                # for f in del_file_list:
                #     file_path = os.path.join(self._cache_dir, f)
                #     if os.path.isfile(file_path):
                #         os.remove(file_path)
                #
                # file = os.path.join(self._cache_dir, filename)
                # with open(file, "wb") as f:
                #     f.write(content)

                async with self._lock:
                    # predict_class = self._engine.inference(file)  # 预测保存的文件
                    predict_class = self._engine.inference_obj(obj_data.decode())
                    # respond = dict(text=predict_class[0][1][-1], returnCode="Successed!", filename=filename)  # 边标签概率
                    # respond = dict(text=predict_class[0][-1], returnCode="Successed!", filename=filename)  # 边标签/牙龈线点
                    if len(predict_class) < 1:
                        respond = dict(text=predict_class, returnCode="Failed!", filename=filename)
                    else:
                        respond = dict(text=predict_class, returnCode="Successed!", filename=filename)

            except Exception as e:
                print(e)
                respond = dict(text='', returnCode="Failed!", filename=filename, returnMsg=repr(e))
        # elif content_type == "application/octet-stream":
        #     print("Enter octet, headers: ", request.headers)
        #     data = await request.post()
        #     respond = dict(text="octet-stream", returnCode="Successed!", filename=filename)
        #     print("data", data, dir(data), data.values)
        else:
            respond = dict(text="Unknown content type, just support application/json and multipart/form-data",
                           returnCode="Failed!", filename=filename)
        print("---** predict is {} **---".format(respond["returnCode"]))
        return web.json_response(json.dumps(respond))

