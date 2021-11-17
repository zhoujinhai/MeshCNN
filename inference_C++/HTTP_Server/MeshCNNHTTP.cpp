#include "MeshCNNHTTP.h"
#include <cstring>
#include "json/json.h"
#include <curl/curl.h>


namespace core {
	MeshCNNHTTP::MeshCNNHTTP(std::string ip, std::string port) {
		IP = ip;
		Port = port;
	}


	MeshCNNHTTP::~MeshCNNHTTP() {

	}


	bool MeshCNNHTTP::reqMeshCNNFilePath(const char* dentalPath, std::string& result, std::string route) {
		bool reqOK = reqMeshCNNFilePath_(IP, Port, dentalPath, result, route);
		if (!reqOK){
			return false;
		}
		/*bool parseOK = parseResult(result);
		if (!parseOK) {
			return false;
		}*/
		return true;
	}


	bool MeshCNNHTTP::reqMeshCNNObjJson(const char* objJsonData, std::string& result, std::string route) {
		bool reqOK = reqMeshCNNObjJson_(IP, Port, objJsonData, result, route);
		if (!reqOK) {
			return false;
		}
		
		return true;
	}


	bool MeshCNNHTTP::reqMeshCNNFileUpload(const char* dentalPath, std::string& result, std::string route) {
		bool reqOK = reqMeshCNNFileUpload_(IP, Port, dentalPath, result, route);
		if (!reqOK) {
			return false;
		}
		/*bool parseOK = parseResult(result);
		if (!parseOK) {
			return false;
		}*/
		return true;
	}


	void MeshCNNHTTP::stringReplace(std::string& strOri, const std::string& strsrc, const std::string& strdst) {
		std::string::size_type pos = 0;
		std::string::size_type srclen = strsrc.size();
		std::string::size_type dstlen = strdst.size();

		while ((pos = strOri.find(strsrc, pos)) != std::string::npos) {
			strOri.replace(pos, srclen, strdst);
			pos += dstlen;
		}
	}


	std::string MeshCNNHTTP::getPathShortName(std::string strFullName) {
		if (strFullName.empty()) {
			return "";
		}

		stringReplace(strFullName, "\\", "/");

		std::string::size_type iPos = strFullName.find_last_of('/') + 1;

		return strFullName.substr(iPos, strFullName.length() - iPos);
	}


	/*std::wstring MeshCNNHTTP::AsciiToUnicode(const std::string& str)
	{
		// 预算-缓冲区中宽字节的长度  
		int unicodeLen = MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, nullptr, 0);
		// 给指向缓冲区的指针变量分配内存  
		wchar_t* pUnicode = (wchar_t*)malloc(sizeof(wchar_t) * unicodeLen);
		// 开始向缓冲区转换字节  
		MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, pUnicode, unicodeLen);
		std::wstring ret_str = pUnicode;
		free(pUnicode);
		return ret_str;
	}


	std::string MeshCNNHTTP::UnicodeToUtf8(const std::wstring& wstr)
	{
		// 预算-缓冲区中多字节的长度  
		int ansiiLen = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
		// 给指向缓冲区的指针变量分配内存  
		char* pAssii = (char*)malloc(sizeof(char) * ansiiLen);
		// 开始向缓冲区转换字节  
		WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, pAssii, ansiiLen, nullptr, nullptr);
		std::string ret_str = pAssii;
		free(pAssii);
		return ret_str;
	}


	std::string MeshCNNHTTP::AsciiToUtf8(const std::string& str)
	{
		return UnicodeToUtf8(AsciiToUnicode(str));
	}*/


	//回调函数
	size_t MeshCNNHTTP::write_data(void* ptr, size_t size, size_t nmemb, void* stream)
	{
		////std::cout << "****** predict done! ******" << std::endl;
		//std::string data((const char*)ptr, (size_t)size * nmemb);
		////*((std::stringstream*) stream) << data << std::endl;
		//std::string* result = static_cast<std::string*>(stream);
		//*result = data;
		//return size * nmemb;

		size_t realsize = size * nmemb;
		struct memory* mem = (struct memory*)stream;
		char* ptr1 = (char*)realloc(mem->response, mem->size + realsize + 1);
		if (ptr1 == NULL)
			return 0;

		mem->response = ptr1;
		std::memcpy(&(mem->response[mem->size]), ptr, realsize);
		mem->size += realsize;
		mem->response[mem->size] = 0;

		return realsize;
	}


	bool MeshCNNHTTP::reqMeshCNNFileUpload_(std::string IP, std::string Port, const char* dentalPath, std::string& result, std::string route) {
		std::string reqURL = "http://" + IP + ":" + Port + route;

		CURL* curl = nullptr;
		struct memory stBody;

		curl_global_init(CURL_GLOBAL_ALL);

		// 初始化easy handler句柄
		curl = curl_easy_init();
		if (nullptr == curl) {
			printf("failed to create curl connection.\n");
			return false;
		}

		// 设置post请求的url地址
		CURLcode code;
		code = curl_easy_setopt(curl, CURLOPT_URL, reqURL.c_str());
		if (code != CURLE_OK) {
			printf("failed to set url.\n");
			return false;
		}

		// 设置回调函数，如果不设置，默认输出到控制台
		code = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		if (code != CURLE_OK) {
			printf("failed to set write data.\n");
			return false;
		}

		//设置接收数据的处理函数和存放变量
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&stBody);

		//初始化form
		curl_mime* form;
		curl_mimepart* field;
		/* Create the form */
		form = curl_mime_init(curl);
		/* Fill in the file upload field */
		field = curl_mime_addpart(form);
		curl_mime_name(field, "file");
		curl_mime_filedata(field, dentalPath);
		/*seed form*/
		curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

		//HTTP报文头
		struct curl_slist* headers = nullptr;
		/*set header info，You can set more than one bar*/
		headers = curl_slist_append(headers, "Accept-Encoding: gzip, deflate");
		headers = curl_slist_append(headers, "User-Agent: curl");
		/* Pass in the HTTP header*/
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		//执行单条请求
		CURLcode res;
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			printf("curl_easy_perform failed! {}\n");
			return false;
		}
		result = stBody.response;

		// 释放资源
		curl_slist_free_all(headers);
		/* 释放form */
		curl_mime_free(form);
		/* always cleanup */
		curl_easy_cleanup(curl);
		/*对curl_global_init做的工作清理 类似于close的函数*/
		curl_global_cleanup();

		return true;
	}


	bool MeshCNNHTTP::reqMeshCNNFilePath_(std::string IP, std::string Port, const char* dentalPath, std::string& result, std::string route) {
		std::string reqURL = "http://" + IP + ":" + Port + route;
		struct memory stBody;
		// json 数据
		std::string filename = getPathShortName(dentalPath);

		Json::Value value;
		value["file_path"] = Json::Value(dentalPath);
		value["filename"] = Json::Value(filename.c_str());
		std::string strResult = value.toStyledString();
		// strResult = AsciiToUtf8(strResult);

		CURL* curl = nullptr;
		curl_global_init(CURL_GLOBAL_ALL);

		// 初始化easy handler句柄
		curl = curl_easy_init();
		if (nullptr == curl) {
			printf("failed to create curl connection.\n");
			return false;
		}

		// 设置post请求的url地址
		CURLcode code;
		code = curl_easy_setopt(curl, CURLOPT_URL, reqURL.c_str());
		if (code != CURLE_OK) {
			printf("failed to set url.\n");
			return false;
		}

		// 设置回调函数，如果不设置，默认输出到控制台
		code = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		if (code != CURLE_OK) {
			printf("failed to set write data.\n");
			return false;
		}

		//设置接收数据的处理函数和存放变量
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strResult.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&stBody);

		//HTTP报文头
		struct curl_slist* headers = nullptr;
		headers = curl_slist_append(headers, "Accept: application/json");
		headers = curl_slist_append(headers, "Content-Type: application/json"); //text/html
		headers = curl_slist_append(headers, "charsets: utf-8");
		// set method to post
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		//执行单条请求
		CURLcode res;
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			printf("curl_easy_perform failed! {}\n");
			return false;
		}
		result = stBody.response;

		curl_slist_free_all(headers);
		/* always cleanup */
		curl_easy_cleanup(curl);
		//在结束libcurl使用的时候，用来对curl_global_init做的工作清理。类似于close的函数
		curl_global_cleanup();

		return true;
	}


	bool MeshCNNHTTP::reqMeshCNNObjJson_(std::string IP, std::string Port, const char* objJson, std::string& result, std::string route) {
		std::string reqURL = "http://" + IP + ":" + Port + route;
		struct memory stBody;
		// json 数据
		Json::Value value;
		value["obj"] = Json::Value(objJson);
		std::string strResult = value.toStyledString();
		// strResult = AsciiToUtf8(strResult);

		CURL* curl = nullptr;
		curl_global_init(CURL_GLOBAL_ALL);

		// 初始化easy handler句柄
		curl = curl_easy_init();
		if (nullptr == curl) {
			printf("failed to create curl connection.\n");
			return false;
		}

		// 设置post请求的url地址
		CURLcode code;
		code = curl_easy_setopt(curl, CURLOPT_URL, reqURL.c_str());
		if (code != CURLE_OK) {
			printf("failed to set url.\n");
			return false;
		}

		// 设置回调函数，如果不设置，默认输出到控制台
		code = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		if (code != CURLE_OK) {
			printf("failed to set write data.\n");
			return false;
		}

		//设置接收数据的处理函数和存放变量
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strResult.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&stBody);

		//HTTP报文头
		struct curl_slist* headers = nullptr;
		headers = curl_slist_append(headers, "Accept: application/json");
		headers = curl_slist_append(headers, "Content-Type: application/json"); //text/html
		headers = curl_slist_append(headers, "charsets: utf-8");
		// set method to post
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		//执行单条请求
		CURLcode res;
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			printf("curl_easy_perform failed! {}\n");
			return false;
		}
		result = stBody.response;

		curl_slist_free_all(headers);
		/* always cleanup */
		curl_easy_cleanup(curl);
		//在结束libcurl使用的时候，用来对curl_global_init做的工作清理。类似于close的函数
		curl_global_cleanup();

		return true;
	}


	bool MeshCNNHTTP::parseResult(std::string result) {
		stringReplace(result, "\\\"", "\"");
		result = std::string(result, 1, result.size() - 2);

		Json::CharReaderBuilder readerBuild;
		Json::CharReader* reader(readerBuild.newCharReader());
		Json::Value rcvRes;
		JSONCPP_STRING jsonErrs;
		bool parseOK = reader->parse(result.c_str(), result.c_str() + result.size(), &rcvRes, &jsonErrs);

		delete reader;
		if (!parseOK) {
			std::cout << "Failed to parse the rcvRes!" << std::endl;
			return false;
		}
		std::cout << "Rcv:\n" << std::endl;

		// TODO 多维数组 多个闭环
		int rcvSize = rcvRes["text"].size();
		std::cout << rcvSize << std::endl;
		for (int i = 0; i < rcvSize; ++i) {
			std::cout << "x: " << rcvRes["text"][i][0] << " y: " << rcvRes["text"][i][1] << " z: " << rcvRes["text"][i][2] << std::endl;
		}
		return true;
	}
}