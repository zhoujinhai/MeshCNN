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
		// Ԥ��-�������п��ֽڵĳ���  
		int unicodeLen = MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, nullptr, 0);
		// ��ָ�򻺳�����ָ����������ڴ�  
		wchar_t* pUnicode = (wchar_t*)malloc(sizeof(wchar_t) * unicodeLen);
		// ��ʼ�򻺳���ת���ֽ�  
		MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, pUnicode, unicodeLen);
		std::wstring ret_str = pUnicode;
		free(pUnicode);
		return ret_str;
	}


	std::string MeshCNNHTTP::UnicodeToUtf8(const std::wstring& wstr)
	{
		// Ԥ��-�������ж��ֽڵĳ���  
		int ansiiLen = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
		// ��ָ�򻺳�����ָ����������ڴ�  
		char* pAssii = (char*)malloc(sizeof(char) * ansiiLen);
		// ��ʼ�򻺳���ת���ֽ�  
		WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, pAssii, ansiiLen, nullptr, nullptr);
		std::string ret_str = pAssii;
		free(pAssii);
		return ret_str;
	}


	std::string MeshCNNHTTP::AsciiToUtf8(const std::string& str)
	{
		return UnicodeToUtf8(AsciiToUnicode(str));
	}*/


	//�ص�����
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

		// ��ʼ��easy handler���
		curl = curl_easy_init();
		if (nullptr == curl) {
			printf("failed to create curl connection.\n");
			return false;
		}

		// ����post�����url��ַ
		CURLcode code;
		code = curl_easy_setopt(curl, CURLOPT_URL, reqURL.c_str());
		if (code != CURLE_OK) {
			printf("failed to set url.\n");
			return false;
		}

		// ���ûص���������������ã�Ĭ�����������̨
		code = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		if (code != CURLE_OK) {
			printf("failed to set write data.\n");
			return false;
		}

		//���ý������ݵĴ������ʹ�ű���
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&stBody);

		//��ʼ��form
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

		//HTTP����ͷ
		struct curl_slist* headers = nullptr;
		/*set header info��You can set more than one bar*/
		headers = curl_slist_append(headers, "Accept-Encoding: gzip, deflate");
		headers = curl_slist_append(headers, "User-Agent: curl");
		/* Pass in the HTTP header*/
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		//ִ�е�������
		CURLcode res;
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
			printf("curl_easy_perform failed! {}\n");
			return false;
		}
		result = stBody.response;

		// �ͷ���Դ
		curl_slist_free_all(headers);
		/* �ͷ�form */
		curl_mime_free(form);
		/* always cleanup */
		curl_easy_cleanup(curl);
		/*��curl_global_init���Ĺ������� ������close�ĺ���*/
		curl_global_cleanup();

		return true;
	}


	bool MeshCNNHTTP::reqMeshCNNFilePath_(std::string IP, std::string Port, const char* dentalPath, std::string& result, std::string route) {
		std::string reqURL = "http://" + IP + ":" + Port + route;
		struct memory stBody;
		// json ����
		std::string filename = getPathShortName(dentalPath);

		Json::Value value;
		value["file_path"] = Json::Value(dentalPath);
		value["filename"] = Json::Value(filename.c_str());
		std::string strResult = value.toStyledString();
		// strResult = AsciiToUtf8(strResult);

		CURL* curl = nullptr;
		curl_global_init(CURL_GLOBAL_ALL);

		// ��ʼ��easy handler���
		curl = curl_easy_init();
		if (nullptr == curl) {
			printf("failed to create curl connection.\n");
			return false;
		}

		// ����post�����url��ַ
		CURLcode code;
		code = curl_easy_setopt(curl, CURLOPT_URL, reqURL.c_str());
		if (code != CURLE_OK) {
			printf("failed to set url.\n");
			return false;
		}

		// ���ûص���������������ã�Ĭ�����������̨
		code = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		if (code != CURLE_OK) {
			printf("failed to set write data.\n");
			return false;
		}

		//���ý������ݵĴ������ʹ�ű���
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strResult.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&stBody);

		//HTTP����ͷ
		struct curl_slist* headers = nullptr;
		headers = curl_slist_append(headers, "Accept: application/json");
		headers = curl_slist_append(headers, "Content-Type: application/json"); //text/html
		headers = curl_slist_append(headers, "charsets: utf-8");
		// set method to post
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		//ִ�е�������
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
		//�ڽ���libcurlʹ�õ�ʱ��������curl_global_init���Ĺ�������������close�ĺ���
		curl_global_cleanup();

		return true;
	}


	bool MeshCNNHTTP::reqMeshCNNObjJson_(std::string IP, std::string Port, const char* objJson, std::string& result, std::string route) {
		std::string reqURL = "http://" + IP + ":" + Port + route;
		struct memory stBody;
		// json ����
		Json::Value value;
		value["obj"] = Json::Value(objJson);
		std::string strResult = value.toStyledString();
		// strResult = AsciiToUtf8(strResult);

		CURL* curl = nullptr;
		curl_global_init(CURL_GLOBAL_ALL);

		// ��ʼ��easy handler���
		curl = curl_easy_init();
		if (nullptr == curl) {
			printf("failed to create curl connection.\n");
			return false;
		}

		// ����post�����url��ַ
		CURLcode code;
		code = curl_easy_setopt(curl, CURLOPT_URL, reqURL.c_str());
		if (code != CURLE_OK) {
			printf("failed to set url.\n");
			return false;
		}

		// ���ûص���������������ã�Ĭ�����������̨
		code = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
		if (code != CURLE_OK) {
			printf("failed to set write data.\n");
			return false;
		}

		//���ý������ݵĴ������ʹ�ű���
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strResult.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&stBody);

		//HTTP����ͷ
		struct curl_slist* headers = nullptr;
		headers = curl_slist_append(headers, "Accept: application/json");
		headers = curl_slist_append(headers, "Content-Type: application/json"); //text/html
		headers = curl_slist_append(headers, "charsets: utf-8");
		// set method to post
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

		//ִ�е�������
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
		//�ڽ���libcurlʹ�õ�ʱ��������curl_global_init���Ĺ�������������close�ĺ���
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

		// TODO ��ά���� ����ջ�
		int rcvSize = rcvRes["text"].size();
		std::cout << rcvSize << std::endl;
		for (int i = 0; i < rcvSize; ++i) {
			std::cout << "x: " << rcvRes["text"][i][0] << " y: " << rcvRes["text"][i][1] << " z: " << rcvRes["text"][i][2] << std::endl;
		}
		return true;
	}
}