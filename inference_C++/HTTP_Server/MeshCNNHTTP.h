#pragma once

#include <iostream>
#include <string>


namespace core {
	class MeshCNNHTTP {
	public:
		MeshCNNHTTP(std::string IP = "127.0.0.1", std::string Port = "8000");
		~MeshCNNHTTP();

		bool reqMeshCNNFileUpload(const char* dentalPath, std::string& result, std::string route = "/mesh/recognize");
		bool reqMeshCNNFilePath(const char* dentalPath, std::string& result, std::string route = "/mesh/recognize");
		bool reqMeshCNNObjJson(const char* objJsonData, std::string& result, std::string route = "/mesh/recognize");

		static void stringReplace(std::string& strOri, const std::string& strsrc, const std::string& strdst);

	private:
		struct memory
		{
			char* response;
			size_t size;
			memory()
			{
				response = NULL;
				size = 0;
			}
		};

		std::string getPathShortName(std::string strFullName);
		// std::wstring AsciiToUnicode(const std::string& str);
		// std::string UnicodeToUtf8(const std::wstring& wstr);
		// std::string AsciiToUtf8(const std::string& str);
		static size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream);
		bool reqMeshCNNFilePath_(std::string IP, std::string Port, const char* dentalPath, std::string& result, std::string route = "/mesh/recognize");    // file_path by json
		bool reqMeshCNNObjJson_(std::string IP, std::string Port, const char* objJson, std::string& result, std::string route = "/mesh/recognize");        // file_data by json
		bool reqMeshCNNFileUpload_(std::string IP, std::string Port, const char* dentalPath, std::string& result, std::string route = "/mesh/recognize");  // upload file 
		bool parseResult(std::string result);

	private:
		std::string IP;
		std::string Port;
	};
}