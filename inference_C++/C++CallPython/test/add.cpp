#include <python3.6m/Python.h>
#include <iostream>


int main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], nullptr);
    if ( program == nullptr ){
        std::cout << "Fatal Error: cannot decode argv[0]!" << std::endl;
        return -1;
    }
    Py_SetProgramName(program);
    Py_Initialize();
    if ( !Py_IsInitialized() ){
        std::cout << "Python init failed!" << std::endl;
	return 0;
    }
    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/heygears/work/C++_python/test/test_add')");
    
    PyObject* pModule = PyImport_ImportModule("add");
    if ( pModule == nullptr ){
        std::cout << "module not found!" << std::endl;
        return 1;
    }
    
    PyObject* pFunc = PyObject_GetAttrString(pModule, "add_num");
    if ( !pFunc || !PyCallable_Check(pFunc) ){
        std::cout << "not found function add_num!" << std::endl;
        return 2;
    }
    
    PyObject* args = Py_BuildValue("(ii)", 28, 103);
    PyObject* pRet = PyObject_CallObject(pFunc, args);
    Py_DECREF(args);
     
    int res = 0;
    PyArg_Parse(pRet, "i", &res);
    Py_DECREF(pRet);
    std::cout << "the res is: " << res << std::endl;

    if ( Py_FinalizeEx() < 0 ){
        exit(120);
    }
    PyMem_RawFree(program);
    return 0;
}
