#pragma once

#ifdef _WIN32
    #define API_CALL __stdcall
    #ifndef _WIN64
        #error unsupported target OS
    #endif
#elif __linux__
    #define API_CALL
    #if (!defined __x86_64__) && (!defined __aarch64__)
        #error unsupported target OS
    #endif
#else
    #error unsupported target OS
#endif

#ifndef DLL_DECL
#define DLL_DECL
#endif

#ifdef __cplusplus
extern "C"
{
#endif

typedef void (*f_chatllm_print)(const char *utf8_str);
typedef void (*f_chatllm_end)(void);

struct chatllm_obj;

DLL_DECL struct chatllm_obj * API_CALL chatllm_create(void);

DLL_DECL void API_CALL chatllm_append_param(struct chatllm_obj *obj, const char *utf8_str);

DLL_DECL int API_CALL chatllm_start(struct chatllm_obj *obj, f_chatllm_print f_print, f_chatllm_end f_end);

DLL_DECL int API_CALL chatllm_user_input(struct chatllm_obj *obj, const char *utf8_str);

#ifdef __cplusplus
}
#endif