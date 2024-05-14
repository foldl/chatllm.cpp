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
    #define API_CALL
    #warning OS not supported, maybe
#endif

#ifndef DLL_DECL
#define DLL_DECL
#endif

#ifdef __cplusplus
extern "C"
{
#endif

enum PrintType
{
    PRINT_CHAT_CHUNK        = 0,
    PRINTLN_META            = 1,    // print a whole line: general information
    PRINTLN_ERROR           = 2,    // print a whole line: error message
    PRINTLN_REF             = 3,    // print a whole line: reference
    PRINTLN_REWRITTEN_QUERY = 4,    // print a whole line: rewritten query
};

typedef void (*f_chatllm_print)(void *user_data, int print_type, const char *utf8_str);
typedef void (*f_chatllm_end)(void *user_data);

struct chatllm_obj;

/**
 * Usage:
 *
 * ```c
 * obj = create(callback functions);
 * append_param(obj, ...);
 * // ...
 * app_param(obj, ...);
 *
 * start(obj);
 * while (true)
 * {
 *     user_input(obj, ...);
 * }
 * ```
*/

// Create ChatLLM object
DLL_DECL struct chatllm_obj * API_CALL chatllm_create(void);

// Append a command line option
DLL_DECL void API_CALL chatllm_append_param(struct chatllm_obj *obj, const char *utf8_str);

// Start
DLL_DECL int API_CALL chatllm_start(struct chatllm_obj *obj, f_chatllm_print f_print, f_chatllm_end f_end, void *user_data);

// Restart (i.e. discard history)
DLL_DECL void API_CALL chatllm_restart(struct chatllm_obj *obj);

// User input
DLL_DECL int API_CALL chatllm_user_input(struct chatllm_obj *obj, const char *utf8_str);

// Abort generation
DLL_DECL void API_CALL chatllm_abort_generation(struct chatllm_obj *obj);

#ifdef __cplusplus
}
#endif