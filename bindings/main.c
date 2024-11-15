#include <stdio.h>
#include <string.h>

#include "libchatllm.h"

static void chatllm_print(void *user_data, int print_type, const char *utf8_str)
{
    switch (print_type)
    {
    case PRINT_CHAT_CHUNK:
        printf("%s", utf8_str);
        break;

    default:
        if (utf8_str)
            printf("%s\n", utf8_str);
        break;
    }

    fflush(stdout);
}

static void chatllm_end(void *user_data)
{
    printf("\n");
}

// Note: To keep thing simple, only ANSI strings are supported.

int main(int argc, const char **argv)
{
    struct chatllm_obj *chat = chatllm_create();
    for (int c = 1; c < argc; c++)
        chatllm_append_param(chat, argv[c]);

    int r = chatllm_start(chat, chatllm_print, chatllm_end, (void *)0);
    if (r != 0)
    {
        printf(">>> chatllm_start error: %d\n", r);
        return r;
    }

    while (1)
    {
        char input[2048];
        printf("You  > ");

        fgets(input, sizeof(input) - 1, stdin);
        if (strlen(input) < 1) continue;

        printf("A.I. > ");
        r = chatllm_user_input(chat, input);
        if (r != 0)
        {
            printf(">>> chatllm_user_input error: %d\n", r);
            break;
        }
    }
}
