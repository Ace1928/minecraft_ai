/* Native stub test: no Wine, Xlib, display sockets, processes, or GUI calls. */
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef int BOOL;
typedef unsigned long Window;
typedef uint32_t DWORD;
typedef uintptr_t HWND;
#define TRUE 1
#define FALSE 0
struct x11drv_thread_data { void *display; };
static Window root_window = 100;
static BOOL ordinary_focus, virtual_desktop, keyboard_grabbed, x_query_ok;
static int winContext = 7;
static Window x_focus;
static HWND foreground;
static DWORD foreground_pid, foreground_tid, current_pid, current_tid, desktop_tid;
static unsigned int x_queries, context_queries, virtual_queries, foreground_queries;
static unsigned int lookup_queries, map_calls, resize_calls, grab_calls;
static BOOL is_virtual_desktop(void) { virtual_queries++; return virtual_desktop; }
static int XGetInputFocus(void *display, Window *focus, int *revert)
{
    assert(display != NULL);
    x_queries++;
    *focus = x_focus;
    *revert = 0;
    return x_query_ok;
}
static int XFindContext(void *display, Window window, int context, char **hwnd)
{
    assert(display != NULL && window == x_focus && context == winContext && hwnd != NULL);
    context_queries++;
    /* The predicate never reads the returned HWND. Only context presence matters. */
    return ordinary_focus ? 0 : 1;
}
static HWND NtUserGetForegroundWindow(void) { foreground_queries++; return foreground; }
static DWORD NtUserGetWindowThread(HWND hwnd, DWORD *pid)
{
    if (hwnd == 1 && pid == NULL) return desktop_tid;
    assert(hwnd == foreground);
    lookup_queries++;
    if (pid) *pid = foreground_pid;
    return foreground_tid;
}
static HWND NtUserGetDesktopWindow(void) { return 1; }
static DWORD GetCurrentProcessId(void) { return current_pid; }
static DWORD GetCurrentThreadId(void) { return current_tid; }
