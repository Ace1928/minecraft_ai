#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0600
#include <windows.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Observation only: no HWND creation, message pump, raw registration, input,
 * capture, focus, clipping setters or cursor warps. Use the existing game's
 * Wine loader/prefix/display and bracket startup externally with X queries.
 * Usage: win32_state_probe.exe GAME_HWND NEW_OUTPUT_PATH
 * Arguments are unquoted ASCII without spaces; HWND accepts decimal or 0x hex.
 * The GUI subsystem avoids a diagnostic console. A new file is required.
 */
int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR command, int show)
{
    char args[2048], *path, *end;
    unsigned long long handle;
    HWND game;
    HANDLE output;
    ULONGLONG start;
    DWORD expected_pid = 0, expected_tid = 0;
    unsigned int sample = 0;
    int failed = 0;
    (void)instance;
    (void)previous;
    (void)show;
    if (!command || !*command || strlen(command) >= sizeof(args) ||
        strpbrk(command, "\"\r\n\t")) return 64;
    strcpy(args, command);
    path = strchr(args, ' ');
    if (!path || path == args || !path[1]) return 64;
    *path++ = 0;
    if (strchr(path, ' ') || args[0] == '-' || args[0] == '+') return 64;
    errno = 0;
    handle = strtoull(args, &end, 0);
    if (errno || *end || !handle || handle != (unsigned long long)(ULONG_PTR)handle)
        return 64;
    game = (HWND)(ULONG_PTR)handle;
    output = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, NULL,
                         CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
    if (output == INVALID_HANDLE_VALUE) return 73;
    start = GetTickCount64();
    do
    {
        SYSTEMTIME utc;
        GUITHREADINFO gti = {0};
        POINT cursor = {0, 0};
        RECT clip = {0, 0, 0, 0};
        DWORD pid = 0, tid, written;
        DWORD thread_error = 0, gui_error = 0, clip_error = 0, cursor_error = 0;
        HWND foreground, foreground_after, minecraft_by_title;
        BOOL identity_match;
        BOOL gui_ok = FALSE, clip_ok, cursor_ok;
        ULONGLONG elapsed, remaining;
        char line[2048];
        int length;
        GetSystemTime(&utc);
        elapsed = GetTickCount64() - start;
        foreground = GetForegroundWindow();
        minecraft_by_title = FindWindowA(NULL, "Minecraft");
        identity_match = minecraft_by_title == game;
        SetLastError(0);
        tid = GetWindowThreadProcessId(game, &pid);
        if (!tid) thread_error = GetLastError();
        if (!sample && tid && identity_match)
        {
            expected_pid = pid;
            expected_tid = tid;
        }
        identity_match = identity_match && pid == expected_pid &&
                         tid == expected_tid && tid != 0;
        gti.cbSize = sizeof(gti);
        /* Never pass zero: that would silently query the foreground thread. */
        if (tid && identity_match)
        {
            SetLastError(0);
            gui_ok = GetGUIThreadInfo(tid, &gti);
            if (!gui_ok) gui_error = GetLastError();
        }
        SetLastError(0);
        clip_ok = GetClipCursor(&clip);
        if (!clip_ok) clip_error = GetLastError();
        SetLastError(0);
        cursor_ok = GetCursorPos(&cursor);
        if (!cursor_ok) cursor_error = GetLastError();
        foreground_after = GetForegroundWindow();
        length = snprintf(line, sizeof(line),
            "{\"kind\":\"win32_state\",\"sample\":%u,"
            "\"utc\":\"%04u-%02u-%02uT%02u:%02u:%02u.%03uZ\","
            "\"tick_ms\":%llu,\"elapsed_ms\":%llu,\"query_pid\":%lu,"
            "\"game_hwnd\":%llu,\"game_pid\":%lu,\"game_tid\":%lu,\"thread_error\":%lu,"
            "\"minecraft_title_hwnd\":%llu,\"identity_match\":%s,"
            "\"foreground_hwnd\":%llu,\"foreground_after_hwnd\":%llu,"
            "\"gui_ok\":%s,\"gui_error\":%lu,\"hwndActive\":%llu,"
            "\"hwndFocus\":%llu,\"hwndCapture\":%llu,\"gui_flags\":%lu,"
            "\"clip_ok\":%s,\"clip_error\":%lu,\"clip\":[%ld,%ld,%ld,%ld],"
            "\"cursor_ok\":%s,\"cursor_error\":%lu,\"cursor\":[%ld,%ld]}\n",
            sample, utc.wYear, utc.wMonth, utc.wDay, utc.wHour, utc.wMinute,
            utc.wSecond, utc.wMilliseconds, (unsigned long long)GetTickCount64(),
            (unsigned long long)elapsed, (unsigned long)GetCurrentProcessId(),
            handle, (unsigned long)pid, (unsigned long)tid, (unsigned long)thread_error,
            (unsigned long long)(ULONG_PTR)minecraft_by_title,
            identity_match ? "true" : "false",
            (unsigned long long)(ULONG_PTR)foreground,
            (unsigned long long)(ULONG_PTR)foreground_after,
            gui_ok ? "true" : "false", (unsigned long)gui_error,
            (unsigned long long)(ULONG_PTR)gti.hwndActive,
            (unsigned long long)(ULONG_PTR)gti.hwndFocus,
            (unsigned long long)(ULONG_PTR)gti.hwndCapture, (unsigned long)gti.flags,
            clip_ok ? "true" : "false", (unsigned long)clip_error,
            (long)clip.left, (long)clip.top, (long)clip.right, (long)clip.bottom,
            cursor_ok ? "true" : "false", (unsigned long)cursor_error,
            (long)cursor.x, (long)cursor.y);
        if (length < 0 || length >= (int)sizeof(line) ||
            !WriteFile(output, line, (DWORD)length, &written, NULL) || written != (DWORD)length)
        {
            failed = 1;
            break;
        }
        if (!tid || !gui_ok || !clip_ok || !cursor_ok) failed = 1;
        if (!identity_match) { failed = 1; break; }
        ++sample;
        elapsed = GetTickCount64() - start;
        if (elapsed >= 3000) break;
        remaining = 3000 - elapsed;
        /* Never burst to catch up after a slow query. */
        Sleep((DWORD)(remaining < 100 ? remaining : 100));
    } while (GetTickCount64() - start < 3000);
    if (!CloseHandle(output)) failed = 1;
    return failed;
}
