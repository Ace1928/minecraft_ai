
/* Model of the existing call-site order, not the full Wine driver. The exact
 * new predicate above is extracted from the patch; this wrapper only checks
 * that denial never reaches simulated physical-side operations. */
static BOOL modeled_grab_boundary(struct x11drv_thread_data *data)
{
    if (NtUserGetWindowThread(NtUserGetDesktopWindow(), NULL) == GetCurrentThreadId())
        return TRUE;
    if (!data) return FALSE;
    if (!clipping_focus_allows_grab(data)) return TRUE;
    if (keyboard_grabbed) return FALSE;
    map_calls++;
    resize_calls++;
    grab_calls++;
    return TRUE;
}

static void reset(void)
{
    ordinary_focus = FALSE;
    virtual_desktop = TRUE;
    keyboard_grabbed = FALSE;
    x_query_ok = TRUE;
    x_focus = root_window;
    foreground = 10;
    foreground_pid = current_pid = 32;
    foreground_tid = current_tid = 36;
    desktop_tid = 44;
    x_queries = context_queries = virtual_queries = foreground_queries = 0;
    lookup_queries = map_calls = resize_calls = grab_calls = 0;
}

static void denied(const char *name, struct x11drv_thread_data *data,
                   unsigned int expected_x_queries)
{
    assert(modeled_grab_boundary(data) == TRUE);
    assert(map_calls == 0 && resize_calls == 0 && grab_calls == 0);
    assert(x_queries == expected_x_queries);
    printf("PASS %s\n", name);
}

int main(void)
{
    int opaque_display;
    struct x11drv_thread_data data = {&opaque_display};
    struct x11drv_thread_data missing_display = {NULL};

    reset(); ordinary_focus = TRUE; virtual_desktop = FALSE; foreground = 0;
    assert(modeled_grab_boundary(&data) == TRUE);
    assert(grab_calls == 1 && map_calls == 1 && resize_calls == 1);
    assert(context_queries == 1 && virtual_queries == 0 && x_queries == 1);
    assert(foreground_queries == 0 && lookup_queries == 0);
    puts("PASS ordinary_focus_unchanged");

    reset();
    assert(modeled_grab_boundary(&data) == TRUE);
    assert(grab_calls == 1 && map_calls == 1 && resize_calls == 1);
    assert(x_queries == 1 && foreground_queries == 1 && lookup_queries == 1);
    puts("PASS exact_virtual_root_and_foreground_owner");

    reset(); virtual_desktop = FALSE;
    denied("not_virtual_desktop", &data, 1);
    reset(); x_focus = 999;
    denied("foreign_nonroot_xfocus", &data, 1);
    reset(); x_focus = 0;
    denied("none_xfocus", &data, 1);
    assert(context_queries == 0);
    reset(); x_focus = 1;
    denied("pointerroot_xfocus", &data, 1);
    reset();
    assert(clipping_focus_allows_grab(NULL) == FALSE);
    assert(modeled_grab_boundary(NULL) == FALSE);
    assert(map_calls == 0 && resize_calls == 0 && grab_calls == 0);
    assert(x_queries == 0 && context_queries == 0 && foreground_queries == 0);
    puts("PASS missing_thread_data_preserves_callsite_failure");
    reset();
    denied("missing_display", &missing_display, 0);
    assert(context_queries == 0);
    reset(); x_query_ok = FALSE; x_focus = root_window;
    denied("failed_xquery_even_with_root_output", &data, 1);
    assert(context_queries == 0 && virtual_queries == 0);
    assert(foreground_queries == 0 && lookup_queries == 0);
    reset(); foreground = 0;
    denied("null_foreground", &data, 1);
    assert(lookup_queries == 0);
    reset(); foreground_tid = 0;
    denied("failed_foreground_thread_lookup", &data, 1);
    reset(); foreground_pid = 64;
    denied("foreground_other_process", &data, 1);
    reset(); foreground_pid = 0;
    denied("foreground_zero_process", &data, 1);

    reset(); desktop_tid = current_tid;
    denied("existing_desktop_thread_exclusion", &data, 0);
    assert(context_queries == 0);
    reset(); keyboard_grabbed = TRUE;
    assert(modeled_grab_boundary(&data) == FALSE);
    assert(map_calls == 0 && resize_calls == 0 && grab_calls == 0);
    puts("PASS existing_keyboard_grab_refusal");
    puts("15 checks passed; predicate and modeled boundary only, not driver integration.");
    return 0;
}
