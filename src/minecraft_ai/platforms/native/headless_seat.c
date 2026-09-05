/* SPDX-License-Identifier: Apache-2.0
 * A seat with no physical devices, parent compositor, or input protocol.
 * Xwayland needs a keyboard-capable seat for its window manager to retain
 * focus. Minecraft AI injects only through the private XTEST connection.
 */
#include <stdlib.h>
#include <libweston/libweston.h>
#include <libweston/version.h>

#if WESTON_VERSION_MAJOR != 13 || WESTON_VERSION_MINOR != 0 || WESTON_VERSION_MICRO != 0
#error "Requalify the virtual seat ABI before building for another Weston version"
#endif

/* Exported libweston 13 backend APIs; declarations from its pinned
 * libweston/libweston-internal.h, absent from the installed public header.
 */
void weston_seat_init(struct weston_seat *, struct weston_compositor *, const char *);
int weston_seat_init_pointer(struct weston_seat *);
int weston_seat_init_keyboard(struct weston_seat *, struct xkb_keymap *);
void weston_seat_release(struct weston_seat *);

struct ai_seat {
    struct weston_seat seat;
    struct wl_listener destroy;
};

static void destroy_seat(struct wl_listener *listener, void *data)
{
    struct ai_seat *state = wl_container_of(listener, state, destroy);
    (void)data;
    wl_list_remove(&state->destroy.link);
    weston_seat_release(&state->seat);
    free(state);
}

WL_EXPORT int wet_module_init(struct weston_compositor *compositor, int *argc, char *argv[])
{
    int major, minor, micro;
    struct ai_seat *state;
    (void)argc;
    (void)argv;

    weston_version(&major, &minor, &micro);
    if (major != 13 || minor != 0 || micro != 0 ||
        !wl_list_empty(&compositor->seat_list)) {
        weston_log("minecraft-ai: refusing incompatible or already seated compositor\n");
        return -1;
    }
    state = calloc(1, sizeof *state);
    if (!state)
        return -1;
    weston_seat_init(&state->seat, compositor, "minecraft-ai-virtual");
    if (weston_seat_init_pointer(&state->seat) < 0 ||
        weston_seat_init_keyboard(&state->seat, NULL) < 0) {
        weston_seat_release(&state->seat);
        free(state);
        return -1;
    }
    if (!weston_compositor_add_destroy_listener_once(compositor, &state->destroy,
                                                    destroy_seat)) {
        weston_seat_release(&state->seat);
        free(state);
        return -1;
    }
    weston_log("minecraft-ai: virtual seat initialized without a host input transport\n");
    return 0;
}
