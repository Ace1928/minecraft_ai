package io.github.ace1928.minecraftai.bridge;

import com.google.gson.JsonObject;

/**
 * Version-specific adapter that applies human-style input semantics inside one
 * Minecraft client. The protocol/server layer never reaches directly into game
 * internals; only a tested adapter may do that.
 */
public interface InputAdapter {
    /** True only after this adapter has passed the scoped-input release gate. */
    boolean liveCapable();

    /** Apply one already-authenticated and lease-validated input command. */
    void apply(JsonObject command);

    /** Release every key/button state this adapter can hold. Must be idempotent. */
    void releaseAll();
}
