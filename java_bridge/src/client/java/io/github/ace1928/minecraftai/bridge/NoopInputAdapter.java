package io.github.ace1928.minecraftai.bridge;

import com.google.gson.JsonObject;

/** Safe default: proves discovery/auth/lease behavior without controlling Minecraft. */
public final class NoopInputAdapter implements InputAdapter {
    @Override
    public boolean liveCapable() {
        return false;
    }

    @Override
    public void apply(JsonObject command) {
        throw new IllegalStateException("Live scoped input adapter is not installed");
    }

    @Override
    public void releaseAll() {
        // Nothing can be held by this adapter.
    }
}
