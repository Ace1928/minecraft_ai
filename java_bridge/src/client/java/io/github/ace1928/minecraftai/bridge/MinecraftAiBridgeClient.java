package io.github.ace1928.minecraftai.bridge;

import net.fabricmc.api.ClientModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/** Fabric client entrypoint. Live input remains disabled until an adapter passes gates. */
public final class MinecraftAiBridgeClient implements ClientModInitializer {
    public static final String MOD_ID = "minecraft-ai-bridge";
    private static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);
    private static volatile BridgeServer server;

    @Override
    public void onInitializeClient() {
        InputAdapter adapter = new NoopInputAdapter();
        BridgeServer bridge = new BridgeServer(adapter);
        try {
            bridge.start();
            server = bridge;
            Runtime.getRuntime().addShutdownHook(new Thread(
                bridge::close,
                "minecraft-ai-bridge-shutdown"
            ));
            LOGGER.info(
                "Minecraft AI bridge ready: instance={} liveInput={}",
                bridge.instanceId(),
                adapter.liveCapable()
            );
        } catch (IOException error) {
            bridge.close();
            LOGGER.error("Minecraft AI bridge failed to start; no control is available", error);
        }
    }

    public static void stopBridge() {
        BridgeServer bridge = server;
        server = null;
        if (bridge != null) {
            bridge.close();
        }
    }
}
