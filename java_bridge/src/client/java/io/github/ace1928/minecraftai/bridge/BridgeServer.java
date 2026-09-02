package io.github.ace1928.minecraftai.bridge;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.loader.api.FabricLoader;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.EnumSet;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Authenticated loopback control plane for one Minecraft client instance.
 *
 * <p>The server owns a second, local copy of the motor lease. Even if the
 * Python process wedges, lease expiry or connection loss invokes releaseAll()
 * inside the game client.</p>
 */
public final class BridgeServer implements AutoCloseable {
    private static final int PROTOCOL_VERSION = 1;
    private static final long WATCHDOG_PERIOD_MS = 25L;
    private static final long MAX_MESSAGE_CHARS = 65_536L;
    private static final Gson GSON = new Gson();

    private final InputAdapter inputAdapter;
    private final String instanceId;
    private final String token;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ScheduledExecutorService watchdog = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread thread = new Thread(r, "minecraft-ai-bridge-watchdog");
        thread.setDaemon(true);
        return thread;
    });

    private volatile Lease lease;
    private volatile ServerSocket serverSocket;
    private volatile Socket activeSocket;
    private volatile Path endpointPath;

    public BridgeServer(InputAdapter inputAdapter) {
        this.inputAdapter = inputAdapter;
        this.instanceId = UUID.randomUUID().toString();
        this.token = randomToken();
    }

    public String instanceId() {
        return instanceId;
    }

    public void start() throws IOException {
        if (!running.compareAndSet(false, true)) {
            return;
        }
        ServerSocket server = new ServerSocket(0, 8, InetAddress.getLoopbackAddress());
        server.setReuseAddress(true);
        serverSocket = server;
        writeEndpointDescriptor(server.getLocalPort());
        watchdog.scheduleAtFixedRate(
            this::watchdogTick,
            WATCHDOG_PERIOD_MS,
            WATCHDOG_PERIOD_MS,
            TimeUnit.MILLISECONDS
        );
        Thread acceptThread = new Thread(this::acceptLoop, "minecraft-ai-bridge-control");
        acceptThread.setDaemon(true);
        acceptThread.start();
    }

    private void acceptLoop() {
        while (running.get()) {
            try {
                Socket socket = serverSocket.accept();
                socket.setSoTimeout(1500);
                Socket previous = activeSocket;
                if (previous != null && !previous.isClosed()) {
                    socket.close();
                    continue;
                }
                activeSocket = socket;
                handleConnection(socket);
            } catch (IOException error) {
                if (running.get()) {
                    failClosed("accept-failure");
                }
            } finally {
                activeSocket = null;
                clearLease("connection-closed");
            }
        }
    }

    private void handleConnection(Socket socket) throws IOException {
        try (
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8)
            );
            BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8)
            )
        ) {
            send(writer, hello());
            JsonObject auth = read(reader);
            if (!authenticate(auth)) {
                sendError(writer, "unauthorized", "Authentication or instance identity failed", true);
                return;
            }
            sendAck(writer, 0L);

            while (running.get() && !socket.isClosed()) {
                JsonObject message = read(reader);
                String kind = string(message, "kind");
                switch (kind) {
                    case "lease_bind" -> handleLeaseBind(message, writer);
                    case "lease_clear" -> {
                        clearLease("remote-lease-clear");
                        sendAck(writer, 0L);
                    }
                    case "release_all" -> {
                        inputAdapter.releaseAll();
                        sendAck(writer, 0L);
                    }
                    case "input" -> handleInput(message, writer);
                    case "heartbeat" -> handleHeartbeat(message, writer);
                    default -> {
                        failClosed("unknown-command");
                        sendError(writer, "unknown_command", "Unsupported bridge command", true);
                        return;
                    }
                }
            }
        } catch (RuntimeException error) {
            failClosed("protocol-error");
            throw error;
        } finally {
            failClosed("connection-end");
        }
    }

    private JsonObject hello() {
        JsonObject hello = new JsonObject();
        hello.addProperty("kind", "hello");
        hello.addProperty("protocol_version", PROTOCOL_VERSION);
        hello.addProperty("nonce", randomToken());
        JsonObject identity = new JsonObject();
        identity.addProperty("edition", "java");
        identity.addProperty("version", minecraftVersion());
        identity.addProperty("instance_id", instanceId);
        identity.addProperty("process_id", ProcessHandle.current().pid());
        identity.addProperty("profile", "fabric");
        hello.add("identity", identity);
        JsonArray capabilities = new JsonArray();
        capabilities.add("window_identity");
        if (inputAdapter.liveCapable()) {
            capabilities.add("input");
        }
        hello.add("capabilities", capabilities);
        return hello;
    }

    private boolean authenticate(JsonObject message) {
        return "authenticate".equals(string(message, "kind"))
            && integer(message, "protocol_version") == PROTOCOL_VERSION
            && token.equals(string(message, "token"))
            && instanceId.equals(string(message, "expected_instance_id"));
    }

    private void handleLeaseBind(JsonObject message, BufferedWriter writer) throws IOException {
        if (!inputAdapter.liveCapable()) {
            clearLease("input-not-live");
            sendError(writer, "input_not_live", "Scoped input adapter has not passed release gates", true);
            return;
        }
        if (!instanceId.equals(string(message, "target_instance_id"))) {
            clearLease("target-mismatch");
            sendError(writer, "target_mismatch", "Lease targets another Minecraft instance", true);
            return;
        }
        long expires = longValue(message, "expires_monotonic_ns");
        long firstSequence = longValue(message, "first_sequence");
        if (expires <= System.nanoTime()) {
            clearLease("lease-already-expired");
            sendError(writer, "lease_expired", "Lease is already expired", true);
            return;
        }
        lease = new Lease(
            string(message, "lease_id"),
            expires,
            Math.max(0L, firstSequence - 1L),
            integer(message, "max_action_duration_ms")
        );
        inputAdapter.releaseAll();
        sendAck(writer, firstSequence);
    }

    private void handleInput(JsonObject message, BufferedWriter writer) throws IOException {
        Lease current = lease;
        long sequence = longValue(message, "sequence");
        if (current == null || !current.id().equals(string(message, "lease_id"))) {
            failClosed("missing-or-wrong-lease");
            sendError(writer, "invalid_lease", "No matching live motor lease", true);
            return;
        }
        long now = System.nanoTime();
        long deadline = longValue(message, "deadline_monotonic_ns");
        if (current.expiresNs() <= now || deadline <= now || deadline > current.expiresNs()) {
            failClosed("input-deadline-invalid");
            sendError(writer, "deadline_invalid", "Input deadline is stale or outside lease", true);
            return;
        }
        if (sequence <= current.lastSequence()) {
            failClosed("replay-or-out-of-order");
            sendError(writer, "sequence_invalid", "Input sequence is stale or replayed", true);
            return;
        }
        int duration = integer(message, "duration_ms");
        if (duration < 0 || duration > current.maxActionDurationMs()) {
            failClosed("duration-limit");
            sendError(writer, "duration_invalid", "Input action duration exceeds lease", true);
            return;
        }
        inputAdapter.apply(message);
        lease = current.withLastSequence(sequence);
        sendAck(writer, sequence);
    }

    private void handleHeartbeat(JsonObject message, BufferedWriter writer) throws IOException {
        Lease current = lease;
        if (current == null || current.expiresNs() <= System.nanoTime()) {
            failClosed("heartbeat-without-live-lease");
            sendError(writer, "lease_expired", "Heartbeat has no live lease", true);
            return;
        }
        sendAck(writer, longValue(message, "sequence"));
    }

    private void watchdogTick() {
        try {
            Lease current = lease;
            if (current != null && current.expiresNs() <= System.nanoTime()) {
                clearLease("local-watchdog-expired");
            }
        } catch (RuntimeException ignored) {
            failClosed("watchdog-failure");
        }
    }

    private void clearLease(String reason) {
        lease = null;
        try {
            inputAdapter.releaseAll();
        } catch (RuntimeException ignored) {
            // There is no safer recovery than continuing to revoke authority.
        }
    }

    private void failClosed(String reason) {
        clearLease(reason);
    }

    private void writeEndpointDescriptor(int port) throws IOException {
        Path directory = FabricLoader.getInstance().getGameDir().resolve(".minecraft-ai");
        Files.createDirectories(directory);
        Path path = directory.resolve("bridge-" + instanceId + ".json");
        JsonObject descriptor = new JsonObject();
        descriptor.addProperty("host", InetAddress.getLoopbackAddress().getHostAddress());
        descriptor.addProperty("port", port);
        descriptor.addProperty("token", token);
        descriptor.addProperty("instance_id", instanceId);
        descriptor.addProperty("edition", "java");
        descriptor.addProperty("version", minecraftVersion());
        descriptor.addProperty("process_id", ProcessHandle.current().pid());
        Files.writeString(path, GSON.toJson(descriptor), StandardCharsets.UTF_8);
        restrictPermissions(path);
        endpointPath = path;
    }

    private static void restrictPermissions(Path path) {
        try {
            Set<PosixFilePermission> permissions = EnumSet.of(
                PosixFilePermission.OWNER_READ,
                PosixFilePermission.OWNER_WRITE
            );
            Files.setPosixFilePermissions(path, permissions);
        } catch (UnsupportedOperationException | IOException ignored) {
            // Windows ACL hardening is handled by the installer/runtime adapter later.
        }
    }

    private static String minecraftVersion() {
        return FabricLoader.getInstance()
            .getModContainer("minecraft")
            .map(container -> container.getMetadata().getVersion().getFriendlyString())
            .orElse("unknown");
    }

    private static String randomToken() {
        byte[] bytes = new byte[32];
        new SecureRandom().nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    private static JsonObject read(BufferedReader reader) throws IOException {
        String line = reader.readLine();
        if (line == null) {
            throw new IOException("Bridge connection closed");
        }
        if (line.length() > MAX_MESSAGE_CHARS) {
            throw new IOException("Bridge message exceeds size limit");
        }
        return JsonParser.parseString(line).getAsJsonObject();
    }

    private static void send(BufferedWriter writer, JsonObject payload) throws IOException {
        writer.write(GSON.toJson(payload));
        writer.newLine();
        writer.flush();
    }

    private void sendAck(BufferedWriter writer, long sequence) throws IOException {
        JsonObject ack = new JsonObject();
        ack.addProperty("kind", "ack");
        ack.addProperty("protocol_version", PROTOCOL_VERSION);
        ack.addProperty("sequence", sequence);
        ack.addProperty("instance_id", instanceId);
        send(writer, ack);
    }

    private void sendError(
        BufferedWriter writer,
        String code,
        String message,
        boolean releaseAll
    ) throws IOException {
        if (releaseAll) {
            failClosed(code);
        }
        JsonObject error = new JsonObject();
        error.addProperty("kind", "error");
        error.addProperty("protocol_version", PROTOCOL_VERSION);
        error.addProperty("code", code);
        error.addProperty("message", message);
        error.addProperty("release_all", releaseAll);
        send(writer, error);
    }

    private static String string(JsonObject object, String key) {
        return object.has(key) ? object.get(key).getAsString() : "";
    }

    private static int integer(JsonObject object, String key) {
        return object.has(key) ? object.get(key).getAsInt() : 0;
    }

    private static long longValue(JsonObject object, String key) {
        return object.has(key) ? object.get(key).getAsLong() : 0L;
    }

    @Override
    public void close() {
        if (!running.compareAndSet(true, false)) {
            return;
        }
        failClosed("bridge-close");
        Socket socket = activeSocket;
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException ignored) {
            }
        }
        ServerSocket server = serverSocket;
        if (server != null) {
            try {
                server.close();
            } catch (IOException ignored) {
            }
        }
        watchdog.shutdownNow();
        Path path = endpointPath;
        if (path != null) {
            try {
                Files.deleteIfExists(path);
            } catch (IOException ignored) {
            }
        }
    }

    private record Lease(
        String id,
        long expiresNs,
        long lastSequence,
        int maxActionDurationMs
    ) {
        Lease withLastSequence(long sequence) {
            return new Lease(id, expiresNs, sequence, maxActionDurationMs);
        }
    }
}
