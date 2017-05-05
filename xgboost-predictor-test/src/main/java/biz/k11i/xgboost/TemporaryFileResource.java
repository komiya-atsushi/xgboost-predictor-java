package biz.k11i.xgboost;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public class TemporaryFileResource implements Closeable {
    private Map<String, Path> tempFiles = new HashMap<>();

    public Path getAsPath(String name) {
        if (tempFiles.containsKey(name)) {
            return tempFiles.get(name);
        }

        Path result = TestHelper.getResourceAsTemporaryFile(name);
        tempFiles.put(name, result);

        return result;
    }

    @Override
    public void close() throws IOException {
        for (Path path : tempFiles.values()) {
            try {
                Files.deleteIfExists(path);
            } catch (IOException ignore) {
            }
        }

        tempFiles.clear();
    }
}
