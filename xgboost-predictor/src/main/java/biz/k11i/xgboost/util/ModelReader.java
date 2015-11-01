package biz.k11i.xgboost.util;

import java.io.Closeable;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Reads the Xgboost model from stream.
 */
public class ModelReader implements Closeable {
    private final DataInputStream stream;
    private byte[] buffer = new byte[8];

    @Deprecated
    public ModelReader(String filename) throws IOException {
        stream = new DataInputStream(new FileInputStream(filename));
    }

    public ModelReader(InputStream in) throws IOException {
        byte[] type = new byte[4];
        if (in.read(type) < 4) {
            throw new IOException("Cannot read format type (shortage)");
        }

        String typeString = new String(type);
        if (!typeString.equals("binf")) {
            // TODO support bs64
            throw new IOException("Unsupported format type: " + typeString);
        }

        stream = new DataInputStream(in);
    }

    public int readInt() throws IOException {
        int numBytesRead = stream.read(buffer, 0, 4);
        if (numBytesRead < 4) {
            throw new IOException("Cannot read int value (shortage): " + numBytesRead);
        }

        return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getInt();
    }

    public int[] readIntArray(int numValues) throws IOException {
        int[] result = new int[numValues];

        for (int i = 0; i < numValues; i++) {
            result[i] = readInt();
        }

        return result;
    }

    public int readUnsignedInt() throws IOException {
        int result = readInt();
        if (result < 0) {
            throw new IOException("Cannot read unsigned int (overflow): " + result);
        }

        return result;
    }

    public long readLong() throws IOException {
        int numBytesRead = stream.read(buffer, 0, 8);
        if (numBytesRead < 8) {
            throw new IOException("Cannot read long value (shortage): " + numBytesRead);
        }

        return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getLong();
    }

    public float readFloat() throws IOException {
        int numBytesRead = stream.read(buffer, 0, 4);
        if (numBytesRead < 4) {
            throw new IOException("Cannot read float value (shortage): " + numBytesRead);
        }

        return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getFloat();
    }

    public float[] readFloatArray(int numValues) throws IOException {
        float[] result = new float[numValues];

        for (int i = 0; i < numValues; i++) {
            result[i] = readFloat();
        }

        return result;
    }

    public void skip(long numBytes) throws IOException {
        long numBytesRead = stream.skip(numBytes);
        if (numBytesRead < numBytes) {
            throw new IOException("Cannot skip bytes: " + numBytesRead);
        }
    }

    public String readString() throws IOException {
        long length = readLong();
        if (length > Integer.MAX_VALUE) {
            throw new IOException("Too long string: " + length);
        }

        return readString((int) length);
    }

    public String readString(int numBytes) throws IOException {
        byte[] buffer = new byte[numBytes];
        int numBytesRead = stream.read(buffer, 0, numBytes);

        if (numBytesRead < numBytes) {
            throw new IOException(String.format("Cannot read string(%d) (shortage): %d", numBytes, numBytesRead));
        }

        return new String(buffer);
    }

    @Override
    public void close() throws IOException {
        stream.close();
    }
}
