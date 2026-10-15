package io.github.jaelee18.mvtree.core

import com.github.mreutegg.laszip4j.LASReader
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile

/**
 * A single-tree point cloud: XYZ in metres, Z up, as float32-rounded values.
 *
 * Coordinates are rounded to float32 on purpose, exactly like `mvtree.io`: the
 * training renders were made from float32 clouds, and the renderer's min-max
 * normalisation is sensitive to that rounding at pixel boundaries.
 */
class PointCloud(val xyz: FloatArray) {
    val size: Int get() = xyz.size / 3

    /** Z extent in metres (the model's `tree_height` input). */
    val height: Double
        get() {
            var lo = Double.POSITIVE_INFINITY
            var hi = Double.NEGATIVE_INFINITY
            for (i in 0 until size) {
                val z = xyz[3 * i + 2].toDouble()
                if (z < lo) lo = z
                if (z > hi) hi = z
            }
            return hi - lo
        }
}

/** Point-cloud formats the app reads, matching `mvtree.io.SUPPORTED_EXTENSIONS`. */
object PointClouds {
    val EXTENSIONS = listOf("las", "laz", "ply", "pcd", "npy", "npz", "xyz", "txt", "csv", "pts")

    fun extension(name: String): String = name.substringAfterLast('.', "").lowercase()

    fun isSupported(name: String): Boolean = extension(name) in EXTENSIONS

    /**
     * Guess the format from the first bytes, for files shared without a usable name
     * (some apps hand over `content://…/1234`). Returns an extension, or null.
     */
    fun sniff(head: ByteArray): String? {
        fun startsWith(s: String) = head.size >= s.length && String(head, 0, s.length, Charsets.ISO_8859_1) == s
        if (startsWith("LASF")) return "las"  // LAS or LAZ: readLas checks the compression bit
        if (startsWith("ply\n") || startsWith("ply\r")) return "ply"
        if (head.size >= 6 && head[0] == 0x93.toByte() && startsWith("\u0093NUMPY")) return "npy"
        if (startsWith("PK\u0003\u0004")) return "npz"
        val text = String(head, Charsets.ISO_8859_1)
        // PCD: a text header (FIELDS ... DATA ascii|binary), possibly followed by binary points.
        val header = text.lines().map { it.trim().uppercase() }
        if (header.any { it.startsWith("FIELDS ") } && header.any { it.startsWith("DATA ") }) return "pcd"
        if (text.any { it.code < 9 || it.code in 14..31 }) return null  // binary, but not a known format
        val lines = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
        val numeric = Regex("^[-+0-9.eE]+([ ,\\t]+[-+0-9.eE]+){2,}")
        val data = lines.dropLast(1).filter { numeric.containsMatchIn(it) }  // the last line may be cut off
        if (data.isEmpty()) return null
        return if (data.all { it.contains(',') }) "csv" else "xyz"
    }

    /** Read any supported format; the file name decides the parser. */
    fun read(file: File, name: String = file.name): PointCloud = when (val ext = extension(name)) {
        "las", "laz" -> readLas(file)
        "ply" -> readPly(file)
        "pcd" -> readPcd(file)
        "npy" -> file.inputStream().use { readNpy(it, name) }
        "npz" -> readNpz(file)
        "xyz", "txt", "csv", "pts" -> readText(file)
        else -> throw IllegalArgumentException(
            "unsupported point-cloud format '.$ext'; use one of ${EXTENSIONS.joinToString(" ") { ".$it" }}"
        )
    }

    /** Collect doubles, round to float32, and drop non-finite rows (`mvtree.io._as_xyz`). */
    private class Builder(capacity: Int = 1 shl 16) {
        private var data = FloatArray(maxOf(capacity, 3) * 3)
        private var n = 0

        fun add(x: Double, y: Double, z: Double) {
            val fx = x.toFloat()
            val fy = y.toFloat()
            val fz = z.toFloat()
            if (!fx.isFinite() || !fy.isFinite() || !fz.isFinite()) return
            if (3 * n + 3 > data.size) data = data.copyOf(data.size * 2)
            data[3 * n] = fx
            data[3 * n + 1] = fy
            data[3 * n + 2] = fz
            n++
        }

        fun build(source: String): PointCloud {
            if (n < 3) throw IllegalArgumentException("$source: fewer than 3 finite points")
            return PointCloud(data.copyOf(3 * n))
        }
    }

    // ---------------------------------------------------------------- LAS / LAZ
    /** LAS 1.0-1.4 (any point format) natively; LAZ through laszip4j. x = X * scale + offset, like laspy. */
    fun readLas(file: File): PointCloud {
        val head = ByteArray(375)
        val got = file.inputStream().use { it.readNBytes(head, 0, head.size) }
        val buf = ByteBuffer.wrap(head, 0, got).order(ByteOrder.LITTLE_ENDIAN)
        require(got >= 227 && String(head, 0, 4, Charsets.US_ASCII) == "LASF") { "${file.name}: not a LAS/LAZ file" }
        val format = head[104].toInt() and 0xFF
        val compressed = format and 0x80 != 0 || format and 0x40 != 0
        return if (compressed) readLaz(file) else readLasRaw(file, buf)
    }

    private fun readLasRaw(file: File, header: ByteBuffer): PointCloud {
        val minor = header.get(25).toInt()
        val offsetToPoints = header.getInt(96).toLong() and 0xFFFFFFFFL
        val recordLength = header.getShort(105).toInt() and 0xFFFF
        var count = header.getInt(107).toLong() and 0xFFFFFFFFL
        if (minor >= 4 && header.limit() >= 255) {
            val extended = header.getLong(247)
            if (extended > 0) count = extended
        }
        val sx = header.getDouble(131)
        val sy = header.getDouble(139)
        val sz = header.getDouble(147)
        val ox = header.getDouble(155)
        val oy = header.getDouble(163)
        val oz = header.getDouble(171)
        require(recordLength >= 12) { "${file.name}: invalid point record length $recordLength" }
        val builder = Builder(count.coerceAtMost(50_000_000).toInt())
        file.inputStream().buffered(1 shl 20).use { input ->
            input.skipNBytes(offsetToPoints)
            val record = ByteArray(recordLength)
            val rb = ByteBuffer.wrap(record).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until count) {
                if (input.readNBytes(record, 0, recordLength) < recordLength) break
                builder.add(rb.getInt(0) * sx + ox, rb.getInt(4) * sy + oy, rb.getInt(8) * sz + oz)
            }
        }
        return builder.build(file.name)
    }

    private fun readLaz(file: File): PointCloud {
        val reader = LASReader(file)
        val h = reader.header
        val builder = Builder(h.numberOfPointRecords.coerceAtMost(50_000_000).toInt())
        reader.closeablePoints.use { points ->
            for (p in points) {
                builder.add(p.x * h.xScaleFactor + h.xOffset, p.y * h.yScaleFactor + h.yOffset,
                    p.z * h.zScaleFactor + h.zOffset)
            }
        }
        return builder.build(file.name)
    }

    // ---------------------------------------------------------------- PLY
    private enum class Scalar(val bytes: Int) { I8(1), U8(1), I16(2), U16(2), I32(4), U32(4), F32(4), F64(8) }

    private fun scalar(type: String): Scalar = when (type) {
        "char", "int8" -> Scalar.I8
        "uchar", "uint8" -> Scalar.U8
        "short", "int16" -> Scalar.I16
        "ushort", "uint16" -> Scalar.U16
        "int", "int32" -> Scalar.I32
        "uint", "uint32" -> Scalar.U32
        "float", "float32" -> Scalar.F32
        "double", "float64" -> Scalar.F64
        else -> throw IllegalArgumentException("PLY: unknown property type '$type'")
    }

    private fun ByteBuffer.read(type: Scalar, at: Int): Double = when (type) {
        Scalar.I8 -> get(at).toDouble()
        Scalar.U8 -> (get(at).toInt() and 0xFF).toDouble()
        Scalar.I16 -> getShort(at).toDouble()
        Scalar.U16 -> (getShort(at).toInt() and 0xFFFF).toDouble()
        Scalar.I32 -> getInt(at).toDouble()
        Scalar.U32 -> (getInt(at).toLong() and 0xFFFFFFFFL).toDouble()
        Scalar.F32 -> getFloat(at).toDouble()
        Scalar.F64 -> getDouble(at)
    }

    private fun readHeaderLines(input: InputStream, end: (String) -> Boolean): List<String> {
        val lines = ArrayList<String>()
        val sb = StringBuilder()
        while (true) {
            val c = input.read()
            if (c < 0) throw IllegalArgumentException("unexpected end of file in header")
            if (c == '\n'.code) {
                val line = sb.toString().trim()
                sb.setLength(0)
                lines.add(line)
                if (end(line)) return lines
            } else if (c != '\r'.code) {
                sb.append(c.toChar())
            }
            if (lines.size > 10_000) throw IllegalArgumentException("header too long")
        }
    }

    /** ASCII and binary (little/big endian) PLY; reads x, y, z of the `vertex` element. */
    fun readPly(file: File): PointCloud = BufferedInputStream(file.inputStream(), 1 shl 20).use { input ->
        val header = readHeaderLines(input) { it == "end_header" }
        require(header.firstOrNull() == "ply") { "${file.name}: not a PLY file" }
        val format = header.first { it.startsWith("format ") }.split(Regex("\\s+"))[1]
        // Elements in file order, each with its properties.
        data class Prop(val name: String, val type: Scalar?, val listCount: Scalar? = null, val listItem: Scalar? = null)
        data class Element(val name: String, val count: Long, val props: MutableList<Prop> = mutableListOf())
        val elements = mutableListOf<Element>()
        for (line in header) {
            val t = line.split(Regex("\\s+"))
            when (t[0]) {
                "element" -> elements.add(Element(t[1], t[2].toLong()))
                "property" -> elements.last().props.add(
                    if (t[1] == "list") Prop(t[4], null, scalar(t[2]), scalar(t[3])) else Prop(t[2], scalar(t[1]))
                )
            }
        }
        val vertexIndex = elements.indexOfFirst { it.name == "vertex" }
        require(vertexIndex >= 0) { "${file.name}: PLY has no vertex element" }
        val vertex = elements[vertexIndex]
        val ix = vertex.props.indexOfFirst { it.name == "x" }
        val iy = vertex.props.indexOfFirst { it.name == "y" }
        val iz = vertex.props.indexOfFirst { it.name == "z" }
        require(ix >= 0 && iy >= 0 && iz >= 0) { "${file.name}: PLY vertex lacks x/y/z" }
        val builder = Builder(vertex.count.coerceAtMost(50_000_000).toInt())
        if (format == "ascii") {
            val reader = input.bufferedReader()
            // Skip earlier elements (rare), one line per item.
            for (e in elements.subList(0, vertexIndex)) repeat(e.count.toInt()) { reader.readLine() }
            repeat(vertex.count.toInt()) {
                val t = reader.readLine()?.trim()?.split(Regex("\\s+")) ?: return@repeat
                builder.add(t[ix].toDouble(), t[iy].toDouble(), t[iz].toDouble())
            }
        } else {
            val order = if (format == "binary_big_endian") ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
            require(vertexIndex == 0 || elements.subList(0, vertexIndex).all { e -> e.props.all { it.type != null } }) {
                "${file.name}: binary PLY with list properties before the vertices is not supported"
            }
            for (e in elements.subList(0, vertexIndex)) input.skipNBytes(e.count * e.props.sumOf { it.type!!.bytes })
            require(vertex.props.all { it.type != null }) { "${file.name}: list properties on vertices" }
            val offsets = vertex.props.runningFold(0) { acc, p -> acc + p.type!!.bytes }
            val stride = offsets.last()
            val record = ByteArray(stride)
            val rb = ByteBuffer.wrap(record).order(order)
            val types = vertex.props.map { it.type!! }
            for (i in 0 until vertex.count) {
                if (input.readNBytes(record, 0, stride) < stride) break
                builder.add(rb.read(types[ix], offsets[ix]), rb.read(types[iy], offsets[iy]), rb.read(types[iz], offsets[iz]))
            }
        }
        builder.build(file.name)
    }

    // ---------------------------------------------------------------- PCD
    /** ASCII and binary (uncompressed) PCD, fields x y z. */
    fun readPcd(file: File): PointCloud = BufferedInputStream(file.inputStream(), 1 shl 20).use { input ->
        val header = readHeaderLines(input) { it.uppercase().startsWith("DATA") }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
            .associate { line -> line.split(Regex("\\s+")).let { it[0].uppercase() to it.drop(1) } }
        val fields = header.getValue("FIELDS")
        val sizes = header.getValue("SIZE").map { it.toInt() }
        val types = header.getValue("TYPE")
        val counts = header["COUNT"]?.map { it.toInt() } ?: List(fields.size) { 1 }
        val n = header.getValue("POINTS")[0].toLong()
        val mode = header.getValue("DATA")[0].lowercase()
        val ix = fields.indexOf("x")
        val iy = fields.indexOf("y")
        val iz = fields.indexOf("z")
        require(ix >= 0 && iy >= 0 && iz >= 0) { "${file.name}: PCD lacks x/y/z fields" }
        val builder = Builder(n.coerceAtMost(50_000_000).toInt())
        when (mode) {
            "ascii" -> {
                val col = counts.runningFold(0) { a, c -> a + c }
                input.bufferedReader().lineSequence().forEach { line ->
                    val t = line.trim().split(Regex("\\s+"))
                    if (t.size >= col.last()) builder.add(t[col[ix]].toDouble(), t[col[iy]].toDouble(), t[col[iz]].toDouble())
                }
            }
            "binary" -> {
                val bytes = sizes.zip(counts) { s, c -> s * c }
                val offsets = bytes.runningFold(0) { a, b -> a + b }
                val record = ByteArray(offsets.last())
                val rb = ByteBuffer.wrap(record).order(ByteOrder.LITTLE_ENDIAN)
                fun field(i: Int): Double {
                    val kind = when (types[i]) { "F" -> if (sizes[i] == 8) Scalar.F64 else Scalar.F32
                        "U" -> mapOf(1 to Scalar.U8, 2 to Scalar.U16, 4 to Scalar.U32).getValue(sizes[i])
                        else -> mapOf(1 to Scalar.I8, 2 to Scalar.I16, 4 to Scalar.I32).getValue(sizes[i]) }
                    return rb.read(kind, offsets[i])
                }
                for (i in 0 until n) {
                    if (input.readNBytes(record, 0, record.size) < record.size) break
                    builder.add(field(ix), field(iy), field(iz))
                }
            }
            else -> throw IllegalArgumentException("${file.name}: PCD DATA '$mode' not supported (use ascii or binary)")
        }
        builder.build(file.name)
    }

    // ---------------------------------------------------------------- NumPy
    /** `.npy` holding an (N, >=3) float32/float64/int array, C order, little endian. */
    fun readNpy(input: InputStream, source: String): PointCloud {
        val data = DataInputStream(BufferedInputStream(input, 1 shl 20))
        val magic = ByteArray(6).also { data.readFully(it) }
        require(magic[0] == 0x93.toByte() && String(magic, 1, 5, Charsets.US_ASCII) == "NUMPY") { "$source: not a .npy file" }
        val major = data.readUnsignedByte()
        data.readUnsignedByte()  // minor version
        // Header length: little-endian uint16 in format 1.x, uint32 from 2.0 on.
        val lenBytes = ByteArray(if (major == 1) 2 else 4).also { data.readFully(it) }
        val headerLen = lenBytes.foldIndexed(0) { i, acc, b -> acc or ((b.toInt() and 0xFF) shl (8 * i)) }
        val header = ByteArray(headerLen).also { data.readFully(it) }.toString(Charsets.ISO_8859_1)
        val descr = Regex("'descr':\\s*'([^']+)'").find(header)?.groupValues?.get(1) ?: error("$source: no dtype")
        val fortran = Regex("'fortran_order':\\s*(True|False)").find(header)?.groupValues?.get(1) == "True"
        val shape = Regex("'shape':\\s*\\(([^)]*)\\)").find(header)!!.groupValues[1]
            .split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toLong() }
        require(shape.size == 2 && shape[1] >= 3) { "$source: expected an (N, >=3) array, got shape $shape" }
        require(!fortran) { "$source: Fortran-ordered arrays are not supported" }
        val order = if (descr.startsWith(">")) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
        val kind = descr.trimStart('<', '>', '|', '=')
        val itemSize = kind.drop(1).toInt()
        val type = when (kind) {
            "f4" -> Scalar.F32; "f8" -> Scalar.F64; "i4" -> Scalar.I32; "i2" -> Scalar.I16; "u2" -> Scalar.U16
            "u4" -> Scalar.U32; "i1" -> Scalar.I8; "u1" -> Scalar.U8
            else -> throw IllegalArgumentException("$source: unsupported dtype $descr")
        }
        val cols = shape[1].toInt()
        val row = ByteArray(cols * itemSize)
        val rb = ByteBuffer.wrap(row).order(order)
        val builder = Builder(shape[0].coerceAtMost(50_000_000).toInt())
        for (i in 0 until shape[0]) {
            data.readFully(row)
            builder.add(rb.read(type, 0), rb.read(type, itemSize), rb.read(type, 2 * itemSize))
        }
        return builder.build(source)
    }

    /** `.npz`: the array named points / xyz / pos / coord, else the first one. */
    fun readNpz(file: File): PointCloud = ZipFile(file).use { zip ->
        val names = zip.entries().asSequence().map { it.name }.filter { it.endsWith(".npy") }.toList()
        val pick = listOf("points", "xyz", "pos", "coord").map { "$it.npy" }.firstOrNull { it in names }
            ?: names.firstOrNull() ?: throw IllegalArgumentException("${file.name}: empty .npz")
        zip.getInputStream(zip.getEntry(pick)).use { readNpy(it, "${file.name}:$pick") }
    }

    // ---------------------------------------------------------------- text
    /** Whitespace-separated text (commas too for .csv); the first three numeric columns of each line. */
    fun readText(file: File): PointCloud {
        val csv = extension(file.name) == "csv"
        val builder = Builder()
        file.bufferedReader().useLines { lines ->
            for (line in lines) {
                val parts = (if (csv) line.replace(',', ' ') else line).trim().split(Regex("\\s+"))
                if (parts.size < 3) continue
                val x = parts[0].toDoubleOrNull() ?: continue
                val y = parts[1].toDoubleOrNull() ?: continue
                val z = parts[2].toDoubleOrNull() ?: continue
                builder.add(x, y, z)
            }
        }
        return builder.build(file.name)
    }
}
