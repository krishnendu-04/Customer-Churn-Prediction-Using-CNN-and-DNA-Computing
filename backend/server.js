require("dotenv").config();
const express  = require("express");
const mongoose = require("mongoose");
const cors     = require("cors");
const multer   = require("multer");
const { GridFSBucket } = require("mongodb");
const Upload   = require("./models/Upload");

const upload = multer({ storage: multer.memoryStorage() });

const app  = express();
const PORT = process.env.PORT || 5000;

/* ── Middleware ─────────────────────────────────────────────────────────────── */
app.use(cors({
  origin: [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
  ],
  methods: ["GET", "POST", "DELETE"],
  allowedHeaders: ["Content-Type"],
}));
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

/* ── MongoDB connection ─────────────────────────────────────────────────────── */
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log(`✓ MongoDB connected → ${process.env.MONGO_URI}`))
  .catch(err => { console.error("✗ MongoDB connection failed:", err.message); process.exit(1); });

/** Returns a GridFSBucket (only call after connection is ready) */
function getBucket() {
  return new GridFSBucket(mongoose.connection.db, { bucketName: "csvfiles" });
}

/* ── Routes ─────────────────────────────────────────────────────────────────── */

/**
 * POST /api/upload-file
 * Multipart form-data fields:
 *   csv       — the raw .csv file
 *   userName  — string
 *   userEmail — string (required)
 *   mapping   — JSON string
 *   stats     — JSON string
 *
 * Stores the CSV in GridFS and saves metadata to the Upload collection.
 */
app.post("/api/upload-file", upload.single("csv"), async (req, res) => {
  try {
    const { userName, userEmail, mapping, stats } = req.body;

    if (!userEmail || !req.file) {
      return res.status(400).json({ error: "userEmail and csv file are required." });
    }

    const bucket = getBucket();

    // Stream buffer into GridFS
    const uploadStream = bucket.openUploadStream(req.file.originalname, {
      contentType: "text/csv",
      metadata: {
        userEmail: userEmail.toLowerCase().trim(),
        userName:  userName || "Unknown",
      },
    });
    uploadStream.end(req.file.buffer);

    const gridfsId = await new Promise((resolve, reject) => {
      uploadStream.on("finish", () => resolve(uploadStream.id));
      uploadStream.on("error",  reject);
    });

    // Parse JSON form fields
    let parsedMapping = {}, parsedStats = {};
    try { parsedMapping = JSON.parse(mapping || "{}"); } catch {}
    try { parsedStats   = JSON.parse(stats   || "{}"); } catch {}

    // Save metadata document
    const doc = await Upload.create({
      userName:  userName || "Unknown",
      userEmail: userEmail.toLowerCase().trim(),
      fileName:  req.file.originalname,
      rowCount:  parsedStats.total || 0,
      gridfsId:  gridfsId.toString(),
      mapping:   parsedMapping,
      stats:     parsedStats,
    });

    console.log(`✓ Saved [${doc._id}] — ${req.file.originalname} by ${userEmail} | GridFS: ${gridfsId}`);
    res.status(201).json({ success: true, id: doc._id, gridfsId: gridfsId.toString() });
  } catch (err) {
    console.error("✗ Upload save error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/uploads?email=user@example.com
 * Returns all upload metadata for a user, newest first.
 */
app.get("/api/uploads", async (req, res) => {
  try {
    const { email } = req.query;
    const filter = email ? { userEmail: email.toLowerCase().trim() } : {};
    const docs = await Upload.find(filter).sort({ uploadedAt: -1 });
    res.json({ uploads: docs });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/uploads/:id
 * Returns the upload metadata document.
 * Use GET /api/file/:gridfsId to download the raw CSV.
 */
app.get("/api/uploads/:id", async (req, res) => {
  try {
    const doc = await Upload.findById(req.params.id);
    if (!doc) return res.status(404).json({ error: "Not found." });
    res.json(doc.toObject());
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/file/:gridfsId
 * Streams the raw CSV file from GridFS.
 */
app.get("/api/file/:gridfsId", async (req, res) => {
  try {
    const bucket = getBucket();
    const id = new mongoose.Types.ObjectId(req.params.gridfsId);
    const files = await bucket.find({ _id: id }).toArray();
    if (!files.length) return res.status(404).json({ error: "File not found." });

    res.setHeader("Content-Type", "text/csv");
    res.setHeader("Content-Disposition", `attachment; filename="${files[0].filename}"`);
    bucket.openDownloadStream(id).pipe(res);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * DELETE /api/uploads/:id
 * Deletes the metadata doc AND the GridFS file.
 */
app.delete("/api/uploads/:id", async (req, res) => {
  try {
    const doc = await Upload.findByIdAndDelete(req.params.id);
    if (doc?.gridfsId) {
      try {
        const bucket = getBucket();
        await bucket.delete(new mongoose.Types.ObjectId(doc.gridfsId));
      } catch (e) {
        console.warn("GridFS delete warning:", e.message);
      }
    }
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ── Admin routes ───────────────────────────────────────────────────────────── */

/**
 * GET /api/admin/stats
 */
app.get("/api/admin/stats", async (req, res) => {
  try {
    const [totalUploads, rowsAgg, uniqueUsers, riskAgg] = await Promise.all([
      Upload.countDocuments(),
      Upload.aggregate([{ $group: { _id: null, total: { $sum: "$rowCount" } } }]),
      Upload.distinct("userEmail"),
      Upload.aggregate([{ $group: {
        _id: null,
        high:   { $sum: "$stats.high" },
        medium: { $sum: "$stats.medium" },
        safe:   { $sum: "$stats.safe" },
      }}]),
    ]);
    const risk = riskAgg[0] || { high: 0, medium: 0, safe: 0 };
    res.json({
      totalUploads,
      totalRows:   rowsAgg[0]?.total || 0,
      uniqueUsers: uniqueUsers.length,
      high:   risk.high   || 0,
      medium: risk.medium || 0,
      safe:   risk.safe   || 0,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/admin/uploads?page=0&limit=50&email=filter
 */
app.get("/api/admin/uploads", async (req, res) => {
  try {
    const page  = Math.max(0, parseInt(req.query.page  || "0"));
    const limit = Math.min(100, parseInt(req.query.limit || "50"));
    const filter = req.query.email
      ? { userEmail: { $regex: req.query.email, $options: "i" } }
      : {};

    const [docs, total] = await Promise.all([
      Upload.find(filter).sort({ uploadedAt: -1 }).skip(page * limit).limit(limit),
      Upload.countDocuments(filter),
    ]);
    res.json({ uploads: docs, total, page, limit });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ── Health check ───────────────────────────────────────────────────────────── */
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", db: mongoose.connection.readyState === 1 ? "connected" : "disconnected" });
});

/* ── Global error handler ───────────────────────────────────────────────────── */
app.use((err, _req, res, _next) => {
  if (err.type === "entity.too.large") {
    return res.status(413).json({ error: "Request payload too large." });
  }
  console.error("✗ Unhandled error:", err.message);
  res.status(500).json({ error: err.message });
});

/* ── Start ──────────────────────────────────────────────────────────────────── */
const { execSync } = require("child_process");

function startServer(retrying = false) {
  const server = app.listen(PORT, () => {
    console.log(`✓ Churn backend listening on http://localhost:${PORT}`);
  });

  server.on("error", (err) => {
    if (err.code === "EADDRINUSE") {
      if (retrying) { console.error(`✗ Port ${PORT} still in use. Exiting.`); process.exit(1); }
      console.warn(`⚠ Port ${PORT} in use — killing old process and retrying…`);
      try {
        const result = execSync(`netstat -ano | findstr :${PORT} | findstr LISTENING`, { encoding: "utf8" });
        result.trim().split("\n").forEach(line => {
          const pid = line.trim().split(/\s+/).pop();
          if (pid && pid !== "0") try { execSync(`taskkill /PID ${pid} /F`); } catch {}
        });
      } catch {}
      setTimeout(() => startServer(true), 1000);
    } else {
      console.error("✗ Server error:", err.message);
      process.exit(1);
    }
  });
}

startServer();
